#!/usr/bin/env python3
"""
Hybrid stack smoke test for AMD Ryzen AI (CPU + iGPU Vulkan + XDNA NPU).

Steps:
  1) Hardware probe
  2) Fetch llama.cpp Ubuntu Vulkan build (cached)
  3) Fetch Qwen2.5-0.5B-Instruct Q4_K_M GGUF (cached)
  4) Run a short Vulkan completion with GPU layer offload
  5) Optional FLM NPU completion if `flm` is installed
  6) Write reports/latest.json and exit non-zero on hard failures

Safety (important on 512 MiB iGPUs shared with the desktop / Cursor):
  - Single-instance lock — concurrent smokes cannot start
  - Subprocess output is streamed/truncated; never unbounded capture into RAM
  - VRAM preflight under /sys/class/drm/*/device/mem_info_vram_*
  - Conservative default -ngl for small iGPUs (override with --ngl)
  - Do not run the Vulkan GPU smoke while Cursor (or other GPU apps) already
    saturate the same tiny VRAM pool — that combination has OOMed the host
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
CACHE = ROOT / ".cache"
BIN_DIR = CACHE / "llama-vulkan"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
LOCK_PATH = CACHE / "smoke_test.lock"

# Pinned for reproducibility. Override with --llama-tag.
DEFAULT_LLAMA_TAG = "b8966"
LLAMA_ASSET = "llama-{tag}-bin-ubuntu-vulkan-x64.tar.gz"
LLAMA_URL = "https://github.com/ggml-org/llama.cpp/releases/download/{tag}/{asset}"

MODEL_REPO = "Qwen/Qwen2.5-0.5B-Instruct-GGUF"
MODEL_FILE = "qwen2.5-0.5b-instruct-q4_k_m.gguf"
MODEL_URL = (
    f"https://huggingface.co/{MODEL_REPO}/resolve/main/{MODEL_FILE}"
)

PROMPT = "Reply with exactly: hybrid-smoke-ok"

# Bound how much subprocess text we keep in process memory.
MAX_CAPTURE_BYTES = 256 * 1024
# Shorter than a full interactive session; hung llama-cli must not linger.
DEFAULT_LLM_TIMEOUT_SEC = 120
DEFAULT_CMD_TIMEOUT_SEC = 60
DEFAULT_NPU_TIMEOUT_SEC = 180
# Refuse GPU smoke when free VRAM is below this (MiB). Cursor + Vulkan share
# the same 512 MiB iGPU pool on this class of hardware.
MIN_FREE_VRAM_MIB = 128
# "auto" ngl caps for small discrete/shared heaps.
NGL_AUTO_TINY_MIB = 768  # <= this → very conservative offload
NGL_AUTO_SMALL_MIB = 2048

# Preferred FLM NPU smoke model (small XDNA2 artifact).
DEFAULT_FLM_MODEL = "qwen3:0.6b"
FLM_SERVE_HOST = "127.0.0.1"
FLM_SERVE_PORT = 8099


@dataclass
class BoundedResult:
    returncode: int
    stdout: str
    stderr: str
    stdout_truncated: bool = False
    stderr_truncated: bool = False


def log(msg: str) -> None:
    print(msg, flush=True)


def _read_bounded(stream, max_bytes: int, sink: list[str], flags: list[bool]) -> None:
    """Drain a text stream, retaining at most max_bytes (rest discarded)."""
    total = 0
    truncated = False
    try:
        while True:
            chunk = stream.read(4096)
            if not chunk:
                break
            if total >= max_bytes:
                truncated = True
                continue
            remain = max_bytes - total
            if len(chunk) <= remain:
                sink.append(chunk)
                total += len(chunk)
            else:
                sink.append(chunk[:remain])
                total += remain
                truncated = True
    finally:
        flags[0] = truncated


def run(
    cmd: list[str],
    *,
    timeout: int | None = None,
    env: dict | None = None,
    cwd: Path | None = None,
    max_capture: int = MAX_CAPTURE_BYTES,
) -> BoundedResult:
    """Run a command with bounded stdout/stderr retention (no unbounded RAM)."""
    log(f"+ {' '.join(cmd)}")
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
            cwd=str(cwd) if cwd else None,
        )
    except FileNotFoundError:
        return BoundedResult(127, "", f"not found: {cmd[0]}")

    out_parts: list[str] = []
    err_parts: list[str] = []
    out_flags = [False]
    err_flags = [False]
    t_out = threading.Thread(
        target=_read_bounded, args=(proc.stdout, max_capture, out_parts, out_flags), daemon=True
    )
    t_err = threading.Thread(
        target=_read_bounded, args=(proc.stderr, max_capture, err_parts, err_flags), daemon=True
    )
    t_out.start()
    t_err.start()
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
        t_out.join(timeout=2)
        t_err.join(timeout=2)
        return BoundedResult(
            124,
            "".join(out_parts),
            "".join(err_parts) + "\n[timeout]",
            stdout_truncated=out_flags[0],
            stderr_truncated=True,
        )
    t_out.join(timeout=5)
    t_err.join(timeout=5)
    return BoundedResult(
        proc.returncode if proc.returncode is not None else 1,
        "".join(out_parts),
        "".join(err_parts),
        stdout_truncated=out_flags[0],
        stderr_truncated=err_flags[0],
    )


def run_to_files(
    cmd: list[str],
    *,
    stdout_path: Path,
    stderr_path: Path,
    timeout: int,
    env: dict | None = None,
) -> int:
    """Run a command writing stdout/stderr directly to files (no RAM spool)."""
    log(f"+ {' '.join(cmd)}")
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    with open(stdout_path, "wb") as out_f, open(stderr_path, "wb") as err_f:
        try:
            proc = subprocess.Popen(cmd, stdout=out_f, stderr=err_f, env=env)
            return proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
            return 124
        except FileNotFoundError:
            err_f.write(f"not found: {cmd[0]}\n".encode())
            return 127


def read_tail(path: Path, max_bytes: int = 8000) -> str:
    if not path.exists():
        return ""
    data = path.read_bytes()
    if len(data) > max_bytes:
        data = data[-max_bytes:]
    return data.decode("utf-8", errors="replace")


class SmokeLock:
    """Exclusive flock so two smoke runs cannot stack llama-cli / VRAM pressure."""

    def __init__(self, path: Path = LOCK_PATH) -> None:
        self.path = path
        self._fh = None

    def __enter__(self) -> SmokeLock:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "w", encoding="utf-8")
        try:
            fcntl.flock(self._fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            holder = ""
            try:
                holder = self.path.read_text(encoding="utf-8", errors="replace").strip()
            except OSError:
                pass
            self._fh.close()
            self._fh = None
            msg = "Another smoke_test.py instance is already running"
            if holder:
                msg += f" (lock holder: {holder})"
            raise RuntimeError(msg) from exc
        self._fh.write(f"pid={os.getpid()} started={time.strftime('%Y-%m-%dT%H:%M:%S')}\n")
        self._fh.flush()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._fh is not None:
            try:
                fcntl.flock(self._fh.fileno(), fcntl.LOCK_UN)
            finally:
                self._fh.close()
                self._fh = None


def download(url: str, dest: Path, *, force: bool = False) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0 and not force:
        log(f"cache hit: {dest}")
        return

    tmp = dest.with_suffix(dest.suffix + ".partial")
    log(f"downloading {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "hybrid-smoke/1.0"})
    with urllib.request.urlopen(req, timeout=600) as resp, open(tmp, "wb") as out:
        shutil.copyfileobj(resp, out)
    tmp.replace(dest)
    log(f"saved {dest} ({dest.stat().st_size} bytes)")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_llama(tag: str, force: bool = False) -> Path:
    asset = LLAMA_ASSET.format(tag=tag)
    tarball = CACHE / "downloads" / asset
    download(LLAMA_URL.format(tag=tag, asset=asset), tarball, force=force)

    marker = BIN_DIR / f".extracted-{tag}"
    llama_bin = BIN_DIR / "llama-cli"
    if marker.exists() and llama_bin.exists() and not force:
        return llama_bin

    if BIN_DIR.exists() and force:
        shutil.rmtree(BIN_DIR)
    BIN_DIR.mkdir(parents=True, exist_ok=True)

    with tarfile.open(tarball, "r:gz") as tar:
        tar.extractall(BIN_DIR)

    # Release layouts vary: binary may be top-level or under a nested dir.
    candidates = list(BIN_DIR.rglob("llama-cli"))
    if not candidates:
        # Newer builds may ship llama-cli under build/bin-like paths or as llama
        candidates = list(BIN_DIR.rglob("llama"))
        candidates = [c for c in candidates if c.is_file() and os.access(c, os.X_OK)]
    if not candidates:
        raise RuntimeError(f"llama-cli not found after extracting {tarball}")

    chosen = candidates[0]
    if chosen != llama_bin:
        if llama_bin.exists() or llama_bin.is_symlink():
            llama_bin.unlink()
        try:
            llama_bin.symlink_to(chosen.resolve())
        except OSError:
            shutil.copy2(chosen, llama_bin)
            llama_bin.chmod(0o755)

    marker.write_text(tag + "\n", encoding="utf-8")
    return llama_bin


def ensure_model(force: bool = False) -> Path:
    dest = MODELS / MODEL_FILE
    download(MODEL_URL, dest, force=force)
    if dest.stat().st_size < 1_000_000:
        raise RuntimeError(f"model file looks too small: {dest} ({dest.stat().st_size} bytes)")
    return dest


def read_vram() -> dict:
    """Read AMDGPU VRAM totals from sysfs (card*/renderD* share the same device)."""
    best: dict | None = None
    drm = Path("/sys/class/drm")
    if not drm.is_dir():
        return {"available": False, "error": "no /sys/class/drm"}

    seen_totals: set[tuple[int, int]] = set()
    devices: list[dict] = []
    for card in sorted(drm.glob("card*")):
        total_p = card / "device" / "mem_info_vram_total"
        used_p = card / "device" / "mem_info_vram_used"
        if not total_p.exists() or not used_p.exists():
            continue
        try:
            total = int(total_p.read_text().strip())
            used = int(used_p.read_text().strip())
        except (OSError, ValueError):
            continue
        key = (total, used)
        if key in seen_totals:
            continue
        seen_totals.add(key)
        free = max(total - used, 0)
        entry = {
            "path": str(card),
            "total_bytes": total,
            "used_bytes": used,
            "free_bytes": free,
            "total_mib": round(total / (1024 * 1024), 1),
            "used_mib": round(used / (1024 * 1024), 1),
            "free_mib": round(free / (1024 * 1024), 1),
        }
        devices.append(entry)
        if best is None or total > best["total_bytes"]:
            best = entry

    if best is None:
        return {"available": False, "error": "no mem_info_vram_* nodes", "devices": []}
    return {
        "available": True,
        "devices": devices,
        **best,
    }


def choose_default_ngl(vram: dict) -> int:
    """Conservative GPU layer offload for small shared iGPU heaps."""
    if not vram.get("available"):
        return 4
    total_mib = float(vram.get("total_mib") or 0)
    if total_mib <= NGL_AUTO_TINY_MIB:
        # 512 MiB Ryzen AI iGPU — ngl=99 nearly fills the heap alone.
        return 8
    if total_mib <= NGL_AUTO_SMALL_MIB:
        return 16
    return 32


def vram_preflight(vram: dict, *, min_free_mib: int, force: bool) -> dict:
    """Refuse GPU smoke when the shared iGPU heap is already nearly full."""
    result = {
        "ok": True,
        "forced": bool(force),
        "min_free_mib": min_free_mib,
        "vram": vram,
        "warning": None,
    }
    note = (
        "Cursor (Electron/GPU) and this Vulkan smoke share the same tiny iGPU "
        "VRAM pool. Close Cursor GPU clients or use --skip-llm / --ngl 0 if "
        "you only need a hardware probe."
    )
    if not vram.get("available"):
        result["ok"] = force
        result["warning"] = f"VRAM sysfs unavailable; {note}"
        if not force:
            result["error"] = "cannot verify free VRAM (pass --force-gpu to override)"
        return result

    free_mib = float(vram.get("free_mib") or 0)
    used_mib = float(vram.get("used_mib") or 0)
    total_mib = float(vram.get("total_mib") or 0)
    log(
        f"VRAM preflight: {used_mib:.0f}/{total_mib:.0f} MiB used "
        f"({free_mib:.0f} MiB free); need >= {min_free_mib} MiB free"
    )
    if free_mib < min_free_mib:
        result["warning"] = note
        if force:
            log(f"WARNING: free VRAM {free_mib:.0f} MiB < {min_free_mib}; continuing (--force-gpu)")
            result["ok"] = True
        else:
            result["ok"] = False
            result["error"] = (
                f"insufficient free VRAM ({free_mib:.0f} MiB free, need "
                f">= {min_free_mib}). {note} Pass --force-gpu to override."
            )
    else:
        result["warning"] = (
            "Reminder: Cursor and Vulkan smoke share this iGPU VRAM — "
            "avoid running the GPU smoke alongside a busy Cursor session."
        )
    return result


def probe_hw() -> dict:
    script = SCRIPTS / "check_hw.py"
    p = run([sys.executable, str(script)], timeout=DEFAULT_CMD_TIMEOUT_SEC)
    if p.returncode != 0 and not p.stdout.strip():
        raise RuntimeError(f"hardware probe failed: {p.stderr}")
    data = json.loads(p.stdout)
    data["_probe_exit"] = p.returncode
    return data


def parse_speed(text: str) -> dict:
    # Formats seen:
    # - "eval time = ... / ... tokens per second"
    # - "[ Prompt: 674.8 t/s | Generation: 171.3 t/s ]"
    out = {}
    m = re.search(r"Generation:\s*([\d.]+)\s*t/s", text, re.I)
    if m:
        out["tokens_per_second"] = float(m.group(1))
    else:
        m = re.search(r"([\d.]+)\s*tokens\s*per\s*second", text, re.I)
        if m:
            out["tokens_per_second"] = float(m.group(1))
        else:
            m = re.search(r"([\d.]+)\s*tok/s", text, re.I)
            if m:
                out["tokens_per_second"] = float(m.group(1))
    m = re.search(r"Prompt:\s*([\d.]+)\s*t/s", text, re.I)
    if m:
        out["prompt_tokens_per_second"] = float(m.group(1))
    m = re.search(r"prompt\s+eval\s+time\s*=\s*([^\n]+)", text, re.I)
    if m:
        out["prompt_eval"] = m.group(1).strip()
    m = re.search(r"(?<![a-z])eval\s+time\s*=\s*([^\n]+)", text, re.I)
    if m:
        out["eval"] = m.group(1).strip()
    return out


def _llama_cmd(llama_bin: Path, model: Path, ngl: int, n_predict: int, log_file: Path) -> list[str]:
    # Force non-interactive completion; chat templates otherwise open conversation mode.
    return [
        str(llama_bin),
        "-m",
        str(model),
        "-p",
        PROMPT,
        "-n",
        str(n_predict),
        "-ngl",
        str(ngl),
        "--no-conversation",
        "--single-turn",
        "--temp",
        "0",
        "--no-display-prompt",
        "--log-file",
        str(log_file),
    ]


def run_vulkan_llm(
    llama_bin: Path,
    model: Path,
    ngl: int,
    *,
    timeout: int = DEFAULT_LLM_TIMEOUT_SEC,
) -> dict:
    env = os.environ.copy()
    # Ensure bundled ggml/*.so (especially libggml-vulkan.so) resolve next to llama-cli.
    lib_dir = str(llama_bin.resolve().parent)
    env["LD_LIBRARY_PATH"] = lib_dir + (":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")

    REPORTS.mkdir(parents=True, exist_ok=True)
    log_file = REPORTS / "llama-vulkan.log"
    stdout_path = REPORTS / "llama-vulkan.stdout"
    stderr_path = REPORTS / "llama-vulkan.stderr"
    for path in (log_file, stdout_path, stderr_path):
        if path.exists():
            path.unlink()

    cmd = _llama_cmd(llama_bin, model, ngl, 24, log_file)
    started = time.time()
    # Stream to disk — never capture_output into an unbounded Python buffer.
    rc = run_to_files(
        cmd,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        timeout=timeout,
        env=env,
    )
    elapsed = time.time() - started
    stdout_text = read_tail(stdout_path, 8000)
    stderr_text = read_tail(stderr_path, 8000)
    log_text = read_tail(log_file, 12000)
    combined = "\n".join([stdout_text, stderr_text, log_text])

    generation = stdout_text.strip()
    ok = rc == 0 and len(generation) > 0

    used_gpu = bool(
        re.search(r"loaded Vulkan backend", combined, re.I)
        or re.search(r"Vulkan0", combined, re.I)
        or re.search(r"offloaded\s+\d+/\d+\s+layers\s+to\s+GPU", combined, re.I)
        or re.search(r"RADV|Radeon|GGML_VK|ggml_vulkan", combined, re.I)
    )

    result = {
        "backend": "llama.cpp-vulkan",
        "model": MODEL_FILE,
        "ngl": ngl,
        "exit_code": rc,
        "elapsed_sec": round(elapsed, 3),
        "timeout_sec": timeout,
        "ok": bool(ok),
        "gpu_offload_detected": bool(used_gpu),
        "generation": generation[-500:],
        "speed": parse_speed(combined),
        "stdout_tail": stdout_text[-2000:],
        "stderr_tail": stderr_text[-2000:],
        "log_tail": log_text[-3000:],
        "output_files": {
            "stdout": str(stdout_path),
            "stderr": str(stderr_path),
            "log": str(log_file),
        },
    }

    if rc != 0:
        log("Vulkan/GPU run failed; retrying CPU-only (ngl=0) for diagnosis")
        cpu_log = REPORTS / "llama-cpu-fallback.log"
        cpu_out = REPORTS / "llama-cpu-fallback.stdout"
        cpu_err = REPORTS / "llama-cpu-fallback.stderr"
        for path in (cpu_log, cpu_out, cpu_err):
            if path.exists():
                path.unlink()
        rc2 = run_to_files(
            _llama_cmd(llama_bin, model, 0, 16, cpu_log),
            stdout_path=cpu_out,
            stderr_path=cpu_err,
            timeout=timeout,
            env=env,
        )
        out2 = read_tail(cpu_out, 4000)
        err2 = read_tail(cpu_err, 4000)
        log2 = read_tail(cpu_log, 8000)
        combined2 = "\n".join([out2, err2, log2])
        result["cpu_fallback"] = {
            "exit_code": rc2,
            "ok": rc2 == 0 and len(out2.strip()) > 0,
            "speed": parse_speed(combined2),
            "stdout_tail": out2[-1500:],
            "stderr_tail": err2[-1500:],
        }
    return result


def find_flm() -> str | None:
    """Locate flm, including common user-local portable installs."""
    found = shutil.which("flm")
    if found:
        return found
    candidates = [
        Path.home() / ".local/bin/flm",
        Path.home() / ".local/share/fastflowlm/flm",
    ]
    for path in candidates:
        if path.is_file() and os.access(path, os.X_OK):
            return str(path)
    return None


def _flm_installed_models(flm: str) -> list[str]:
    p = run([flm, "list", "-j", "--filter", "installed"], timeout=DEFAULT_CMD_TIMEOUT_SEC)
    if p.returncode != 0 or not (p.stdout or "").strip():
        # Fallback: parse plain list for checkmarks / local paths.
        p2 = run([flm, "list", "--filter", "installed"], timeout=DEFAULT_CMD_TIMEOUT_SEC)
        text = (p2.stdout or "") + "\n" + (p2.stderr or "")
        return re.findall(r"^\s*-\s*([A-Za-z0-9._:-]+)", text, re.M)

    try:
        data = json.loads(p.stdout)
    except json.JSONDecodeError:
        return []

    models = data.get("models", data if isinstance(data, list) else [])
    out: list[str] = []
    for item in models:
        if not isinstance(item, dict):
            continue
        if item.get("installed") is False:
            continue
        name = item.get("model") or item.get("name") or item.get("id")
        if name:
            out.append(str(name))
    return out


def _wait_http_ok(url: str, timeout_sec: float = 60.0) -> bool:
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:
                if 200 <= resp.status < 300:
                    return True
        except Exception:  # noqa: BLE001 - probe loop
            time.sleep(0.5)
    return False


def run_npu_optional(model: str = DEFAULT_FLM_MODEL) -> dict:
    """Run a short NPU completion via FastFlowLM OpenAI-compatible serve API."""
    flm = find_flm()
    if not flm:
        return {
            "backend": "flm",
            "skipped": True,
            "reason": "flm not installed (install portable FastFlowLM to ~/.local)",
            "ok": True,  # soft skip
        }

    p_val = run([flm, "validate"], timeout=DEFAULT_CMD_TIMEOUT_SEC)
    validate_ok = p_val.returncode == 0 and "/dev/accel" in ((p_val.stdout or "") + (p_val.stderr or ""))
    installed = _flm_installed_models(flm)
    if model not in installed:
        # Prefer the requested tiny model; otherwise first installed.
        if installed:
            model = installed[0]
        else:
            log(f"pulling FLM model {DEFAULT_FLM_MODEL}")
            p_pull = run(
                [flm, "pull", DEFAULT_FLM_MODEL],
                timeout=600,
            )
            if p_pull.returncode != 0:
                return {
                    "backend": "flm",
                    "skipped": True,
                    "reason": f"flm pull failed for {DEFAULT_FLM_MODEL}",
                    "ok": True,
                    "flm_path": flm,
                    "validate_ok": validate_ok,
                    "pull_stderr_tail": (p_pull.stderr or "")[-1500:],
                }
            model = DEFAULT_FLM_MODEL

    REPORTS.mkdir(parents=True, exist_ok=True)
    serve_log = REPORTS / "flm-serve.log"
    port = FLM_SERVE_PORT
    base = f"http://{FLM_SERVE_HOST}:{port}"
    serve_cmd = [
        flm,
        "serve",
        model,
        "--host",
        FLM_SERVE_HOST,
        "-p",
        str(port),
        "--quiet",
    ]
    log(f"+ {' '.join(serve_cmd)}")
    with open(serve_log, "wb") as logf:
        proc = subprocess.Popen(
            serve_cmd,
            stdout=logf,
            stderr=subprocess.STDOUT,
            text=False,
        )
        result = {
            "backend": "flm",
            "skipped": False,
            "flm_path": flm,
            "model": model,
            "validate_ok": validate_ok,
            "serve_port": port,
            "serve_log": str(serve_log),
            "ok": False,
        }
        try:
            if not _wait_http_ok(f"{base}/v1/models", timeout_sec=90):
                result["reason"] = "flm serve did not become ready"
                result["serve_tail"] = serve_log.read_text(encoding="utf-8", errors="replace")[-2000:]
                return result

            payload = json.dumps(
                {
                    "model": model,
                    "messages": [{"role": "user", "content": PROMPT}],
                    "max_tokens": 24,
                    "temperature": 0,
                }
            ).encode("utf-8")
            req = urllib.request.Request(
                f"{base}/v1/chat/completions",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            started = time.time()
            with urllib.request.urlopen(req, timeout=DEFAULT_NPU_TIMEOUT_SEC) as resp:
                body = resp.read().decode("utf-8", errors="replace")
                status = resp.status
            elapsed = time.time() - started
            data = json.loads(body)
            content = (
                data.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
            )
            usage = data.get("usage") or {}
            ok = status == 200 and bool(content.strip())
            result.update(
                {
                    "ok": ok,
                    "http_status": status,
                    "elapsed_sec": round(elapsed, 3),
                    "content": content[:500],
                    "usage": {
                        "prompt_tokens": usage.get("prompt_tokens"),
                        "completion_tokens": usage.get("completion_tokens"),
                        "prefill_speed_tps": usage.get("prefill_speed_tps"),
                        "decoding_speed_tps": usage.get("decoding_speed_tps"),
                    },
                    "npu_engaged": "NPU Locked" in serve_log.read_text(
                        encoding="utf-8", errors="replace"
                    )
                    or bool(usage.get("decoding_speed_tps")),
                }
            )
            return result
        except Exception as exc:  # noqa: BLE001
            result["reason"] = f"flm NPU request failed: {exc}"
            result["serve_tail"] = serve_log.read_text(encoding="utf-8", errors="replace")[-2000:]
            return result
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)


def write_report(report: dict) -> Path:
    REPORTS.mkdir(parents=True, exist_ok=True)
    latest = REPORTS / "latest.json"
    stamp = time.strftime("%Y%m%d-%H%M%S")
    archived = REPORTS / f"smoke-{stamp}.json"
    payload = json.dumps(report, indent=2)
    latest.write_text(payload + "\n", encoding="utf-8")
    archived.write_text(payload + "\n", encoding="utf-8")
    return latest


def summarize(report: dict) -> int:
    hw = report["hardware"]["status"]
    llm = report["vulkan_llm"]
    npu = report["npu"]

    npu_ok = bool(npu.get("ok")) and not npu.get("skipped")
    if llm.get("skipped"):
        # Intentional --skip-llm (ok=True): NPU-only or hardware probe.
        # Safety refusal (ok=False): hard fail unless NPU path still passed.
        hard_ok = bool(hw.get("npu") and npu_ok)
        soft_ok = bool(hw.get("hybrid_ready_cpu_gpu") and llm.get("ok") and hw.get("npu"))
    else:
        hard_ok = (
            hw.get("hybrid_ready_cpu_gpu")
            and llm.get("ok")
            and llm.get("gpu_offload_detected")
        )
        # If GPU offload wasn't detected but inference succeeded, treat as soft warning.
        soft_ok = hw.get("hybrid_ready_cpu_gpu") and llm.get("ok")
        # NPU success alongside GPU is bonus, not required for hard pass.

    log("\n=== SMOKE SUMMARY ===")
    log(f"CPU:          {'PASS' if hw.get('cpu') else 'FAIL'} — {report['hardware']['cpu'].get('model')}")
    log(f"GPU Vulkan:   {'PASS' if hw.get('gpu_vulkan') else 'FAIL'}")
    log(f"NPU device:   {'PASS' if hw.get('npu') else 'FAIL'}")
    if llm.get("skipped"):
        log(f"Vulkan LLM:   SKIP — {llm.get('reason')}")
    else:
        log(
            f"Vulkan LLM:   {'PASS' if llm.get('ok') else 'FAIL'} "
            f"(gpu_offload={llm.get('gpu_offload_detected')}, "
            f"ngl={llm.get('ngl')}, "
            f"tps={llm.get('speed', {}).get('tokens_per_second')})"
        )
    if npu.get("skipped"):
        log(f"NPU infer:    SKIP — {npu.get('reason')}")
    else:
        usage = npu.get("usage") or {}
        log(
            f"NPU infer:    {'PASS' if npu.get('ok') else 'FAIL'} "
            f"(model={npu.get('model')}, "
            f"decode_tps={usage.get('decoding_speed_tps')}, "
            f"npu_engaged={npu.get('npu_engaged')})"
        )

    report["summary"] = {
        "hard_pass": bool(hard_ok),
        "soft_pass": bool(soft_ok),
        "npu_device_ok": bool(hw.get("npu")),
        "npu_infer_ok": bool(npu_ok),
    }

    if hard_ok and not llm.get("skipped"):
        log("RESULT: PASS (CPU+GPU hybrid offload verified)")
        return 0
    if hard_ok and llm.get("skipped") and npu_ok:
        log("RESULT: PASS (NPU inference verified; Vulkan LLM skipped)")
        return 0
    if soft_ok:
        log("RESULT: PASS_WITH_WARNINGS (LLM ok, GPU offload not clearly detected)")
        return 0
    if llm.get("skipped") and llm.get("ok") is False and report.get("vram_preflight", {}).get("ok") is False:
        log("RESULT: FAIL (refused GPU smoke — VRAM / concurrency safety)")
        return 4
    log("RESULT: FAIL")
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Hybrid CPU/GPU/NPU smoke test",
        epilog=(
            "WARNING: On ~512 MiB AMD iGPUs, Cursor and this Vulkan smoke share "
            "the same VRAM pool. Prefer running GPU smoke outside Cursor, or use "
            "--skip-llm for a hardware-only probe."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--llama-tag", default=DEFAULT_LLAMA_TAG)
    parser.add_argument(
        "--ngl",
        type=int,
        default=None,
        help=(
            "GPU layers to offload (default: auto from VRAM size; "
            f"typically {choose_default_ngl({'available': True, 'total_mib': 512})} "
            "on a 512 MiB iGPU, never 99)"
        ),
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_LLM_TIMEOUT_SEC,
        help=f"llama-cli timeout seconds (default {DEFAULT_LLM_TIMEOUT_SEC})",
    )
    parser.add_argument(
        "--min-free-vram-mib",
        type=int,
        default=MIN_FREE_VRAM_MIB,
        help=f"refuse GPU smoke if free VRAM below this (default {MIN_FREE_VRAM_MIB})",
    )
    parser.add_argument(
        "--force-gpu",
        action="store_true",
        help="override VRAM preflight refusal (dangerous on tiny iGPUs)",
    )
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--skip-llm", action="store_true")
    parser.add_argument("--skip-npu-infer", action="store_true")
    parser.add_argument(
        "--flm-model",
        default=DEFAULT_FLM_MODEL,
        help=f"FastFlowLM NPU model tag (default: {DEFAULT_FLM_MODEL})",
    )
    args = parser.parse_args()

    if platform.machine() not in {"x86_64", "AMD64"}:
        log(f"unsupported arch: {platform.machine()}")
        return 2

    with SmokeLock():
        report: dict = {
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "root": str(ROOT),
            "llama_tag": args.llama_tag,
            "model": {"repo": MODEL_REPO, "file": MODEL_FILE},
        }

        log("== 1/4 hardware probe ==")
        hw = probe_hw()
        report["hardware"] = hw
        log(json.dumps(hw["status"], indent=2))

        vram = read_vram()
        report["vram"] = vram
        if vram.get("available"):
            log(
                f"iGPU VRAM: {vram['used_mib']:.0f}/{vram['total_mib']:.0f} MiB used "
                f"({vram['free_mib']:.0f} MiB free)"
            )
        else:
            log(f"iGPU VRAM: unavailable ({vram.get('error')})")

        ngl = args.ngl if args.ngl is not None else choose_default_ngl(vram)
        report["ngl_requested"] = args.ngl
        report["ngl_effective"] = ngl
        if args.ngl is None:
            log(f"auto -ngl={ngl} (based on VRAM; override with --ngl)")

        if args.skip_llm:
            report["vulkan_llm"] = {"skipped": True, "ok": True, "reason": "cli flag"}
            report["vram_preflight"] = {"ok": True, "skipped": True}
        else:
            log("== VRAM / Cursor safety preflight ==")
            pre = vram_preflight(
                vram, min_free_mib=args.min_free_vram_mib, force=args.force_gpu
            )
            report["vram_preflight"] = pre
            if pre.get("warning"):
                log(f"NOTE: {pre['warning']}")
            if not pre["ok"]:
                log(f"REFUSING GPU smoke: {pre.get('error')}")
                report["vulkan_llm"] = {
                    "skipped": True,
                    "ok": False,
                    "reason": pre.get("error"),
                }
            else:
                log("== 2/4 fetch llama.cpp Vulkan ==")
                llama_bin = ensure_llama(args.llama_tag, force=args.force_download)
                report["llama_bin"] = str(llama_bin)

                log("== 3/4 fetch model ==")
                model = ensure_model(force=args.force_download)
                report["model"]["path"] = str(model)
                report["model"]["sha256"] = sha256_file(model)

                log(f"== 4/4 Vulkan LLM completion (ngl={ngl}, timeout={args.timeout}s) ==")
                report["vulkan_llm"] = run_vulkan_llm(
                    llama_bin, model, ngl, timeout=args.timeout
                )

        if args.skip_npu_infer:
            report["npu"] = {"skipped": True, "reason": "cli flag", "ok": True}
        else:
            log("== optional NPU inference ==")
            report["npu"] = run_npu_optional(model=args.flm_model)

        code = summarize(report)
        path = write_report(report)
        log(f"report: {path}")
        return code


if __name__ == "__main__":
    try:
        sys.exit(main())
    except urllib.error.URLError as exc:
        log(f"network error: {exc}")
        sys.exit(3)
    except Exception as exc:  # noqa: BLE001 - top-level smoke harness
        log(f"error: {exc}")
        sys.exit(1)
