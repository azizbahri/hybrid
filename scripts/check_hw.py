#!/usr/bin/env python3
"""Probe CPU / GPU (Vulkan) / NPU (XDNA) for the hybrid smoke test."""

from __future__ import annotations

import json
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path


def _run(cmd: list[str], timeout: int = 15) -> tuple[int, str, str]:
    try:
        p = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        return p.returncode, p.stdout, p.stderr
    except FileNotFoundError:
        return 127, "", f"not found: {cmd[0]}"
    except subprocess.TimeoutExpired:
        return 124, "", "timeout"


def cpu_info() -> dict:
    info = {
        "arch": platform.machine(),
        "cpus_logical": os.cpu_count(),
        "model": None,
        "vendor": None,
        "flags_ai": [],
    }
    try:
        text = Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        info["error"] = str(exc)
        return info

    for line in text.splitlines():
        if line.startswith("model name") and not info["model"]:
            info["model"] = line.split(":", 1)[1].strip()
        elif line.startswith("vendor_id") and not info["vendor"]:
            info["vendor"] = line.split(":", 1)[1].strip()
        elif line.startswith("flags") or line.startswith("Features"):
            flags = line.split(":", 1)[1].strip().split()
            interesting = [
                "avx2",
                "avx512f",
                "avx512_bf16",
                "avx512_vnni",
                "avx_vnni",
                "fma",
            ]
            info["flags_ai"] = [f for f in interesting if f in flags]
            break
    return info


def gpu_info() -> dict:
    info = {
        "vulkan_available": False,
        "devices": [],
        "drm_render": [],
        "amdgpu_loaded": False,
    }

    render = Path("/dev/dri")
    if render.is_dir():
        info["drm_render"] = sorted(p.name for p in render.glob("renderD*"))

    code, out, _ = _run(["lsmod"])
    if code == 0:
        info["amdgpu_loaded"] = "amdgpu" in out

    code, out, err = _run(["vulkaninfo", "--summary"], timeout=30)
    if code != 0:
        info["vulkan_error"] = (err or out or "vulkaninfo failed").strip()[:500]
        return info

    info["vulkan_available"] = True
    devices = []
    current: dict | None = None
    for line in out.splitlines():
        line = line.strip()
        if line.startswith("GPU"):
            if current:
                devices.append(current)
            current = {"title": line}
        elif current is not None and "=" in line:
            key, val = [x.strip() for x in line.split("=", 1)]
            if key in {"deviceName", "driverName", "driverInfo", "deviceType", "apiVersion"}:
                current[key] = val
    if current:
        devices.append(current)
    info["devices"] = devices
    return info


def npu_info() -> dict:
    info = {
        "accel_nodes": [],
        "amdxdna_loaded": False,
        "firmware_hint": None,
        "pci": None,
        "flm_present": False,
    }

    accel = Path("/dev/accel")
    if accel.is_dir():
        info["accel_nodes"] = sorted(p.name for p in accel.iterdir() if p.name.startswith("accel"))

    code, out, _ = _run(["lsmod"])
    if code == 0:
        info["amdxdna_loaded"] = "amdxdna" in out

    code, out, _ = _run(["lspci", "-nn"])
    if code == 0:
        for line in out.splitlines():
            if re.search(r"Neural Processing Unit|1022:17f0", line, re.I):
                info["pci"] = line.strip()
                break

    # Best-effort firmware path used by Strix XDNA2
    for candidate in (
        Path("/usr/lib/firmware/amdnpu/17f0_10"),
        Path("/lib/firmware/amdnpu/17f0_10"),
    ):
        if candidate.is_dir():
            info["firmware_hint"] = str(candidate)
            break

    flm = shutil.which("flm")
    if not flm:
        for candidate in (
            Path.home() / ".local/bin/flm",
            Path.home() / ".local/share/fastflowlm/flm",
        ):
            if candidate.is_file() and os.access(candidate, os.X_OK):
                flm = str(candidate)
                break
    info["flm_path"] = flm
    if flm:
        code, out, err = _run([flm, "validate"], timeout=15)
        text = (out or "") + (err or "")
        info["flm_present"] = code == 0
        info["flm_validate_ok"] = code == 0 and "/dev/accel" in text
    else:
        info["flm_present"] = False
        info["flm_validate_ok"] = False

    return info


def evaluate(report: dict) -> dict:
    cpu_ok = bool(report["cpu"].get("model"))
    gpu_ok = bool(report["gpu"].get("vulkan_available")) and bool(report["gpu"].get("devices"))
    npu_ok = bool(report["npu"].get("accel_nodes")) and report["npu"].get("amdxdna_loaded")
    return {
        "cpu": cpu_ok,
        "gpu_vulkan": gpu_ok,
        "npu": npu_ok,
        "hybrid_ready_cpu_gpu": cpu_ok and gpu_ok,
        "hybrid_ready_with_npu_device": cpu_ok and gpu_ok and npu_ok,
    }


def main() -> int:
    report = {
        "hostname": platform.node(),
        "kernel": platform.release(),
        "os": platform.platform(),
        "cpu": cpu_info(),
        "gpu": gpu_info(),
        "npu": npu_info(),
    }
    report["status"] = evaluate(report)
    print(json.dumps(report, indent=2))
    # Hardware probe fails only if CPU missing; GPU/NPU are soft for this helper.
    return 0 if report["status"]["cpu"] else 1


if __name__ == "__main__":
    sys.exit(main())
