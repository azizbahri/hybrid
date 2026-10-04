# hybrid

Smoke tests for hybrid local inference on AMD Ryzen AI (CPU + Radeon iGPU + XDNA NPU).

## What it checks

1. **Hardware probe** — CPU, Vulkan iGPU, NPU (`/dev/accel/accel0`)
2. **CPU + GPU LLM** — short completion via llama.cpp Vulkan with GPU layer offload
3. **NPU readiness** — device/firmware present; optional FLM inference if installed

## Model

Primary smoke model: **Qwen2.5-0.5B-Instruct Q4_K_M** GGUF  
Small, fast download, enough to prove CPU+GPU layer offload.

## Run

Hardware probe only (safe alongside Cursor):

```bash
python3 scripts/check_hw.py
# or
./scripts/smoke_test.py --skip-llm --skip-npu-infer
```

Full Vulkan GPU smoke (prefer **outside** Cursor):

```bash
./scripts/smoke_test.py
```

Artifacts land under `.cache/` (binaries) and `models/` (GGUF). A JSON report is written to `reports/latest.json`.

### Safety notes (512 MiB iGPU)

Cursor and this Vulkan smoke share the **same tiny iGPU VRAM pool**. Running `llama-cli -ngl 99` while Cursor is GPU-active has filled host RAM (unbounded subprocess capture) and nearly exhausted 512 MiB VRAM, crashing the IDE.

The smoke harness now:

- Uses a **single-instance lock** (`.cache/smoke_test.lock`) so concurrent smokes cannot stack
- **Streams/truncates** subprocess output (no unbounded `capture_output` into RAM)
- **Auto-selects a conservative `-ngl`** from sysfs VRAM size (≈8 on 512 MiB; override with `--ngl`)
- **Refuses GPU runs** when free VRAM is below 128 MiB (`--force-gpu` to override; not recommended)
- Uses a **shorter llama-cli timeout** (default 120s)

Prefer closing GPU-heavy apps (or leaving Cursor) before a full smoke. Use `--skip-llm` when you only need the hardware probe.

## Requirements

- Linux x86_64 with Vulkan (Mesa AMD)
- Network on first run (downloads llama.cpp Vulkan build + model)
- Optional: FastFlowLM (`flm`) for NPU inference step
