# hybrid

Smoke tests for hybrid local inference on AMD Ryzen AI (CPU + Radeon iGPU + XDNA NPU).

## What it checks

1. **Hardware probe** — CPU, Vulkan iGPU, NPU (`/dev/accel/accel0`)
2. **CPU + GPU LLM** — short completion via llama.cpp Vulkan with GPU layer offload
3. **NPU inference** — FastFlowLM (`flm`) OpenAI-compatible serve + chat completion

## Models

| Path | Model |
|---|---|
| Vulkan CPU+GPU | **Qwen2.5-0.5B-Instruct Q4_K_M** GGUF |
| NPU (FLM) | **qwen3:0.6b** (`Qwen3-0.6B-NPU2`) |

## Run

Hardware probe only (safe alongside Cursor):

```bash
python3 scripts/check_hw.py
```

NPU-only smoke (safe alongside Cursor; uses XDNA, not iGPU VRAM):

```bash
./scripts/smoke_test.py --skip-llm
```

Full Vulkan GPU smoke (prefer **outside** Cursor):

```bash
./scripts/smoke_test.py
```

Artifacts land under `.cache/` (binaries) and `models/` (GGUF). A JSON report is written to `reports/latest.json`.

### FastFlowLM (NPU) install — user-local, no sudo

```bash
mkdir -p ~/.local/share/fastflowlm ~/.local/bin
curl -L -o /tmp/fastflowlm_linux.tar.gz \
  https://github.com/ROCm/FastFlowLM/releases/download/v1.0.7/fastflowlm_1.0.7_linux.tar.gz
tar -xzf /tmp/fastflowlm_linux.tar.gz -C ~/.local/share/fastflowlm
printf '%s\n' '#!/usr/bin/env bash' 'exec '"$HOME"'/.local/share/fastflowlm/flm "$@"' > ~/.local/bin/flm
chmod +x ~/.local/bin/flm
export PATH="$HOME/.local/bin:$PATH"
flm validate
flm pull qwen3:0.6b
```

Do **not** `ln -s` the portable `flm` into `~/.local/bin` and then overwrite that path — that can clobber the upstream wrapper.

### Safety notes (512 MiB iGPU)

Cursor and the Vulkan smoke share the **same tiny iGPU VRAM pool**. The harness uses a lock, truncated subprocess I/O, auto `-ngl`, and refuses GPU runs when free VRAM < 128 MiB.

NPU smoke does **not** need free iGPU VRAM.

## Requirements

- Linux x86_64 with Vulkan (Mesa AMD)
- Network on first run (downloads llama.cpp / GGUF / FLM models)
- Optional/recommended: portable FastFlowLM in `~/.local` for NPU inference
