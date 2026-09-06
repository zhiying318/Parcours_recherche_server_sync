# MiniMax H3 local image-to-video smoke test

This deployment serves the official `FL2VA` checkpoint through SGLang and uses a local image as the exact first frame. Output is 768-pixel short-edge H.264 video at 24 FPS with 32 kHz stereo AAC audio.

## Install

```bash
./minimax_h3/setup.sh
./minimax_h3/download_weights.sh
```

The download command stores FL2VA in `checkpoints/MiniMax-H3`. Hugging Face reports the partition as about 144 GB; allow extra room for cache metadata and temporary download files. It is resumable. If access is gated, run `hf auth login` or export `HF_TOKEN` first. If the local directory is absent, the server falls back to downloading from the Hub cache automatically.

After a successful download, the script writes `checkpoints/MiniMax-H3/.fl2va-download-complete`. The server refuses a partially populated local directory, preventing accidental startup with incomplete weights.

The script enables the Xet high-performance transfer mode and 16 workers by default. If your network is rate-limited, lower parallelism with `MINIMAX_H3_DOWNLOAD_WORKERS=4`.

## Start on four A100-80GB GPUs

Edit the default in `serve_a100.sh`, or select physical devices at launch:

```bash
MINIMAX_H3_GPUS=0,1,2,3 ./minimax_h3/serve_a100.sh 2>&1 | tee minimax_h3/server.log
```

The service binds only to `127.0.0.1:30010`. SGLang sees the selected physical devices renumbered as logical devices 0–3. The launch uses TP2 + Ulysses2, the official fastest measured four-card 80 GB topology. On A100, budget the same 66–70 GB per GPU as the official H100 measurement; A100 inference will be slower.

The launch script also redirects Hugging Face, TorchInductor, Triton, and XDG caches into the project because the cluster's home-level `.cache` may be read-only.

## Start a single-A100 smoke-test server

This slower CPU-offload route is intended only for one-image smoke tests:

```bash
MINIMAX_H3_GPU=5 ./minimax_h3/serve_one_a100.sh 2>&1 | tee minimax_h3/server-one-a100.log
```

It is deliberately separate from `serve_a100.sh`: the latter is a four-card resident server, while this script uses one card with DiT, text-encoder, and VAE layerwise CPU offload.

## Run smoke tests

In a second shell:

```bash
./minimax_h3/run_smokes.sh
```

Outputs go to `minimax_h3/outputs/`. To test a different image or motion:

```bash
.venv-minimax-h3/bin/python minimax_h3/smoke_i2v.py path/to/image.png \
  --motion orbit-right --duration 5 --steps 50
```

Supported presets are `orbit-left`, `orbit-right`, and `dolly-in`. Inspect a request without submitting it by adding `--dry-run`.

## Hardware notes

- Recommended resident deployment: **4 × A100-80GB**, approximately **66–70 GB VRAM per GPU** (264–280 GB aggregate allocated VRAM). This is an estimate transferred from SGLang's measured H100-80GB TP2 + Ulysses2 peak of 66.04 GB/GPU; validate the exact A100 peak locally.
- Lower-memory resident topology: TP4 + Ulysses1 measured 49.80 GB/GPU on four H100s. It should fit four A100-80GB cards but is slightly slower; change the two topology flags if needed.
- Capacity topology: FSDP + Ulysses4 measured 57.01 GB/GPU on four H100s. It trades communication for memory and can be selected with `--use-fsdp-inference true`.
- Offload can technically run on fewer GPUs, even one card, but requires roughly 108 GB of model weights plus substantial host RAM/disk traffic and is far slower. This host has enough RAM, but the four-card resident route is the meaningful A100 smoke-test configuration.
- Checkpoint storage is about **144 GB** for FL2VA. SGLang documents about **108 GB** of active weights (61.73 GB DiT + 46.18 GB text encoder).

The fully local open release is H3-Base only: it generates 768p. MiniMax's Context-IR prompt orchestration and H3-Regenerate-2K are hosted-only at present.
