#!/usr/bin/env python3
"""Submit one or more first-frame-to-video smoke tests to MiniMax H3."""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


PROMPTS = {
    "orbit-left": (
        "For the target video, at 0.00 seconds, <Picture 1> is fully referenced. "
        "The scene and every object are completely static and rigid. The camera performs one "
        "very slow, smooth cinematic orbit to the left around the scene, creating physically "
        "consistent parallax and revealing a slightly new side view. Preserve object identity, "
        "geometry, material, lighting, colors, relative positions, and background. No object, "
        "person, fabric, shadow, or light moves or deforms. No cuts, zoom, new objects, or scene "
        "changes. Stable tripod-height camera, constant focal length. Quiet room tone only; no music."
    ),
    "orbit-right": (
        "For the target video, at 0.00 seconds, <Picture 1> is fully referenced. "
        "The scene and every object are completely static and rigid. The camera performs one "
        "very slow, smooth cinematic orbit to the right around the scene, with coherent parallax. "
        "Preserve exact identity, 3D geometry, materials, lighting, colors, spatial relationships, "
        "and background. Nothing in the scene moves or deforms. No cuts, zoom, additions, or "
        "relighting. Stable camera height and constant focal length. Quiet room tone only; no music."
    ),
    "dolly-in": (
        "For the target video, at 0.00 seconds, <Picture 1> is fully referenced. "
        "Everything in the scene remains perfectly static. The camera makes a very slow, smooth "
        "straight dolly forward, producing natural perspective parallax; this is physical camera "
        "translation, not digital zoom. Preserve all geometry, appearance, lighting, positions, and "
        "the background. No subject motion, deformation, cuts, new content, or focus change. "
        "Quiet room tone only; no music."
    ),
}


def request_json(url: str, method: str = "GET", payload: dict | None = None) -> dict:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=body, method=method)
    if body is not None:
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=120) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {url}: {detail}") from exc


def download(url: str, destination: Path) -> None:
    with urllib.request.urlopen(url, timeout=600) as response:
        destination.write_bytes(response.read())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("images", nargs="+", type=Path)
    parser.add_argument("--motion", choices=sorted(PROMPTS), default="orbit-left")
    parser.add_argument("--prompt", help="Override the camera-motion prompt")
    parser.add_argument("--server", default="http://127.0.0.1:30010")
    parser.add_argument("--output-dir", type=Path, default=Path("minimax_h3/outputs"))
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=2101)
    parser.add_argument("--timeout", type=float, default=4 * 60 * 60)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not 4 <= args.duration <= 15:
        parser.error("--duration must be between 4 and 15 seconds")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    prompt = args.prompt or PROMPTS[args.motion]

    for index, image_arg in enumerate(args.images):
        image = image_arg.expanduser().resolve()
        if not image.is_file():
            raise FileNotFoundError(image)
        payload = {
            "model": "MiniMaxAI/MiniMax-H3",
            "prompt": prompt,
            "seconds": args.duration,
            "task": "fl2va",
            "conditions": [{
                "type": "image",
                "uri": image.as_uri(),
                "role": "keyframe",
                "frame_index": 0,
            }],
            "target": {
                "short_edge": 768,
                "aspect_ratio": "auto",
                "duration_seconds": args.duration,
            },
            "num_outputs_per_prompt": 1,
            "num_inference_steps": args.steps,
            "flow_shift": 12.0,
            "audio_flow_shift": 3.0,
            "seed": args.seed + index,
        }
        if args.dry_run:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
            continue

        created = request_json(f"{args.server}/v1/videos", "POST", payload)
        video_id = created.get("id")
        if not video_id:
            raise RuntimeError(f"Server returned no job id: {created}")
        print(f"[{image.name}] submitted as {video_id}", flush=True)

        deadline = time.monotonic() + args.timeout
        while True:
            status_data = request_json(f"{args.server}/v1/videos/{video_id}")
            status = status_data.get("status")
            print(f"[{image.name}] {status}", flush=True)
            if status == "completed":
                break
            if status in {"failed", "cancelled"}:
                raise RuntimeError(json.dumps(status_data, ensure_ascii=False, indent=2))
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Timed out waiting for {video_id}")
            time.sleep(5)

        output = args.output_dir / f"{image.stem}_{args.motion}_seed{args.seed + index}.mp4"
        download(f"{args.server}/v1/videos/{video_id}/content", output)
        print(f"Saved {output} ({output.stat().st_size / 1024**2:.1f} MiB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
