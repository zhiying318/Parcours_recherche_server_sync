"""Download and offline-verify every Geometry Teacher model snapshot."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

from huggingface_hub import snapshot_download


MODEL_IDS = (
    "IDEA-Research/grounding-dino-base",
    "facebook/sam2.1-hiera-large",
    "usyd-community/vitpose-base",
    "facebook/VGGT-1B",
)


def directory_size(path: Path) -> int:
    return sum(
        item.stat().st_size
        for item in path.rglob("*")
        if item.is_file() and not item.is_symlink()
    )


def human_size(size: int) -> str:
    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024.0 or unit == "TiB":
            return f"{value:.1f} {unit}"
        value /= 1024.0
    raise AssertionError("unreachable")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, default=None)
    args = parser.parse_args()

    if os.environ.get("HF_HUB_DISABLE_XET") != "1":
        raise RuntimeError("HF_HUB_DISABLE_XET=1 is required for this download")
    cache_dir = args.cache_dir or Path(os.environ.get("HF_HOME", "~/.cache/huggingface")).expanduser()
    hub_dir = cache_dir / "hub"
    hub_dir.mkdir(parents=True, exist_ok=True)

    print(f"Hugging Face cache: {cache_dir}", flush=True)
    print("Xet disabled: yes", flush=True)
    failures: list[tuple[str, str]] = []
    snapshots: dict[str, Path] = {}

    for index, model_id in enumerate(MODEL_IDS, start=1):
        before = directory_size(cache_dir)
        started = time.monotonic()
        print(f"\n[{index}/{len(MODEL_IDS)}] Downloading {model_id}", flush=True)
        try:
            snapshot = Path(snapshot_download(repo_id=model_id, cache_dir=hub_dir))
            snapshots[model_id] = snapshot
            elapsed = time.monotonic() - started
            after = directory_size(cache_dir)
            print(
                f"SUCCESS {model_id} in {elapsed:.1f}s; "
                f"cache={human_size(after)}; added={human_size(max(0, after - before))}",
                flush=True,
            )
        except Exception as error:  # Keep going so the report covers every model.
            elapsed = time.monotonic() - started
            failures.append((model_id, repr(error)))
            print(f"FAILED {model_id} after {elapsed:.1f}s: {error!r}", flush=True)

    print("\nOffline snapshot verification", flush=True)
    os.environ["HF_HUB_OFFLINE"] = "1"
    for model_id in MODEL_IDS:
        try:
            snapshot = Path(
                snapshot_download(repo_id=model_id, cache_dir=hub_dir, local_files_only=True)
            )
            if not snapshot.is_dir() or not any(snapshot.iterdir()):
                raise RuntimeError(f"empty snapshot directory: {snapshot}")
            snapshots[model_id] = snapshot
            print(f"OFFLINE OK {model_id}: {snapshot}", flush=True)
        except Exception as error:
            if not any(item[0] == model_id for item in failures):
                failures.append((model_id, repr(error)))
            print(f"OFFLINE FAILED {model_id}: {error!r}", flush=True)

    incomplete = sorted(hub_dir.rglob("*.incomplete"))
    print(f"\nFinal cache size: {human_size(directory_size(cache_dir))}", flush=True)
    if incomplete:
        print("Incomplete cache files remain:", flush=True)
        for path in incomplete:
            print(f"  {path} ({human_size(path.stat().st_size)})", flush=True)
    else:
        print("Incomplete cache files: none", flush=True)

    if failures or incomplete:
        print("\nPREFETCH FAILED", file=sys.stderr, flush=True)
        for model_id, error in failures:
            print(f"  {model_id}: {error}", file=sys.stderr, flush=True)
        return 1
    print("\nPREFETCH AND OFFLINE VERIFICATION SUCCEEDED", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
