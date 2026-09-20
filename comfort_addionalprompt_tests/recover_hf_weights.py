"""Recover missing HF weight shards with bounded, resumable HTTP Range requests.

Run in the evaluation container: python comfort_addionalprompt_tests/recover_hf_weights.py
Uses the configured HTTP(S) proxy and HF cache. Existing complete blobs are retained.
"""
import argparse
import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from filelock import FileLock
from huggingface_hub import get_hf_file_metadata, hf_hub_download, hf_hub_url
from huggingface_hub.errors import RemoteEntryNotFoundError
from huggingface_hub.utils import build_hf_headers

MODEL = 'OpenGVLab/InternVL3_5-8B-HF'
CHUNK = 32 * 1024 * 1024


def recover(model_id=MODEL):
    try:
        index = Path(hf_hub_download(model_id, 'model.safetensors.index.json'))
        names = sorted(set(json.loads(index.read_text())['weight_map'].values()))
    except RemoteEntryNotFoundError:
        # Smaller models can publish one safetensors file without a shard index.
        index = Path(hf_hub_download(model_id, 'config.json'))
        names = ['model.safetensors']
    headers = build_hf_headers()
    revision = index.parent.name
    cache = index.parent.parent.parent
    for name in names:
        snapshot_file = index.parent / name
        if snapshot_file.is_file():
            print(f'Cached: {name}', flush=True)
            continue
        url = hf_hub_url(model_id, name, revision=revision)
        metadata = get_hf_file_metadata(url, timeout=30)
        size, digest = metadata.size, metadata.etag
        if size is None or digest is None or len(digest) != 64:
            raise ValueError(f'Missing size/SHA256 for {name}')
        blob = cache / 'blobs' / digest
        parts = cache / 'range-recovery' / digest
        parts.mkdir(parents=True, exist_ok=True)
        with FileLock(str(parts / 'recovery.lock')):
            def fetch(start):
                end = min(start + CHUNK, size) - 1
                part = parts / str(start)
                expected = end - start + 1
                if part.exists() and part.stat().st_size == expected:
                    return expected
                for attempt in range(12):
                    try:
                        t = time.monotonic()
                        # Request a fresh redirect for each range; never log signed URLs.
                        with requests.get(url, headers={**headers, 'Range': f'bytes={start}-{end}'},
                                          stream=True, timeout=(15, 20)) as response:
                            response.raise_for_status()
                            if response.status_code != 206 or response.headers.get('Content-Range') != f'bytes {start}-{end}/{size}':
                                raise ValueError('Server did not honor the requested byte range')
                            with part.open('wb') as stream:
                                count = 0
                                for data in response.iter_content(256 * 1024):
                                    count += len(data)
                                    if count > expected:
                                        raise ValueError('Oversized range response')
                                    stream.write(data)
                                    if time.monotonic() - t > 120:
                                        raise TimeoutError('Range exceeded 120 seconds')
                            if count != expected:
                                raise ValueError('Truncated range response')
                        return expected
                    except (requests.RequestException, ValueError, TimeoutError) as exc:
                        print(f'Retry {name} offset={start} attempt={attempt + 1}: {type(exc).__name__}', flush=True)
                        if attempt == 11:
                            raise
                        time.sleep(min(attempt + 1, 5))

            offsets = list(range(0, size, CHUNK))
            done = 0
            with ThreadPoolExecutor(max_workers=4) as pool:
                for future in as_completed([pool.submit(fetch, start) for start in offsets]):
                    done += future.result()
                    print(f'{name}: {done}/{size} bytes ({100 * done / size:.1f}%)', flush=True)
            assembled = parts / 'assembled'
            checksum = hashlib.sha256()
            with assembled.open('wb') as output:
                for start in offsets:
                    with (parts / str(start)).open('rb') as source:
                        while data := source.read(8 * 1024 * 1024):
                            checksum.update(data)
                            output.write(data)
            if checksum.hexdigest() != digest:
                raise ValueError(f'SHA256 mismatch for {name}; recovery parts retained for inspection')
            os.replace(assembled, blob)
            if not snapshot_file.is_symlink():
                snapshot_file.symlink_to(os.path.relpath(blob, snapshot_file.parent))
            for start in offsets:
                (parts / str(start)).unlink()
            print(f'Verified SHA256 and cached: {name}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-id', default=MODEL)
    recover(parser.parse_args().model_id)
