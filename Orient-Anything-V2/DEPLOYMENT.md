# Local deployment notes

The official source is pinned locally at commit
`73b11c9dc83e84daeb563d0c766831f2c66b0a18`.

## Environment

```bash
conda activate orianyv2
cd /hpc_stor03/sjtu_home/zhiying.zou/Parcours_recherche_server_sync/Orient-Anything-V2
```

The environment uses Python 3.11 and the dependencies from `requirements.txt`.
PyTorch's CUDA build must be checked against the driver on the eventual GPU node.

## Checkpoint

Download the official checkpoint:

<https://huggingface.co/Viglong/OriAnyV2_ckpt/resolve/main/demo_ckpts/rotmod_realrotaug_best.pt>

Place it at:

```text
checkpoints/demo_ckpts/rotmod_realrotaug_best.pt
```

Its expected size is exactly `5,048,116,892` bytes.
The Hugging Face file metadata reports SHA-256
`7b6b7f258d32b95123b9d023005ecca357d8ab944fb83476f532d3cf7a2295eb`.

The server-side download has been verified and installed at the path above.

Example upload from a local machine:

```bash
scp rotmod_realrotaug_best.pt USER@CLUSTER:/hpc_stor03/sjtu_home/zhiying.zou/Parcours_recherche_server_sync/Orient-Anything-V2/checkpoints/demo_ckpts/
```

On the cluster, verify it with:

```bash
sha256sum checkpoints/demo_ckpts/rotmod_realrotaug_best.pt
```

## Single-image inference

The input should contain one main object. For a COCO image, crop one person first.

```bash
python infer_cli.py assets/examples/bottle.jpg --output outputs/bottle.json
```

Background removal is optional and triggers an additional rembg model download on
its first run:

```bash
python infer_cli.py person_crop.jpg --remove-background --output outputs/person.json
```

The command checks CUDA and checkpoint completeness before constructing the model.
It reports azimuth, elevation, in-plane rotation, and the predicted number of front
directions as JSON.

The included GPU YAML is a Kubernetes template pending the cluster-specific
image, storage claim, and submission schema.

## COCO smoke-test batch

Eight person crops and eight directional-object crops are prepared under
`test_data/`. Batch inference loads the model once and then calls the official
`utils.app_utils.inf_single_case` function for every image:

```bash
./run_coco_batch.sh
```

Results are written to `outputs/coco_smoke/results.json` and `results.csv`.
Background removal is disabled for the first baseline so the natural-image
context remains intact. It can be enabled by invoking `infer_batch.py` with
`--remove-background` in a separate run.

Render the official Blender axes over the saved model inputs without rerunning
GPU inference:

```bash
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
python visualize_results.py
```

The individual overlays and a contact sheet are written under
`outputs/coco_smoke/visualizations/`.

`gpu_job.kubernetes.yaml` is a standard Kubernetes Job template. The repository
does not currently contain a cluster-specific job example, so its image and PVC
placeholders must be replaced before submission. If the cluster uses a custom
YAML schema, retain the command `bash run_coco_batch.sh` and translate only the
scheduler wrapper.

## SJTU bare-metal A800 workflow

The supplied SJTU note says this server does not use a scheduler. Log in to the
A800 host, build the image once, and start a Docker container directly:

```bash
cd /aistor/hpc_stor03/sjtu_home/zhiying.zou/Parcours_recherche_server_sync
docker build -t docker.v2.aispeech.com/sjtu/sjtu_chenlu-zzy-cuda_12.4-ubuntu_22.04-torch_2.6-orient-anything:v0.1 .
./Orient-Anything-V2/run_docker_gpu.sh
```

Select another GPU or image through environment variables:

```bash
ORIANY_GPU=1 ORIANY_IMAGE=docker.v2.aispeech.com/sjtu/sjtu_chenlu-zzy-cuda_12.4-ubuntu_22.04-torch_2.6-orient-anything:v0.1 \
  ./Orient-Anything-V2/run_docker_gpu.sh
```

The script mounts the project at `/workspace/project`, uses the caller's UID/GID
for output ownership, and runs `run_coco_batch.sh`. The Kubernetes YAML is not
part of this bare-metal workflow.

## Gradio UI

After entering a GPU job and activating the environment, use the wrapper so that
Blender can find the Conda-provided shared libraries:

```bash
./run_gradio.sh
```

The UI is optional. `infer_cli.py` does not import Blender and is the preferred
entry point for batch evaluation.
