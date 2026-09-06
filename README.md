### Commonly used commands in this project
```
ps aux | grep python # find the pid of a process

source /home/zzou/.dataset/bin/activate # environment to generate WhatsUp dataset
deactivate

cd COMFORT
conda activate comfort # environment for using COMFORT framework
python data_generation/generate_dataset.py  --dataset_name comfort_human_car  --save_path ./data
```

### COMFORT prompt-ablation evaluation

Use the Docker container environment for non-API model evaluation. For
example, to run `test07_camera_geometry_before_question` independently:

```bash
QWEN_GPU=2 bash comfort_addionalprompt_tests/run_docker_gpu_COMFORT.sh \
  qwen35 test07_camera_geometry_before_question

GEMMA_GPU=3 bash comfort_addionalprompt_tests/run_docker_gpu_COMFORT.sh \
  gemma test07_camera_geometry_before_question

INTERNVL_GPU=0 bash comfort_addionalprompt_tests/run_docker_gpu_COMFORT.sh \
  internvl test07_camera_geometry_before_question
```

The Docker script sets `HF_ENDPOINT=https://huggingface.co` inside the
container by default. See
[`comfort_addionalprompt_tests/README.md`](comfort_addionalprompt_tests/README.md)
for GPU isolation and other options.
