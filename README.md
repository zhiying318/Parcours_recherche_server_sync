### Commonly used commands in this project
```
ps aux | grep python # find the pid of a process

source /home/zzou/.dataset/bin/activate # environment to generate WhatsUp dataset
deactivate

cd COMFORT
conda activate comfort # environment for using COMFORT framework
python data_generation/generate_dataset.py  --dataset_name comfort_human_car  --save_path ./data
```
