#!/bin/sh
#PBS -l select=1:ncpus=20:ngpus=1
#PBS -l walltime=23:59:00
#PBS -j oe
#PBS -N postcavaComparisons0Shot
#PBS -q a100f

cd /fastwork/psalvador/JHU/AnnotationVLM

/fastwork/psalvador/JHU/AnnotationVLM/anaconda3/bin/conda init
source ~/.bashrc
conda activate vllm

TRANSFORMERS_CACHE=./HFCache HF_HOME=./HFCache CUDA_VISIBLE_DEVICES=0 vllm serve "Qwen/Qwen2-VL-72B-Instruct-AWQ" --dtype=half --tensor-parallel-size 1 --limit-mm-per-prompt image=4 --gpu_memory_utilization 0.95 --port 8004 > IVCapi.log 2>&1 &

# Check if the API is up
while ! curl -s http://localhost:8004/v1/models; do
    echo "Waiting for API to be ready..."
    sleep 5
done
# API is ready, log this event
echo "API is ready. Running RunAPI.py" >> ComparisonsAtlasBench0ShotDSC7.log

# Run the Python script with real-time logging
python3 RunAPI.py --path /fastwork/psalvador/JHU/data/AbdomenAtlas1.0Projections/projections_AtlasBench_beta_pro/ \
--csv_path ComparisonsAtlasBench0ShotMaxDSC7.csv --continuing --dice_list /fastwork/psalvador/JHU/data/AbdomenAtlas1.0Projections/ \
--organ_list [postcava] --dice_th_max 0.7 --port 8004 >> ComparisonsAtlasBench0ShotDSC7.log 2>&1

# Log completion
echo "RunAPI.py has finished executing." >> ComparisonsAtlasBench0ShotDSC7.log
