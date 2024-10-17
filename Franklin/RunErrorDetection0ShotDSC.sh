#!/bin/sh
#PBS -l select=1:ncpus=10:ngpus=1
#PBS -l walltime=23:59:00
#PBS -j oe
#PBS -N DSC_ED0Shot
#PBS -q gpu_a100

cd /fastwork/psalvador/JHU/AnnotationVLM

/fastwork/psalvador/JHU/AnnotationVLM/anaconda3/bin/conda init
source ~/.bashrc
conda activate vllm

TRANSFORMERS_CACHE=./HFCache HF_HOME=./HFCache CUDA_VISIBLE_DEVICES=0 vllm serve "Qwen/Qwen2-VL-72B-Instruct-AWQ" --dtype=half --tensor-parallel-size 1 --limit-mm-per-prompt image=3 --gpu_memory_utilization 0.95 --port 8000 > DSC_EDapi.log 2>&1 &

# Check if the API is up
while ! curl -s http://localhost:8000/v1/models; do
    echo "Waiting for API to be ready..."
    sleep 5
done

# API is ready, log this event
echo "API is ready. Running RunAPI.py" >> DSC_ErrorDetectionAtlasBench0Shot.log

# Run the Python script with real-time logging
python3 RunErrorDetection.py --path /fastwork/psalvador/JHU/data/AbdomenAtlas1.0Projections/projections_AtlasBench_beta_pro/ --port 8000 \
--csv_path DSC_ErrorDetectionAtlasBench0Shot.csv --continuing --organ_list all --examples 0 \
--dice_list /fastwork/psalvador/JHU/data/AbdomenAtlas1.0Projections/ --dice_check >> DSC_ErrorDetectionAtlasBench0Shot.log 2>&1

# Log completion
echo "RunAPI.py has finished executing." >> DSC_ErrorDetectionAtlasBench0Shot.log
