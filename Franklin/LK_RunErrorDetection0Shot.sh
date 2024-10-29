#!/bin/sh
#PBS -l select=1:ncpus=20:ngpus=2
#PBS -l walltime=23:59:00
#PBS -j oe
#PBS -N LK_ED0Shot
#PBS -q gpu_a100

cd /fastwork/psalvador/JHU/AnnotationVLM

/fastwork/psalvador/JHU/AnnotationVLM/anaconda3/bin/conda init
source ~/.bashrc
conda activate vllm

TRANSFORMERS_CACHE=./HFCache HF_HOME=./HFCache CUDA_VISIBLE_DEVICES=0 vllm serve "Qwen/Qwen2-VL-72B-Instruct-AWQ" --dtype=half --tensor-parallel-size 1 --limit-mm-per-prompt image=3 --gpu_memory_utilization 0.95 --port 8002 > EDapi.log 2>&1 &
TRANSFORMERS_CACHE=./HFCache HF_HOME=./HFCache CUDA_VISIBLE_DEVICES=1 vllm serve "Qwen/Qwen2-VL-72B-Instruct-AWQ" --dtype=half --tensor-parallel-size 1 --limit-mm-per-prompt image=3 --gpu_memory_utilization 0.95 --port 8003 > EDapi.log 2>&1 &

# Check if the API is up
while ! curl -s http://localhost:8002/v1/models; do
    echo "Waiting for API to be ready..."
    sleep 5
done
while ! curl -s http://localhost:8003/v1/models; do
    echo "Waiting for API to be ready..."
    sleep 5
done

# API is ready, log this event
echo "API is ready. Running RunAPI.py" >> ErrorDetectionAtlasBench0Shot.log

# Run the Python script with real-time logging
python3 RunErrorDetection.py --path /fastwork/psalvador/JHU/data/AbdomenAtlas1.0Projections/projections_AtlasBench_beta_pro/ --port 8002 \
--csv_path ErrorDetectionAtlasBench0Shot.csv --continuing --organ_list liver --examples 0 --limit 1000 >> liver_ErrorDetectionAtlasBench0Shot.log 2>&1 &


# Run the Python script with real-time logging
python3 RunErrorDetection.py --path /fastwork/psalvador/JHU/data/AbdomenAtlas1.0Projections/projections_AtlasBench_beta_pro/ --port 8003 \
--csv_path ErrorDetectionAtlasBench0Shot.csv --continuing --organ_list kidneys --examples 0 --limit 1000 >> kidneys_ErrorDetectionAtlasBench0Shot.log 2>&1 &

wait

# Log completion
echo "RunAPI.py has finished executing." >> ErrorDetectionAtlasBench0Shot.log
