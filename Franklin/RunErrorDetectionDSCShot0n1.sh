#!/bin/sh
#PBS -l select=1:ncpus=20:ngpus=2
#PBS -l walltime=23:59:00
#PBS -j oe
#PBS -N DSC_EDMulti
#PBS -q gpu_a100

cd /fastwork/psalvador/JHU/AnnotationVLM

/fastwork/psalvador/JHU/AnnotationVLM/anaconda3/bin/conda init
source ~/.bashrc
conda activate vllm

TRANSFORMERS_CACHE=./HFCache HF_HOME=./HFCache CUDA_VISIBLE_DEVICES=0 vllm serve "Qwen/Qwen2-VL-72B-Instruct-AWQ" --dtype=half --tensor-parallel-size 1 --limit-mm-per-prompt image=3 --gpu_memory_utilization 0.95 --port 8001 > DSC_EDapi.log 2>&1 &
TRANSFORMERS_CACHE=./HFCache HF_HOME=./HFCache CUDA_VISIBLE_DEVICES=1 vllm serve "Qwen/Qwen2-VL-72B-Instruct-AWQ" --dtype=half --tensor-parallel-size 1 --limit-mm-per-prompt image=3 --gpu_memory_utilization 0.95 --port 8002 > DSC_EDapi.log 2>&1 &
#TRANSFORMERS_CACHE=./HFCache HF_HOME=./HFCache CUDA_VISIBLE_DEVICES=2 vllm serve "Qwen/Qwen2-VL-72B-Instruct-AWQ" --dtype=half --tensor-parallel-size 1 --limit-mm-per-prompt image=4 --gpu_memory_utilization 0.95 --port 8003 > DSC_EDapi.log 2>&1 &
#TRANSFORMERS_CACHE=./HFCache HF_HOME=./HFCache CUDA_VISIBLE_DEVICES=3 vllm serve "Qwen/Qwen2-VL-72B-Instruct-AWQ" --dtype=half --tensor-parallel-size 1 --limit-mm-per-prompt image=11 --gpu_memory_utilization 0.95 --port 8004 > DSC_EDapi.log 2>&1 &

# Check if the API is up
while ! curl -s http://localhost:8001/v1/models; do
    echo "Waiting for API to be ready..."
    sleep 5
done

# Check if the API is up
while ! curl -s http://localhost:8002/v1/models; do
    echo "Waiting for API to be ready..."
    sleep 5
done

# Check if the API is up
#while ! curl -s http://localhost:8003/v1/models; do
#    echo "Waiting for API to be ready..."
#    sleep 5
#done

# Check if the API is up
##while ! curl -s http://localhost:8004/v1/models; do
#    echo "Waiting for API to be ready..."
#    sleep 5
#done

# Run the Python script with real-time logging
python3 RunErrorDetection.py --path /fastwork/psalvador/JHU/data/AbdomenAtlas1.0Projections/projections_AtlasBench_beta_pro/ --port 8001 \
--csv_path DSC_ErrorDetectionAtlasBench0Shot.csv --continuing --organ_list all --examples 0 \
--dice_list /fastwork/psalvador/JHU/data/AbdomenAtlas1.0Projections/ --dice_check >> DSC_ErrorDetectionAtlasBench0Shot.log 2>&1 &

python3 RunErrorDetection.py --path /fastwork/psalvador/JHU/data/AbdomenAtlas1.0Projections/projections_AtlasBench_beta_pro/ --port 8002 \
--csv_path DSC_ErrorDetectionAtlasBench1Shot.csv --continuing --organ_list all --examples 1 \
--dice_list /fastwork/psalvador/JHU/data/AbdomenAtlas1.0Projections/ --dice_check >> DSC_ErrorDetectionAtlasBench1Shot.log 2>&1 &

#python3 RunErrorDetection.py --path /fastwork/psalvador/JHU/data/AbdomenAtlas1.0Projections/projections_AtlasBench_beta_pro/ --port 8003 \
#--csv_path DSC_ErrorDetectionAtlasBench2Shot.csv --continuing --organ_list all --examples 2 \
#--dice_list /fastwork/psalvador/JHU/data/AbdomenAtlas1.0Projections/ --dice_check >> DSC_ErrorDetectionAtlasBench2Shot.log 2>&1 &

#python3 RunErrorDetection.py --path /fastwork/psalvador/JHU/data/AbdomenAtlas1.0Projections/projections_AtlasBench_beta_pro/ --port 8004 \
#--csv_path DSC_ErrorDetectionAtlasBench10Shot.csv --continuing --organ_list all --examples 10 \
#--dice_list /fastwork/psalvador/JHU/data/AbdomenAtlas1.0Projections/ --dice_check >> DSC_ErrorDetectionAtlasBench10Shot.log 2>&1 &

wait

