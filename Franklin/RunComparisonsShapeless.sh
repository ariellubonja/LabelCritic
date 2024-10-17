#!/bin/sh
#PBS -l select=1:ncpus=60:ngpus=4
#PBS -l walltime=23:59:00
#PBS -j oe
#PBS -N shapComparisons0Shot
#PBS -q a100f

cd /fastwork/psalvador/JHU/AnnotationVLM

/fastwork/psalvador/JHU/AnnotationVLM/anaconda3/bin/conda init
source ~/.bashrc
conda activate vllm

TRANSFORMERS_CACHE=./HFCache HF_HOME=./HFCache CUDA_VISIBLE_DEVICES=0 vllm serve "Qwen/Qwen2-VL-72B-Instruct-AWQ" --dtype=half --tensor-parallel-size 1 --limit-mm-per-prompt image=3 --gpu_memory_utilization 0.95 --port 8001 > Shap1api.log 2>&1 &
TRANSFORMERS_CACHE=./HFCache HF_HOME=./HFCache CUDA_VISIBLE_DEVICES=1 vllm serve "Qwen/Qwen2-VL-72B-Instruct-AWQ" --dtype=half --tensor-parallel-size 1 --limit-mm-per-prompt image=4 --gpu_memory_utilization 0.95 --port 8002 > Shap2api.log 2>&1 &
TRANSFORMERS_CACHE=./HFCache HF_HOME=./HFCache CUDA_VISIBLE_DEVICES=2 vllm serve "Qwen/Qwen2-VL-72B-Instruct-AWQ" --dtype=half --tensor-parallel-size 1 --limit-mm-per-prompt image=12 --gpu_memory_utilization 0.95 --port 8003 > Shap3api.log 2>&1 &
TRANSFORMERS_CACHE=./HFCache HF_HOME=./HFCache CUDA_VISIBLE_DEVICES=3 vllm serve "Qwen/Qwen2-VL-72B-Instruct-AWQ" --dtype=half --tensor-parallel-size 1 --limit-mm-per-prompt image=22 --gpu_memory_utilization 0.95 --port 8004 > Shap4api.log 2>&1 &

# Check if the API is up
while ! curl -s http://localhost:8001/v1/models; do
    echo "Waiting for API to be ready..."
    sleep 5
done

while ! curl -s http://localhost:8002/v1/models; do
    echo "Waiting for API to be ready..."
    sleep 5
done

while ! curl -s http://localhost:8003/v1/models; do
    echo "Waiting for API to be ready..."
    sleep 5
done

while ! curl -s http://localhost:8004/v1/models; do
    echo "Waiting for API to be ready..."
    sleep 5
done


# Run the Python script with real-time logging
python3 RunAPI.py --path /fastwork/psalvador/JHU/data/AbdomenAtlas1.0Projections/projections_AtlasBench_beta_pro/ --port 8001 --shapeless --organ_list all_shapeless \
--csv_path ShapelessComparisonsAtlasBench0Shot.csv --continuing --dice_list /fastwork/psalvador/JHU/data/AbdomenAtlas1.0Projections/ >> ShapelessComparisonsAtlasBench0Shot.log 2>&1 &

python3 RunAPI.py --path /fastwork/psalvador/JHU/data/AbdomenAtlas1.0Projections/projections_AtlasBench_beta_pro/ --port 8002 --shapeless --examples 1 --organ_list all_shapeless \
--csv_path ShapelessComparisonsAtlasBench1Shot.csv --continuing --dice_list /fastwork/psalvador/JHU/data/AbdomenAtlas1.0Projections/ >> ShapelessComparisonsAtlasBench1Shot.log 2>&1 &

python3 RunAPI.py --path /fastwork/psalvador/JHU/data/AbdomenAtlas1.0Projections/projections_AtlasBench_beta_pro/ --port 8003 --shapeless --examples 5 --organ_list all_shapeless \
--csv_path ShapelessComparisonsAtlasBench5Shot.csv --continuing --dice_list /fastwork/psalvador/JHU/data/AbdomenAtlas1.0Projections/ >> ShapelessComparisonsAtlasBench5Shot.log 2>&1 &

python3 RunAPI.py --path /fastwork/psalvador/JHU/data/AbdomenAtlas1.0Projections/projections_AtlasBench_beta_pro/ --port 8004 --shapeless --examples 10 --organ_list all_shapeless \
--csv_path ShapelessComparisonsAtlasBench10Shot.csv --continuing --dice_list /fastwork/psalvador/JHU/data/AbdomenAtlas1.0Projections/ >> ShapelessComparisonsAtlasBench10Shot.log 2>&1 &

wait