#!/bin/sh
#PBS -l select=1:ncpus=20:ngpus=2
#PBS -l walltime=23:59:00
#PBS -j oe
#PBS -N Comparisons0Shot
#PBS -q a100f

cd /fastwork/psalvador/JHU/AnnotationVLM

/fastwork/psalvador/JHU/AnnotationVLM/anaconda3/bin/conda init
source ~/.bashrc
conda activate vllm

TRANSFORMERS_CACHE=./HFCache HF_HOME=./HFCache CUDA_VISIBLE_DEVICES=0,1 vllm serve "Qwen/Qwen2-VL-72B-Instruct-AWQ" --dtype=half --tensor-parallel-size 2 --limit-mm-per-prompt image=3 --gpu_memory_utilization 0.9 --port 8000 > api.log 2>&1 &

# Check if the API is up
while ! curl -s http://localhost:8000/v1/models; do
    echo "Waiting for API to be ready..."
    sleep 5
done

# API is ready, log this event
echo "API is ready. Running RunAPI.py" >> ComparisonsAtlasBench0Shot.log

# Run the Python script with real-time logging
python3 RunAPI.py --path /fastwork/psalvador/JHU/data/AbdomenAtlas1.0Projections/projections_AtlasBench_beta_pro/ \
--csv_path ComparisonsAtlasBench0Shot.csv --continuing --dice_list /fastwork/psalvador/JHU/data/AbdomenAtlas1.0Projections/ >> ComparisonsAtlasBench0Shot.log 2>&1

# Log completion
echo "RunAPI.py has finished executing." >> ComparisonsAtlasBench0Shot.log
