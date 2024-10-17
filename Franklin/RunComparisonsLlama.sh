#!/bin/sh
#PBS -l select=1:ncpus=120:ngpus=8
#PBS -l walltime=23:59:00
#PBS -j oe
#PBS -N Comparisons0Shot
#PBS -q a100f

cd /fastwork/psalvador/JHU/AnnotationVLM

/fastwork/psalvador/JHU/AnnotationVLM/anaconda3/bin/conda init
source ~/.bashrc
conda activate llama

export HF_TOKEN=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

TRANSFORMERS_CACHE=./HFCache HF_HOME=./HFCache vllm serve "meta-llama/Llama-3.2-90B-Vision-Instruct" --tensor-parallel-size 8 --limit-mm-per-prompt image=2 --gpu_memory_utilization 0.95 --port 8888 > api.log 2>&1 &

# Check if the API is up
while ! curl -s http://localhost:8888/v1/models; do
    echo "Waiting for API to be ready..."
    sleep 5
done

# API is ready, log this event
echo "API is ready. Running RunAPI.py" >> ComparisonsAtlasBench0ShotLlama90B.log

# Run the Python script with real-time logging
python3 RunAPI.py --path /fastwork/psalvador/JHU/data/AbdomenAtlas1.0Projections/projections_AtlasBench_beta_pro/ \
--csv_path ComparisonsAtlasBench0ShotLlama90B.csv --continuing --dice_list /fastwork/psalvador/JHU/data/AbdomenAtlas1.0Projections/ --port 8888 >> ComparisonsAtlasBench0ShotLlama90B.log 2>&1

# Log completion
echo "RunAPI.py has finished executing." >> ComparisonsAtlasBench0ShotLlama90B.log
