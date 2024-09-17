# AnnotationLVM

### Installation and running

Install
```bash
conda create -n lmdeploy python=3.11cond -y
conda activate lmdeploy
conda install pip
pip install lmdeploy==0.6.0
pip install timm
```

Deploy API locally (tp should be the number of GPUs)
```bash
mkdir HFCache
export TRANSFORMERS_CACHE=./HFCache
export HF_HOME=./HFCache
CUDA_VISIBLE_DEVICES=0,1,2,3 lmdeploy serve api_server OpenGVLab/InternVL2-40B-AWQ --backend turbomind --server-port 23333 --model-format awq --tp 4 --session-len 8192
#use --cache-max-entry-count 0.5 to reduce memory cost
```

Run label comparison code
```bash
python RunAPI.py
```

