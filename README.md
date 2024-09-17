# AnnotationLVM

### Installation and running

Install
```bash
conda create -n lmdeploy python=3.11cond -y
conda activate lmdeploy
pip install lmdeploy==0.6.0
```

Deploy API locally (tp should be the number of GPUs)
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 lmdeploy serve api_server OpenGVLab/InternVL2-40B-AWQ --backend turbomind --server-port 23333 --model-format awq --tp 4 #--session-len 8192
```

Run label comparison code
```bash
python RunAPI.py
```

