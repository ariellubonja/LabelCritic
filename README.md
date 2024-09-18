# AnnotationLVM

### Installation and running

Install
```bash
git clone https://github.com/PedroRASB/AnnotationVLM
cd AnnotationVLM
conda create -n lmdeploy python=3.11
conda activate lmdeploy
conda install ipykernel
conda install pip
pip install -r requirements.txt
```

Deploy API locally (tp should be the number of GPUs)
```bash
mkdir HFCache
export TRANSFORMERS_CACHE=./HFCache
export HF_HOME=./HFCache
CUDA_VISIBLE_DEVICES=0,1,2,3 lmdeploy serve api_server OpenGVLab/InternVL2-40B-AWQ --backend turbomind --server-port 23333 --model-format awq --tp 4 --session-len 8192
#use --cache-max-entry-count 0.5 to reduce memory cost
```

Make Call to API (Python)
```python
import ErrorDetector as ed

ct='path/to/ct.nii.gz'
y1='path/to/segmentation_1.nii.gz'
y2='path/to/segmentation_2.nii.gz'

answer=ed.project_and_compare(ct,y1,y2)
```


