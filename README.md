# Use VLM to compare the per-voxel organ annotations of 2 semantic segmenters

<p align="center">
  <img src="https://github.com/PedroRASB/Cerberus/blob/main/misc/Cerberus.png" alt="Project Logo" width="250"/>
</p>

### Installation and running

<details>
<summary style="margin-left: 25px;">[Optional] Install Anaconda on Linux</summary>
<div style="margin-left: 25px;">
    
```bash
wget https://repo.anaconda.com/archive/Anaconda3-2024.06-1-Linux-x86_64.sh
bash Anaconda3-2024.06-1-Linux-x86_64.sh -b -p ./anaconda3
./anaconda3/bin/conda init
source ~/.bashrc
```
</div>
</details>

Install
```bash
git clone https://github.com/PedroRASB/AnnotationVLM
cd AnnotationVLM
conda create -n vllm python=3.12 -y
conda activate vllm
conda install ipykernel
conda install pip
pip install vllm==0.6.1.post2
pip install git+https://github.com/huggingface/transformers@21fac7abba2a37fae86106f87fcf9974fd1e3830
pip install -r requirements.txt
mkdir HFCache
```

Deploy API locally (tp should be the number of GPUs, and it accepts only powers of 2)
```bash
TRANSFORMERS_CACHE=./HFCache HF_HOME=./HFCache CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve "Qwen/Qwen2-VL-72B-Instruct-AWQ" --dtype=half --tensor-parallel-size 4 --limit-mm-per-prompt image=3 --gpu_memory_utilization 0.9 --port 8000
```

# High level API 
```python
import ErrorDetector as ed

ct='path/to/ct.nii.gz'
y1='path/to/segmentation_1.nii.gz'
y2='path/to/segmentation_2.nii.gz'

answer=ed.project_and_compare(ct,y1,y2)
```
Example: see MyAPITest.ipynb

# Project a dataset:
```bash
python3 ProjectDatasetFlex.py --good_folder /mnt/T9/AbdomenAtlasPro/ --bad_folder /mnt/sdc/pedro/JHH/nnUnetResultsBad/ --output_dir1 /projections/directory/ --num_processes 10 --file_list /mnt/sdc/pedro/ErrorDetection/ErrorLists/low_dice_benchmark_nnUnet_vs_JHH.txt
```

# Run label comparison over projected dataset 
### (faster than using high level API)

```bash
python3 RunAPI.py --path projections/directory/organ/ > comparisons.log 2>&1
```

# Run Error Detection

```bash
python3 RunErrorDetection.py --path /mnt/sdc/pedro/ErrorDetection/good_labels_beta_full/ --port 8000 --organ [kidneys] --file_structure auto --examples 0 --good_examples_pth /mnt/sdc/pedro/ErrorDetection/good_labels_beta_full/kidneys/ --bad_examples_pth /mnt/sdc/pedro/ErrorDetection/errors_nnUnet_full/kidneys/ > organ.log 2>&1
```

Or, for running over all datasets:

```bash
bash /mnt/sdg/pedro/AnnotationVLM/RunED.sh --organ postcava --port 8000 --annotation_vlm_root /mnt/sdg/pedro/AnnotationVLM/ --error_detection_root /mnt/sdg/pedro/
```
