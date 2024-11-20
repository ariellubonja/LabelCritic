# Label Critic: Using LVLMs to Compare Medical Segmentations and Correct Label Errors

<p align="center">
  <img src="https://github.com/PedroRASB/Cerberus/blob/main/misc/LabelCriticModel.png" alt="Project Logo" width="900"/>
</p>


Label Critic is an automated tool for selecting the best AI-generated annotations among multiple options to streamline medical dataset labeling and revise existing datasets, substituting low-quality labels by better alternatives. Leveraging pre-trained Large Vision-Language Models (LVLMs) to perform pair-wise label comparisons, Label Critic achieves 96.5% accuracy in choosing the optimal label for each CT scan and class. Label Critic can also assess the quality of single AI annotations, flagging lower-quality cases for further review if necessary. Label Critic provides class-tailored prompts for evaluating and comparing CT's per-voxel annotations for pancreas, liver, stomach, spleen, gallbladder, kidneys, aorta and postcava. It also provides efortless adaptation to new classes.

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
conda install -y ipykernel
conda install -y pip
pip install vllm==0.6.1.post2
pip install git+https://github.com/huggingface/transformers@21fac7abba2a37fae86106f87fcf9974fd1e3830
pip install -r requirements.txt
mkdir HFCache
```

Deploy API locally (tp should be the number of GPUs, and it accepts only powers of 2)
```bash
TRANSFORMERS_CACHE=./HFCache HF_HOME=./HFCache CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve "Qwen/Qwen2-VL-72B-Instruct-AWQ" --dtype=half --tensor-parallel-size 4 --limit-mm-per-prompt image=3 --gpu_memory_utilization 0.9 --port 8000
```


## Project a dataset:
```bash
python3 ProjectDatasetFlex.py --good_folder /mnt/T9/AbdomenAtlasPro/ --bad_folder /mnt/sdc/pedro/JHH/nnUnetResultsBad/ --output_dir1 /projections/directory/ --num_processes 10 --file_list /mnt/sdc/pedro/ErrorDetection/ErrorLists/low_dice_benchmark_nnUnet_vs_JHH.txt
```

## Run label comparison over projected dataset 
### (faster than using high level API)

```bash
python3 RunAPI.py --path projections/directory/ > comparisons.log 2>&1
```

## Run Error Detection

```bash
python3 RunErrorDetection.py --path /mnt/sdc/pedro/ErrorDetection/good_labels_beta_full/ --port 8000 --organ [kidneys] --file_structure auto --examples 0 --good_examples_pth /mnt/sdc/pedro/ErrorDetection/good_labels_beta_full/kidneys/ --bad_examples_pth /mnt/sdc/pedro/ErrorDetection/errors_nnUnet_full/kidneys/ > organ.log 2>&1
```

Or, for running over all datasets:

```bash
bash /mnt/sdg/pedro/AnnotationVLM/RunED.sh --organ postcava --port 8000 --annotation_vlm_root /mnt/sdg/pedro/AnnotationVLM/ --error_detection_root /mnt/sdg/pedro/
```

# Citation

Bassi, Pedro & Wu, Qilong & Li, Wenxuan & Decherchi, Sergio & Cavalli, Andrea & Yuille, Alan & Zhou, Zongwei. (2024). Label Critic: Design Data Before Models. 10.48550/arXiv.2411.02753. 

```
@misc{bassi2024labelcriticdesigndata,
      title={Label Critic: Design Data Before Models}, 
      author={Pedro R. A. S. Bassi and Qilong Wu and Wenxuan Li and Sergio Decherchi and Andrea Cavalli and Alan Yuille and Zongwei Zhou},
      year={2024},
      eprint={2411.02753},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2411.02753}, 
}
```

<p align="center">
  <img src="https://github.com/PedroRASB/Cerberus/blob/main/misc/LabelCritic.png" alt="Project Logo" width="150"/>
</p>

