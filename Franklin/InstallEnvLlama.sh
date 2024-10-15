cd /fastwork/psalvador/JHU/AnnotationVLM
./anaconda3/bin/conda init
source ~/.bashrc
conda create -n llama python=3.12 -y
conda activate llama
conda install -y ipykernel
conda install -y pip
pip install vllm
pip install -r requirements.txt