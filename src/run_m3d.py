from ast import arg
import numpy as np
import torch, warnings, os
import nibabel as nib
from transformers import AutoTokenizer, AutoModelForCausalLM
import AnnotationVLM.src.custom_tf as ctf
from importlib import reload
reload(ctf)

warnings.filterwarnings("ignore")
os.environ['TRANSFORMERS_CACHE'] = './HFCache'
os.environ['HF_HOME'] = './HFCache'
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'

device = torch.device('cuda') # 'cpu', 'cuda'
dtype = torch.bfloat16 # or bfloat16, float16, float32

model_name_or_path = '/mnt/sdh/qwu59/ckpts/m3d/M3D-LaMed-Phi-3-4B'
proj_out_num = 256

model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    torch_dtype=dtype,
    device_map='auto',
    trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(
    model_name_or_path,
    model_max_length=512,
    padding_side="right",
    use_fast=False,
    trust_remote_code=True
)
model = model.to(device=device)

def inference(question, ct_pro, seg_enable=False, branch='ct'):
    image_tokens = "<im_patch>" * proj_out_num
    input_txt = image_tokens + question
    input_id = tokenizer(
        input_txt,
        return_tensors="pt"
    )['input_ids'].to(device=device)
    attention_mask = tokenizer(
        input_txt,
        return_tensors="pt", 
        padding=True
    )['attention_mask'].to(device=device)

    if branch == 'ct':
        image_pt = ct_pro.transformed["image"].unsqueeze(0).to(device=device).type(dtype)
    elif branch == 'ctmin':
        image_pt = ct_pro.ctmin_transformed["image"].unsqueeze(0).to(device=device).type(dtype)

    try:
        outputs = model.generate(
            image_pt,
            input_id,
            attention_mask=attention_mask,
            seg_enable=seg_enable,
            max_new_tokens=256,
            do_sample=True,
            top_p=0.9,
            temperature=1.0
        )
        if seg_enable:
            generation, seg_logit = outputs
            seg_mask = ((torch.sigmoid(seg_logit) > 0.5) * 1.0).squeeze(0)
            return tokenizer.batch_decode(generation, skip_special_tokens=True)[0], seg_mask
        else:
            generation = outputs
            return tokenizer.batch_decode(generation, skip_special_tokens=True)[0]
    except:
        return "The image is damaged or the model is not able to generate the answer."

import json, csv
from tqdm import tqdm

def append_dict_to_csv(dict_data, csv_path):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=dict_data.keys())
        if file.tell() == 0:
            writer.writeheader()
        writer.writerow(dict_data)
        
def step_1_q(organ):
    return "Does the bone ct image contain the {}? Answer yes or no.".format(organ)

def step_2_q(organ):
    return (
        "The lowest intensity white area within the body in this bone ct image"
        "is the {} mask annotation. ".format(organ) +
        "What do you think of this? Is it correct? Only answer yes or no."
    )
    
result_path = "results/m3d/"

task1_path = "/mnt/sdh/pedro/AbdomenAtlasBeta/"
task1 = "bad_labels_AbdomenAtlasBeta.json"

task2_path = "/mnt/sdc/pedro/JHH/nnUnetResults"
task2_path_ = "/mnt/sdc/pedro/ErrorDetection/cropped_nnunet_results_250Epch_liver"
task2 = "bad_labels_nnUnet.json"

task3_path = "/mnt/sdh/pedro/AbdomenAtlasBeta/"
task3 = "good_labels_AbdomenAtlasBeta.json"

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--task", type=int, default=2)
parser.add_argument("--organs", nargs='+', default=["liver", "kidney", "spleen"])

# --task 2 --organs gall_bladder
# --task 2 --organs postcava
# --task 2 --organs stomach
# --task 2 --organs liver kidneys spleen
# --task 2 --organs aorta

args = parser.parse_args()

if args.task == 2:
    # load the json file for task 1
    with open(task2) as f:
        task2_data = json.load(f)

    for i, organ in enumerate(tqdm(task2_data)):
        print("Organ:", organ)
        if organ not in args.organs:
            continue
        question1 = step_1_q(organ)
        question2 = step_2_q(organ)
        for j, case in enumerate(tqdm(task2_data[organ])):
            # check whether the case exists in the final csv
            check_table = os.path.join(result_path, "final", "errors_nnUnet_full.csv")
            skip_sign = False
            if os.path.exists(check_table):
                with open(check_table, mode='r') as file:
                    reader = csv.DictReader(file)
                    for row in reader:
                        if row["sample"] == case and row["organ"] == organ:
                            skip_sign = True
                            break
            if skip_sign:
                continue
            try:
                case_path = os.path.join(task2_path, case)
                # print("Case:", case, case_path)
                ct_pro = ctf.CTImageProcessor(case_path, ct_name="ct", mask_name=organ)
            except:
                case_path = os.path.join(task2_path_, case)
                ct_pro = ctf.CTImageProcessor(case_path, ct_name="ct", mask_name=organ)
            text1 = inference(question1, ct_pro, branch='ct')
            text2 = inference(question2, ct_pro, branch='ctmin')
            task2_raw = {
                "sample": case,
                "organ": organ,
                "part": "errors_nnUnet_full",
                "question1": question1,
                "answer1": text1,
                "question2": question2,
                "answer2": text2
            }
            task2_single = {
                "sample": case,
                "organ": organ,
                "part": "errors_nnUnet_full",
                "result step 1": "present" if "yes" in text1.lower() else "no",
                "label step 1": "present" if ct_pro.mask_present else "no",
                "result step 2": "Correct" if "yes" in text2.lower() else "Incorrect",
                "label step 2": "Incorrect",
            }
            print(task2_single)
            append_dict_to_csv(task2_raw, os.path.join(result_path, "raw", "errors_nnUnet_full.csv"))
            append_dict_to_csv(task2_single, os.path.join(result_path, "final", "errors_nnUnet_full.csv"))

elif args.task == 3:
    # load the json file for task 3
    with open(task3) as f:
        task3_data = json.load(f)

    for i, organ in enumerate(tqdm(task3_data)):
        print("Organ:", organ)
        question1 = step_1_q(organ)
        question2 = step_2_q(organ)
        for j, case in enumerate(tqdm(task3_data[organ])):
            # check whether the case exists in the final csv
            check_table = os.path.join(result_path, "final", "good_labels_beta_full.csv")
            skip_sign = False
            if os.path.exists(check_table):
                with open(check_table, mode='r') as file:
                    reader = csv.DictReader(file)
                    for row in reader:
                        if row["sample"] == case and row["organ"] == organ:
                            skip_sign = True
                            break
            if skip_sign:
                continue
            case_path = os.path.join(task3_path, case)
            # print("Case:", case, case_path)
            ct_pro = ctf.CTImageProcessor(case_path, ct_name="ct", mask_name=organ)
            text1 = inference(question1, ct_pro, branch='ct')
            text2 = inference(question2, ct_pro, branch='ctmin')
            task3_raw = {
                "sample": case,
                "organ": organ,
                "part": "good_labels_beta_full",
                "question1": question1,
                "answer1": text1,
                "question2": question2,
                "answer2": text2
            }
            task3_single = {
                "sample": case,
                "organ": organ,
                "part": "good_labels_beta_full",
                "result step 1": "present" if "yes" in text1.lower() else "no",
                "label step 1": "present" if ct_pro.mask_present else "no",
                "result step 2": "Correct" if "yes" in text2.lower() else "Incorrect",
                "label step 2": "Correct",
            }
            print(task3_single)
            append_dict_to_csv(task3_raw, os.path.join(result_path, "raw", "good_labels_beta_full.csv"))
            append_dict_to_csv(task3_single, os.path.join(result_path, "final", "good_labels_beta_full.csv"))
