from IPython.display import display
import os
import random
import requests
from PIL import Image
import torch
#from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration
import numpy as np
import gc
import io
import base64
import gc
try:
    from . import projection as prj
except:
    import projection as prj
import tempfile
import shutil

import matplotlib.pyplot as plt

ZeroShotInstructions=("The image I am sending is a frontal projection of a CT scan. "
                      "It is not a CT slice, we have transparency and can see through the entire body, "
                      "like a X-ray. The left side of the image represents the right side of the human body. "
                      "The %(organ)s region in the image should be marked in red, "
                      "using an overlay. However, I am not sure if the red overlay correctly "
                      "or incorrectly marks the %(organ)s. Check if the red region is corerent with "
                      "the expected shape and location of a %(organ)s.")
SummarizeInstructions=("Summarize your last answer, using only 2 words: "
                       "'good annotation' or 'bad annotation'.")
SummarizeInstructionsFewShot=("Summarize your last answer, using only 2 words: "
                            "'good annotation' or 'bad annotation'. "
                            "Is the annotation for the last image I sent a 'good annotation' or a 'bad annotation'?")

OneShotFirstPart=("The images I am sending are a frontal projections of a CT scans. "
                     "They are not CT slices, instead, they have transparency and let you see throgh "
                     "the entire human body, like an X-ray does. "
                      "The left side of the image represents the right side of the human body. "
                      "The %(organ)s region in the images should "
                     "be marked in red, using an overlay. However, I am not sure if the red overlay correctly "
                      "or incorrectly marks the %(organ)s. You must if the red region is corerent with "
                      "the expected shape and location of a %(organ)s."
                      "However, now I am sending you just an example image, which is a 'good annotation'. "
                      "I will send you the image for evaluation after you see the example. "
                      "Take a good look and learn with it.")

FewShotFirstPart=("The images I am sending are a frontal projections of a CT scans. "
                     "They are not CT slices, instead, they have transparency and let you see throgh "
                     "the entire human body, like an X-ray does. "
                      "The left side of the image represents the right side of the human body. "
                      "The %(organ)s region in the images should "
                     "be marked in red, using an overlay. However, I am not sure if the red overlay correctly "
                      "or incorrectly marks the %(organ)s. You must if the red region is corerent with "
                      "the expected shape and location of a %(organ)s."
                      "However, now I am sending you just an example image, which is a 'good annotation'. "
                      "I will send mutiple examples in diverse prompts. "
                      "Take a good look and learn with the examples.")

OneShotSecondPart=("I am sending you now the image for evaluaiton. Check if it has a 'good annotation' or a 'bad annotation'. "
                  "The example image should help you evaluate it. Think throughly and provide a detailed explanation for "
                  "why the last image represents a 'good annotation' or a 'bad annotation'.")


FewShotSecondPart=("I am sending you now the image for evaluaiton. Check if the red overlay on it is a 'good annotation' or a 'bad annotation'. "
                  "The example images should help you evaluate it. Think throughly and provide a detailed explanation for "
                  "why the last image represents a 'good annotation' or a 'bad annotation'.")
                  

#OneShotInstructions=("The image I am sending are a frontal projections of a CT scans. "
#                     "It is not CT slices, instead, they have transparency and let you see throgh "
#                     "the entire human body, like an X-ray does. The left side of the image represents the right side of the human body. The %(organ)s region in the images should "
#                     "be marked in red, using an overlay. However, some images present correct overlays, "
#                     "while others present incorrect overlays. Image 1 is an example of a good %(organ)s annotation."
#                     "I want you to evaluate the correctness of the %(organ)s annotation in the second image. "
#                     "The example (Image 1) should help you evaluate it. Think throughly and provide a detailed explanation for "
#                     "why the last image represents a 'good annotation' or a 'bad annotation'.")

#FewShotInstructions=("The images I am sending are a frontal projections of a CT scans. "
#                     "They are not CT slices, instead, they have transparency and let you see throgh "
#                     "the entire human body, like an X-ray does. The left side of the image represents the right side of the human body. The %(organ)s region in the images should "
 #                    "be marked in red, using an overlay. However, some images present correct overlays, "
 #                    "while others present incorrect overlays. The first %(examples)s are examples, and "
 #                    "I will inform you (below) if each one of them is a 'good annotation' or a 'bad annotation': \n")

#FewShotInstructionsEnd=("I want you to evaluate the correctness of the %(organ)s annotation in the last image, %(last)s. "
#                        "The examples should help you evaluate it. Think throughly and provide a detailed explanation for "
#                        "why image %(last)s represents a 'good annotation' or a 'bad annotation'.")

liver=(" Consider the following anatomical information: the liver is a large, triangular organ located in the upper right quadrant of the abdomen, "
      "just below the diaphragm. It is a single structure. It spans across the midline, partially extending"
      " into the left upper quadrant. "
      "The liver position is primarily under the rib cage. In assessing the annotation, I want you to answer the following questions: \n"
      "1. Is the red overlay a single contiguous object? \n"
      "2. Does the shape of the red overlay resemble the typical triangular or wedge-like shape of the liver? \n"
      "3. Is the red overlay primarily located in the upper right quadrant of the abdomen, just below the diaphragm? \n")
liver_no_question=(" Consider the following anatomical information: the liver is a large, triangular organ located in the upper right quadrant of the abdomen, "
      "just below the diaphragm. It is a single structure. It spans across the midline, partially extending"
      " into the left upper quadrant. "
      "The liver position is primarily under the rib cage. \n")

liver_describe=(" Consider the following anatomical information: the liver is a large, triangular organ located in the upper right quadrant of the abdomen, "
      "just below the diaphragm. It is a single structure. It spans across the midline, partially extending"
      " into the left upper quadrant. "
      "The liver position is primarily under the rib cage. \n"
      "Throuhgly describe the overlay: what is its shape? where is it? Then you say if it corresponds to the liver or not. ")

KidneysDescriptionED=("Consider the following anatomical information:\n"
                   "a) A person usually has two kidneys. Does Image 1 show two distinct kidney overlays? Does Image 2 show two distinct kidney overlays? If both show a single kidney, the patient may truly have only one kidney.  \n"
                   "b) Each kidney has a bean-shaped structure, with a convex lateral surface and a concave medial surface. \n"
                   "c) The kidneys are located in the retroperitoneal space, on either side of the spine, at the level of the lower ribs. \n")


AdrenalGlandDescriptionED=("Consider the following anatomical information:\n"
                   "a) Number: a person usually has two adrenal glands, one on top of each kidney.\n"
                   "b) Location: the adrenal glands are located on the superior aspect of each kidney, in the retroperitoneal space.\n"
                   "c) Right Adrenal Gland Shape (right side of the body): triangular shape.\n"
                   "d) Left Adrenal Gland Shape (left side of the body): Generally crescent-shaped or semilunar. May appear as a curved line or elongated structure above the kidney.\n"
                   "e) Size: adrenal glands are relatively small compared to the kidneys.\n")

DescriptionsED={"liver":liver_no_question,
              "kidneys":KidneysDescriptionED,
              "adrenal_glands":AdrenalGlandDescriptionED}


organ_descriptions={'liver':liver}

def resize_image(img, size):
    # Get the original dimensions
    width, height = img.size
    
    # Determine the scaling factor based on the largest axis
    if width > height:
        new_width = size
        new_height = int((size / width) * height)
    else:
        new_height = size
        new_width = int((size / height) * width)
    
    # Resize the image with the new dimensions
    img = img.resize((new_width, new_height))
    
    return img

def ZeroShot(img,processor,model,text=ZeroShotInstructions,organ='liver',size=None):
    text=text%{'organ':organ}
    print(text)
    if isinstance(img, str):
        img = Image.open(img)
    if size is not None:
        img=resize_image(img,size)

    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": text}
                ],
        }
    ]
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = processor(text=prompt, images=[img],
                       return_tensors="pt").to(model.device).to(torch.float16)
    generate_kwargs = {"max_new_tokens": 400, "do_sample": False}
    output = model.generate(**inputs, **generate_kwargs)
    generated_text = processor.batch_decode(output, skip_special_tokens=True)
    return generated_text

def ZeroShot2Steps(img,processor,model,text=ZeroShotInstructions,
                    text2=SummarizeInstructions,
                    organ='liver',size=None):
    text=text%{'organ':organ}
    if isinstance(img, str):
        img = Image.open(img)#.resize((224,224))
    
    if size is not None:
        img=resize_image(img,size)

    generated_text=ZeroShot(img=img,text=text,processor=processor,model=model,size=size)

    model_text=generated_text[0][generated_text[0].rfind('assistant\n')+len('assistant\n'):]

    print('First answer:',model_text)
    conversation = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": text},
            {"type": "image"},  # Placeholder for the first image
        ],
    },
    {
        "role": "assistant",
        "content": [
            {"type": "text", "text": model_text}
        ],
    },
    {
        "role": "user",
        "content": [
            {"type": "text", "text": text2}
            # No image for the second user input
        ],
    },
]
    
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = processor(text=prompt, images=[img],
                       return_tensors="pt").to(model.device).to(torch.float16)
    generate_kwargs = {"max_new_tokens": 20, "do_sample": False}
    output = model.generate(**inputs, **generate_kwargs)
    generated_text = processor.batch_decode(output, skip_special_tokens=True)
    answer=generated_text[0][generated_text[0].rfind('assistant\n')+len('assistant\n'):]
    print('Second answer:',answer)

    if 'good annotation' in answer.lower() and ' not ' not in answer.lower():
        return 1.0
    elif 'bad annotation' in answer.lower() and ' not ' not in answer.lower():
        return 0.0
    else:
        return 0.5

def OneShot(img,good_example,processor,model,text1=OneShotFirstPart,
            text2=OneShotSecondPart,organ='liver',size=None):
    text1=text1%{'organ':organ}

    if isinstance(img, str):
        img = Image.open(img)#.resize((224,224))
    if isinstance(good_example, str):
        good_example = Image.open(good_example)
    if size is not None:
        img=resize_image(img,size)
        good_example=resize_image(good_example,size)

    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": text1},
                {"type": "image"}
                ],
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": 'Thank you for the example. The liver looks correct.'},
                ],
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": text2},
                {"type": "image"}
                ],
        },
    ]
    print(conversation)
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = processor(text=prompt, images=[good_example,img],
                       return_tensors="pt").to(model.device).to(torch.float16)
    generate_kwargs = {"max_new_tokens": 400, "do_sample": False}
    output = model.generate(**inputs, **generate_kwargs)
    generated_text = processor.batch_decode(output, skip_special_tokens=True)
    return generated_text

def OneShot2Steps(img,good_example,processor,model,text1=OneShotFirstPart,
            text2=OneShotSecondPart,organ='liver',size=None):
    text1=text1%{'organ':organ}
    if isinstance(img, str):
        img = Image.open(img)#.resize((224,224))
    if isinstance(good_example, str):
        good_example = Image.open(good_example)
    if size is not None:
        img=resize_image(img,size)
        good_example=resize_image(good_example,size)
    
    generated_text=OneShot(img=img,good_example=good_example,text1=text1,processor=processor,
                           model=model,size=size,text2=text2)

    model_text=generated_text[0][generated_text[0].rfind('assistant\n')+len('assistant\n'):]

    print('First answer:',model_text)
    conversation = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": text1},
            {"type": "image"}
            ],
    },
    {
        "role": "assistant",
        "content": [
            {"type": "text", "text": 'Thank you for the example. The liver looks correct.'},
            ],
    },
    {
        "role": "user",
        "content": [
            {"type": "text", "text": text2},
            {"type": "image"}
            ],
    },
    {
        "role": "assistant",
        "content": [
            {"type": "text", "text": model_text}
        ],
    },
    {
        "role": "user",
        "content": [
            {"type": "text", "text": SummarizeInstructions}
            # No image for the second user input
        ],
    },
]
    
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = processor(text=prompt, images=[good_example,img],
                       return_tensors="pt").to(model.device).to(torch.float16)
    generate_kwargs = {"max_new_tokens": 20, "do_sample": False}
    output = model.generate(**inputs, **generate_kwargs)
    generated_text = processor.batch_decode(output, skip_special_tokens=True)
    answer=generated_text[0][generated_text[0].rfind('assistant\n')+len('assistant\n'):]
    print('Second answer:',answer)

    if 'good annotation' in answer.lower() and ' not ' not in answer.lower():
        return 1.0
    elif 'bad annotation' in answer.lower() and ' not ' not in answer.lower():
        return 0.0
    else:
        return 0.5


def FewShot(img,good_examples,bad_examples,processor,model,
            text1=FewShotFirstPart+liver_no_question+' If the overlay has any clear error, consider it a bad annotation.',
            text2=FewShotSecondPart+liver_no_question+' If the overlay has any clear error, consider it a bad annotation.',
            organ='liver',organ_descriptions=organ_descriptions,
            size=None):
    if isinstance(img, str):
        img = Image.open(img)#.resize((224,224))
    if size is not None:
        img=resize_image(img,size)

    tmp=[]
    for good_example in good_examples:
        if isinstance(good_example, str):
            good_example = Image.open(good_example)
        if size is not None:
            good_example=resize_image(good_example,size)
        tmp.append(good_example)
    good_examples=tmp
    tmp=[]
    for bad_example in bad_examples:
        if isinstance(bad_example, str):
            bad_example = Image.open(bad_example)
        if size is not None:
            bad_example=resize_image(bad_example,size)
        tmp.append(bad_example)
    bad_examples=tmp
    del tmp

    text1=text1%{'organ':organ}

    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": text1},
                {"type": "image"}
                ],
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Thank you for the example. The red overlay is indeed a 'good annotation' for {organ}."%{'organ':organ}},
                ],
        }
    ]

    last='good'
    examples=[]
    examples.append(good_examples[0])
    good_i=1
    bad_i=0

    for i in range(len(good_examples)+len(bad_examples)-1):
        if last=='good' and bad_i<len(bad_examples):
            current='bad'
        elif last=='bad' and good_i<len(good_examples):
            current='good'
        elif last=='good' and bad_i>=len(bad_examples):
            current='good'
        elif last=='bad' and good_i>=len(good_examples):
            current='bad'

        if current=='good':
            examples.append(good_examples[good_i])
            conversation.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": f"This image is a 'good annotation' for {organ}."%{'organ':organ}},
                    {"type": "image"}
                    ],
            })
            conversation.append(
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": f"Thank you for the example. The red overlay is indeed a 'good annotation' for {organ}."%{'organ':organ}},
                    ],
            })
            good_i+=1
            last='good'
        else:
            examples.append(bad_examples[bad_i])
            conversation.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": f"This image is a 'bad annotation' for {organ}."%{'organ':organ}},
                    {"type": "image"}
                    ],
            })
            conversation.append(
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": f"Thank you for the example. The red overlay is indeed a 'bad annotation' for {organ}."%{'organ':organ}},
                    ],
            })
            bad_i+=1
            last='bad'

    conversation.append({
            "role": "user",
            "content": [
                {"type": "text", "text": text2},
                {"type": "image"}
                ],
        })

    print(conversation)

    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = processor(text=prompt, images=examples+[img],
                       return_tensors="pt").to(model.device).to(torch.float16)
    generate_kwargs = {"max_new_tokens": 200, "do_sample": False}
    output = model.generate(**inputs, **generate_kwargs)
    generated_text = processor.batch_decode(output, skip_special_tokens=True)
    return generated_text,conversation,examples+[img]

def FewShot2Steps(img,good_examples,bad_examples,processor,model,
                  text1=FewShotFirstPart+liver_no_question+' If the overlay has any clear error, consider it a bad annotation.',
                    text2=FewShotSecondPart+liver_no_question+' If the overlay has any clear error, consider it a bad annotation.',
                    organ='liver',
                    summarize_instructions=SummarizeInstructionsFewShot,
                    prt=False,size=None):

    generated_text,conversation,images=FewShot(img=img,good_examples=good_examples,
                                               bad_examples=bad_examples,text1=text1,text2=text2,
                                               organ=organ,processor=processor,model=model,size=size)

    if prt:
        for image in images:
            display(image)

    model_text=generated_text[0][generated_text[0].rfind('assistant\n')+len('assistant\n'):]
    print('First answer:',model_text)

    conversation.append({
                            "role": "assistant",
                            "content": [
                                {"type": "text", "text": model_text}
                            ],
                        })
    
    conversation.append({
                            "role": "user",
                            "content": [
                                {"type": "text", "text": summarize_instructions}
                            ],
                        })
    
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = processor(text=prompt, images=images, return_tensors="pt").to(model.device).to(torch.float16)
    generate_kwargs = {"max_new_tokens": 20, "do_sample": False}
    output = model.generate(**inputs, **generate_kwargs)
    generated_text = processor.batch_decode(output, skip_special_tokens=True)
    answer=generated_text[0][generated_text[0].rfind('assistant\n')+len('assistant\n'):]
    print('Second answer:',answer)

    if 'good annotation' in answer.lower() and ' not ' not in answer.lower():
        return 1.0
    elif 'bad annotation' in answer.lower() and ' not ' not in answer.lower():
        return 0.0
    else:
        return 0.5
    
def get_random_file_paths(folder_path, n, exclude_path,contains='overlay_axis_1'):
    # Get all file paths in the specified folder
    if isinstance(folder_path, list):
        all_files = [file for file in folder_path 
                    if (os.path.isfile(file) and (contains in file) \
                    and (exclude_path not in file))]
    else:
        all_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) 
                    if (os.path.isfile(os.path.join(folder_path, file)) and (contains in file) \
                    and (exclude_path not in file))]


    # Get n random file paths from the list
    random_files = random.sample(all_files, min(n, len(all_files)))

    print('Example files:',random_files)

    return random_files

#say people have annatomical variations and the liver must not look perfect? it you have too many FP only

def calculate_accuracy(answers_good, answers_bad):
    """
    Calculates the accuracy of binary predictions given two vectors:
    answers_good (should ideally be all 1s) and answers_bad (should ideally be all 0s).

    Parameters:
        answers_good (np.ndarray): A vector of binary predictions where the expected values are 1.
        answers_bad (np.ndarray): A vector of binary predictions where the expected values are 0.

    Returns:
        float: The accuracy of the predictions.
    """
    answers_good, answers_bad = np.array(answers_good), np.array(answers_bad)

    # Combine predictions
    predictions = np.concatenate((answers_good, answers_bad))

    # Create ground truth vector
    ground_truth = np.concatenate((np.ones_like(answers_good), np.zeros_like(answers_bad)))

    # Calculate the number of correct predictions
    correct_predictions = np.sum(predictions == ground_truth)

    # Calculate accuracy
    accuracy = correct_predictions / len(predictions)
    
    return accuracy

def ZeroShotSystematicEval(good_annos,bad_annos,model,processor,
                           text=ZeroShotInstructions):
    answers_good=[]
    for target in good_annos:
        if 'overlay_axis_1' not in target:
            continue
        print(target)
        Image.open(target).show()
        answers_good.append(ZeroShot2Steps(img=target,
                        model=model,processor=processor,
                        text=text))
        print('Traget:',target,'Answer:',answers_good[-1])

    
    answers_bad=[]
    for target in bad_annos:
        if 'overlay_axis_1' not in target:
            continue
        print(target)
        Image.open(target).show()
        answers_bad.append(ZeroShot2Steps(img=target,
                        model=model,processor=processor,
                        text=text))
        print('Traget:',target,'Answer:',answers_bad[-1])

    print('Correct good annotation: ',np.array(answers_good).sum(),'/',len(answers_good))
    print('Correct bad annotation: ',(1-np.array(answers_bad)).sum(),'/',len(answers_bad))

    print('Accuracy: ',calculate_accuracy(answers_good,answers_bad))

def OneShotSystematicEval(good_annos,bad_annos,model,processor,text1=OneShotFirstPart,
                          text2=OneShotSecondPart):
    answers_good=[]
    for target in good_annos:
        if 'overlay_axis_1' not in target:
            continue
        print(target)
        Image.open(target).show()
        answers_good.append(OneShot2Steps(img=target,
                        good_example=get_random_file_paths(good_annos,
                                n=1,exclude_path=target)[0],
                        model=model,processor=processor,
                        text1=text1,
                        text2=text2))
        print('Traget:',target,'Answer:',answers_good[-1])

    
    answers_bad=[]
    for target in bad_annos:
        if 'overlay_axis_1' not in target:
            continue
        print(target)
        Image.open(target).show()
        answers_bad.append(OneShot2Steps(img=target,
                        good_example=get_random_file_paths(good_annos,
                                                           n=1,exclude_path=target)[0],
                        model=model,processor=processor,
                        text1=text1,
                        text2=text2))
        print('Traget:',target,'Answer:',answers_bad[-1])

    print('Correct good annotation: ',np.array(answers_good).sum(),'/',len(answers_good))
    print('Correct bad annotation: ',(1-np.array(answers_bad)).sum(),'/',len(answers_bad))

    print('Accuracy: ',calculate_accuracy(answers_good,answers_bad))

def FewShotSystematicEval(good_annos,bad_annos,model,processor,
                          text1=FewShotFirstPart,text2=FewShotSecondPart,n=1):
    answers_good=[]
    for target in good_annos:
        if 'overlay_axis_1' not in target:
            continue
        print(target)
        Image.open(target).show()
        answers_good.append(FewShot2Steps(img=target,
                        good_examples=get_random_file_paths(good_annos,
                                n=n,exclude_path=target),
                        bad_examples=get_random_file_paths(bad_annos,
                                n=n,exclude_path=target),
                        model=model,processor=processor,
                        text1=text1,
                        text2=text2))
        print('Traget:',target,'Answer:',answers_good[-1])

    
    answers_bad=[]
    for target in bad_annos:
        if 'overlay_axis_1' not in target:
            continue
        print(target)
        Image.open(target).show()
        answers_bad.append(FewShot2Steps(img=target,
                        good_examples=get_random_file_paths(good_annos,
                                n=n,exclude_path=target),
                        bad_examples=get_random_file_paths(bad_annos,
                                n=n,exclude_path=target),
                        model=model,processor=processor,
                        text1=text1,
                        text2=text2))
        print('Traget:',target,'Answer:',answers_bad[-1])

    print('Correct good annotation: ',np.array(answers_good).sum(),'/',len(answers_good))
    print('Correct bad annotation: ',(1-np.array(answers_bad)).sum(),'/',len(answers_bad))

    print('Accuracy: ',calculate_accuracy(answers_good,answers_bad))

def LoadLLaVAOneVision7B(bits=4):
    import torch
    from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration
    model_id = "llava-hf/llava-onevision-qwen2-7b-ov-hf"
    if bits==8:
        model = LlavaOnevisionForConditionalGeneration.from_pretrained(
            model_id, 
            torch_dtype=torch.float16, 
            low_cpu_mem_usage=True, 
            load_in_4bit=True,
            device_map='auto',
        )
    elif bits==4:
        model = LlavaOnevisionForConditionalGeneration.from_pretrained(
            model_id, 
            torch_dtype=torch.float16, 
            low_cpu_mem_usage=True, 
            load_in_4bit=True,
            device_map='auto',
        )
    else:
        raise ValueError('Invalid bits value. Use 4 or 8.')
    processor = AutoProcessor.from_pretrained(model_id)
    return model,processor

def LoadLLaVAOneVision72B(bits=8):
    import torch
    from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration
    model_id = "llava-hf/llava-onevision-qwen2-72b-ov-hf"
    if bits==8:
        model = LlavaOnevisionForConditionalGeneration.from_pretrained(
                    model_id, 
                    torch_dtype=torch.float16, 
                    low_cpu_mem_usage=True, 
                    load_in_8bit=True,
                    device_map='auto',
                    #use_flash_attention_2=True
                )
    elif bits==4:
        model = LlavaOnevisionForConditionalGeneration.from_pretrained(
                    model_id, 
                    torch_dtype=torch.float16, 
                    low_cpu_mem_usage=True, 
                    load_in_4bit=True,
                    device_map='auto',
                    #use_flash_attention_2=True
                )
    else:
        raise ValueError('Invalid bits value. Use 4 or 8.')
    processor = AutoProcessor.from_pretrained(model_id)
    return model,processor















CompareInstructionsZeroShotFirst=("The images I am sending are a frontal projections of a CT scans. "
                     "They are not CT slices, instead, they have transparency and let you see throgh "
                     "the entire human body, like an X-ray does. "
                      "The left side of the image represents the right side of the human body. "
                      "The %(organ)s region in the images should "
                     "be marked in red, using an overlay. However, the red overlays may correctly "
                      "or incorrectly mark the %(organ)s. I will send 2 two images, and I want you to "
                      "compare them and tell me which on, Image 1 or Image 2, has a better overlay for the %(organ)s. "
                      " Remember that both overlays may have mistakes, but you must select the one that is closer to the"
                    " expected shape and location of the %(organ)s. Here, I am sending you 'Image 1' only. ")

CompareInstructionsZeroShotSecond=("I am sending you now 'Image 2' for comparison. "
                                   "Compare the overlays in the two images and say if Image 1 "
                                   "or Image 2 has a better overlay for the %(organ)s. "
                                   "Justify your answer very well.")

CompareInstructionsZeroShotUnified=("The images I am sending are a frontal projections of a CT scans. "
                     "They are not CT slices, instead, they have transparency and let you see throgh "
                     "the entire human body, like an X-ray does. "
                      "The left side of the image represents the right side of the human body. "
                      "The image may not present the whole human body. But, if the %(organ)s is present in the images, it should "
                     "be marked in red, using an overlay. However, the red overlays may correctly "
                      "or incorrectly mark the %(organ)s. I am sending here 2 two images, and I want you to "
                      "compare them and tell me which of them, Image 1 or Image 2, has a better overlay for the %(organ)s. "
                      " Remember that both overlays may have mistakes, but you must select the one that is closer to the"
                    " expected shape and location of the %(organ)s. Or, in case the %(organ)s should not be present in the image, "
                    "you select the image with as little red as possible. ")


CompareAdditionalInstructionsLiver=(" When evaluating and comparing the annotaions, "
                                    "consider the following anatomical information: the liver is a large,"
                                    " triangular organ located in the upper right quadrant of the abdomen (left in the image), "
                                    "just below the diaphragm. It is a single structure. It spans across the midline, partially extending"
                                    " into the left upper quadrant. "
                                    "The liver position is primarily under the rib cage. "
                                    "If the CT scan does not include the liver region (e.g., a scan showing just"
                                    "the pelvis), the correct image is the one with no red overlay."
                                    "Therefore, images with no red region may represent the correct overlay.\n")

CompareSummarizeInstructions=("The text below represents a comparisons of 2 images, 'Image 1' and 'Image 2'. "
                              "The images are frontal projections of a CT scan, with transparency, like an X-ray. "
                              "A red overlay marks the %(organ)s region in the images. "
                              "The comparison in the text evaluates which image has a better overlay for the %(organ)s. "
                              "According to the text, which image has the better overlay for the %(organ)s? "
                              "In other words, which image is better?"
                              "Answer in 2 words only, 'Image 1' or 'Image 2'.\n"
                              "The text you should analzye is:\n")

CompareSummarizeInstructionsRepeatImage=("Summarize your last answer, which image has the better overlay for the %(organ)s? "
                                        "In other words, which image is better?"
                                        "Answer in 2 words only, 'Image 1' or 'Image 2'.")

InstructionList=("Answer each of these questions for me, in order: \n"
                "1- Which region of the body does the image represent? \n"
                "2- In which region of the human body is the %(organ)s usually present? \n"
                "3- Is the region you answered for question 2 part of the region you answered for question 1? The abdomen is NOT part of the pelvis.  \n"
                "4- Which image has the least ammount of red overlay (or no red colors at all)? \n"
                "If you answered no to question 3, conclude that the correct image is the one with no red overlay, or with the smallest red area, i.e., your answer to question 4, and do not answer the remaining questions. \n"
                "5- Considering your answer to question 1, in which region of the image it should the %(organ)s be (the right side of the image is the left side of the human body)? \n"
                "6- Consider the region you asnwered in question 5, in which image do you see red in this region? Image 1, Image 2, both images, or none? \n"
                "If you answered 'Image 1' to question 6, conclude that the correct image is Image 1, and do not answer the remaining questions. \n"
                "If you answered 'Image 2' to question 6, conclude that the correct image is Image 2, and do not answer the remaining questions. \n"
                "7- In which image is the red overlay shape more similar to the expected shape of the %(organ)s? \n"
                "If you arrived at this point and did not stop earlier (after question 4 or 6), your final answer is your answer to question 7.")





def Compare2AnnotationsZeroShot(img1,img2,model,processor,
                                text1=CompareInstructionsZeroShotFirst,
                                text2=CompareInstructionsZeroShotSecond,
                                organ='liver',size=None,prt=True):
    text1=text1%{'organ':organ}
    text2=text2%{'organ':organ}

    if isinstance(img1, str):
        img1 = Image.open(img1)
    if size is not None:
        img1=resize_image(img1,size)
    if isinstance(img2, str):
        img2 = Image.open(img2)
    if size is not None:
        img2=resize_image(img2,size)

    if prt:
        print('Image 1:')
        display(img1)
        print('Image 2:')
        display(img2)

    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": text1}
                ],
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", 
                 "text": "Thank you for the first image, please send me the second on for comparison."},
                ],
        },
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": text2}
                ],
        },
    ]
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    print(prompt)
    inputs = processor(text=prompt, images=[img1,img2],
                       return_tensors="pt").to(model.device).to(torch.float16)
    generate_kwargs = {"max_new_tokens": 2000, "do_sample": False}
    output = model.generate(**inputs, **generate_kwargs)
    generated_text = processor.batch_decode(output, skip_special_tokens=True)
    return generated_text

def Compare2AnnotationsZeroShotSinglePrompt(img1,img2,model,processor,
                                text=CompareInstructionsZeroShotUnified,
                                organ='liver',size=None,prt=True):
    text=text%{'organ':organ}

    if isinstance(img1, str):
        img1 = Image.open(img1)
    if size is not None:
        img1=resize_image(img1,size)
    if isinstance(img2, str):
        img2 = Image.open(img2)
    if size is not None:
        img2=resize_image(img2,size)

    if prt:
        print('Image 1:')
        display(img1)
        print('Image 2:')
        display(img2)

    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "image"},
                {"type": "text", "text": text}
                ],
        }
    ]
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = processor(text=prompt, images=[img1,img2],
                       return_tensors="pt").to(model.device).to(torch.float16)
    generate_kwargs = {"max_new_tokens": 400, "do_sample": False}
    output = model.generate(**inputs, **generate_kwargs)
    generated_text = processor.batch_decode(output, skip_special_tokens=True)
    return generated_text


    
def Compare2AnnotationsZeroShot2StepsSinglePrompt(  img1,img2,processor,model,
                                                    text=CompareInstructionsZeroShotUnified,
                                                    textSummary=CompareSummarizeInstructions,
                                                    organ='liver',size=None):
    
    generated_text=Compare2AnnotationsZeroShotSinglePrompt(img1=img1,img2=img2,
                        model=model,processor=processor,
                        text=text,
                        organ=organ,size=size)

    model_text=generated_text[0][generated_text[0].rfind('assistant\n')+len('assistant\n'):]

    print('First answer:',model_text)
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": textSummary+model_text}
                ],
        }
    ]
    
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = processor(text=prompt, 
                       return_tensors="pt").to(model.device).to(torch.float16)
    generate_kwargs = {"max_new_tokens": 5, "do_sample": False}
    output = model.generate(**inputs, **generate_kwargs)
    generated_text = processor.batch_decode(output, skip_special_tokens=True)
    answer=generated_text[0][generated_text[0].rfind('assistant\n')+len('assistant\n'):]
    print('Second answer:',answer)

    if 'image 1' in answer.lower() and 'image 2' not in answer.lower():
        return 1
    elif 'image 2' in answer.lower() and 'image 1' not in answer.lower():
        return 2
    else:
        return 0.5

    
def Compare2AnnotationsZeroShot2Steps(  img1,img2,processor,model,
                                        text1=CompareInstructionsZeroShotFirst,
                                        text2=CompareInstructionsZeroShotSecond,
                                        textSummary=CompareSummarizeInstructions,
                                        organ='liver',size=None):
    
    generated_text=Compare2AnnotationsZeroShot(img1=img1,img2=img2,
                        model=model,processor=processor,
                        text1=text1,
                        text2=text2,
                        organ=organ,size=size)

    model_text=generated_text[0][generated_text[0].rfind('assistant\n')+len('assistant\n'):]

    print('First answer:',model_text)
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": textSummary+model_text}
                ],
        }
    ]
    
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = processor(text=prompt,return_tensors="pt").to(model.device).to(torch.float16)
    generate_kwargs = {"max_new_tokens": 5, "do_sample": False}
    output = model.generate(**inputs, **generate_kwargs)
    generated_text = processor.batch_decode(output, skip_special_tokens=True)
    answer=generated_text[0][generated_text[0].rfind('assistant\n')+len('assistant\n'):]
    print('Second answer:',answer)

    if 'image 1' in answer.lower() and 'image 2' not in answer.lower():
        return 1
    elif 'image 2' in answer.lower() and 'image 1' not in answer.lower():
        return 2
    else:
        return 0.5

def Compare2AnnotationsZeroShot2StepsLarge(  img1,img2,processor,model,
                                        text1=CompareInstructionsZeroShotFirst,
                                        text2=CompareInstructionsZeroShotSecond,
                                        textSummary=CompareSummarizeInstructionsRepeatImage,
                                        organ='liver',size=None):
    
    generated_text=Compare2AnnotationsZeroShot(img1=img1,img2=img2,
                        model=model,processor=processor,
                        text1=text1,
                        text2=text2,
                        organ=organ,size=size)

    text1=text1%{'organ':organ}
    text2=text2%{'organ':organ}
    if isinstance(img1, str):
        img1 = Image.open(img1)
    if size is not None:
        img1=resize_image(img1,size)
    if isinstance(img2, str):
        img2 = Image.open(img2)
    if size is not None:
        img2=resize_image(img2,size)

    model_text=generated_text[0][generated_text[0].rfind('assistant\n')+len('assistant\n'):]

    print('First answer:',model_text)
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": text1}
                ],
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", 
                 "text": "Thank you for the first image, please send me the second on for comparison."},
                ],
        },
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": text2}
                ],
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": model_text}
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": textSummary},
            ],
    },
]
    
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = processor(text=prompt, images=[img1,img2],
                       return_tensors="pt").to(model.device).to(torch.float16)
    generate_kwargs = {"max_new_tokens": 5, "do_sample": False}
    output = model.generate(**inputs, **generate_kwargs)
    generated_text = processor.batch_decode(output, skip_special_tokens=True)
    answer=generated_text[0][generated_text[0].rfind('assistant\n')+len('assistant\n'):]
    print('Second answer:',answer)

    if 'image 1' in answer.lower() and 'image 2' not in answer.lower():
        return 1
    elif 'image 2' in answer.lower() and 'image 1' not in answer.lower():
        return 2
    else:
        return 0.5
    
def Compare2AnnotationsZeroShot2Steps(  img1,img2,processor,model,
                                        text1=CompareInstructionsZeroShotFirst,
                                        text2=CompareInstructionsZeroShotSecond,
                                        textSummary=CompareSummarizeInstructions,
                                        organ='liver',size=None):
    
    generated_text=Compare2AnnotationsZeroShot(img1=img1,img2=img2,
                        model=model,processor=processor,
                        text1=text1,
                        text2=text2,
                        organ=organ,size=size)

    model_text=generated_text[0][generated_text[0].rfind('assistant\n')+len('assistant\n'):]

    print('First answer:',model_text)
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": textSummary+model_text}
                ],
        }
    ]
    
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = processor(text=prompt, 
                       return_tensors="pt").to(model.device).to(torch.float16)
    generate_kwargs = {"max_new_tokens": 5, "do_sample": False}
    output = model.generate(**inputs, **generate_kwargs)
    generated_text = processor.batch_decode(output, skip_special_tokens=True)
    answer=generated_text[0][generated_text[0].rfind('assistant\n')+len('assistant\n'):]
    print('Second answer:',answer)

    if 'image 1' in answer.lower() and 'image 2' not in answer.lower():
        return 1
    elif 'image 2' in answer.lower() and 'image 1' not in answer.lower():
        return 2
    else:
        return 0.5
    
def SystematicComparison(good_annos,bad_annos,model,processor,
                          text1=CompareInstructionsZeroShotFirst,
                          text2=CompareInstructionsZeroShotSecond,
                          organ='liver',size=None):
    answers=[]
    for i,target in enumerate(good_annos,0):
        if 'overlay_axis_1' not in target:
            continue
        print(target)
        good=random.randint(1,2)
        if good==1:
            answer=Compare2AnnotationsZeroShot2Steps(
                            img1=good_annos[i],img2=bad_annos[i],
                            model=model,processor=processor,
                            text1=text1,
                            text2=text2,
                            organ=organ,size=size)
            print('Traget:',target,'Answer:',answer,'Label: Image 1')
            if answer==1:
                answers.append(1)
            elif answer==2:
                answers.append(0)
        else:
            answer=Compare2AnnotationsZeroShot2Steps(
                            img1=bad_annos[i],img2=good_annos[i],
                            model=model,processor=processor,
                            text1=text1,
                            text2=text2,
                            organ=organ,size=size)
            print('Traget:',target,'Answer:',answer,'Label: Image 2')
            if answer==1:
                answers.append(0)
            elif answer==2:
                answers.append(1)
        
            # Clean up
            del answer
            torch.cuda.empty_cache()
            gc.collect()
    acc=np.array(answers).sum()/len(answers)
    acc=np.round(100*acc,1)
    print('Accuracy: ',acc, '(',np.array(answers).sum(),'/',len(answers),')')

def SystematicComparisonSinglePrompt(good_annos,bad_annos,model,processor,
                                                    text=CompareInstructionsZeroShotUnified,
                          organ='liver',size=None):
    answers=[]
    for i,target in enumerate(good_annos,0):
        if 'overlay_axis_1' not in target:
            continue
        print(target)
        good=random.randint(1,2)
        if good==1:
            answer=Compare2AnnotationsZeroShot2StepsSinglePrompt(
                            img1=good_annos[i],img2=bad_annos[i],
                            model=model,processor=processor,
                            text=text,
                            organ=organ,size=size)
            print('Traget:',target,'Answer:',answer,'Label: Image 1')
            if answer==1:
                answers.append(1)
            elif answer==2:
                answers.append(0)
        else:
            answer=Compare2AnnotationsZeroShot2StepsSinglePrompt(
                            img1=bad_annos[i],img2=good_annos[i],
                            model=model,processor=processor,
                            text=text,
                            organ=organ,size=size)
            print('Traget:',target,'Answer:',answer,'Label: Image 2')
            if answer==1:
                answers.append(0)
            elif answer==2:
                answers.append(1)
        
        
        # Clean up
        del answer
        torch.cuda.empty_cache()
        gc.collect()
    acc=np.array(answers).sum()/len(answers)
    acc=np.round(100*acc,1)
    print('Accuracy: ',acc, '(',np.array(answers).sum(),'/',len(answers),')')



def SendMessageQwen(img_file_list, model, processor,  process_vision_info,text, conversation=[],size=None,
                prt=True):
        
    conversation.append({"role": "user","content": [{"type": "text", "text": text}]})
    if len(img_file_list) > 0:
        for i,img in enumerate(img_file_list,0):
            conversation[-1]["content"].append({"type": "image","image":img})
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True,
                                           tokenize=False)
    
    image_inputs, video_inputs = process_vision_info(conversation)
    
    inputs = processor(
        text=[prompt],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    generate_kwargs = {"max_new_tokens": 200, "do_sample": False}
    output = model.generate(**inputs, **generate_kwargs)
    generated_text = processor.batch_decode(output, skip_special_tokens=True)
    answer=generated_text[0][generated_text[0].rfind('assistant\n')+len('assistant\n'):]
    conversation.append({"role": "assistant","content": [{"type": "text", "text": answer}]})

    if prt:
        for i in range(len(img_file_list)):
            display(Image.open(img_file_list[i]))
        print('Text:',text)
        print('Answer:',answer)

    return conversation, answer

def resize_and_encode_image(image_path, size=512):
    # Open the image using PIL
    with Image.open(image_path) as img:
        # Get the original width and height of the image
        original_width, original_height = img.size

        # Determine the scaling factor to adjust the largest side to the desired size
        if original_width > original_height:
            # Width is the largest side
            new_width = size
            new_height = int((original_height / original_width) * size)
        else:
            # Height is the largest side
            new_height = size
            new_width = int((original_width / original_height) * size)

        # Resize the image while maintaining aspect ratio
        img = img.resize((new_width, new_height))

        # Save the resized image to a temporary buffer
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        print('Image resized to:', img.size)

        # Encode the image as base64
        return base64.b64encode(buffered.getvalue()).decode('utf-8')
    
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')


def truncate_base64(data, max_length=30):
    if len(data) > max_length:
        return f"{data[:max_length]}... (truncated, {len(data)} chars)"
    return data

def get_image_size_from_base64(base64_string):
    # Remove the "data:image/png;base64," prefix if present
    base64_data = base64_string.split(",")[1] if "," in base64_string else base64_string
    # Decode the base64 string to bytes
    image_data = base64.b64decode(base64_data)
    # Load image using PIL
    image = Image.open(io.BytesIO(image_data))
    # Get image size (width, height)
    img_size = image.size
    # Get file size in bytes
    img_file_size = len(image_data)
    return img_size, img_file_size

clt=None
mdl=None

def InitializeOpenAIClient(base_url='http://0.0.0.0:23333/v1'):
    from openai import OpenAI
    global clt, mdl
    if clt is not None:
        return clt,mdl
    else:
        # Initialize the client with the API key and base URL
        clt = OpenAI(api_key='YOUR_API_KEY', base_url=base_url)

        # Define the model name and the image path
        mdl = clt.models.list().data[0].id# Update this with the actual path to your PNG image
        print('Initialized model and client.')
        return clt,mdl

def CreateConversation(img_file_list, text, conver,size=None,prt=False):
    #if no previous conversation, send conver=[]. Do not automatically define conver above.
    
    conver.append({
            'role': 'user',
            'content': [{
                'type': 'text',
                'text': text,
            }],
        })
    
    imgs=[]
    for i,img in enumerate(img_file_list,0):
        if isinstance(size, list):
            s=size[i]
        else:
            s=size
        if s!=None:
            img = resize_and_encode_image(img, s)
        else:
            img = encode_image(img)
        conver[-1]['content'].append({
                                            "type": "image_url",
                                            "image_url": {
                                                "url": f"data:image/png;base64,{img}"
                                            },
                                        })
        if prt:
            image_size, file_size = get_image_size_from_base64(f"data:image/png;base64,{img}")
            print(f"Image Size (WxH) in prompt: {image_size}, File Size: {file_size} bytes")
    return conver

def SendMessageLmdeploy(img_file_list, text, conver, base_url='http://0.0.0.0:23333/v1',  
                        size=None,prt=True,print_conversation=False,max_tokens=None):
    """
    Sends a message to the LM deploy API.

    Args:
        img_file_list (list): A list of image file paths.
        text (str): The text message to send.
        conver (list): A list of conversation objects.
        base_url (str, optional): The base URL of the LM deploy API. Defaults to 'http://0.0.0.0:23333/v1'.
        size (int, optional): The size to resize the images to. Defaults to None.
        prt (bool, optional): Whether to print the images and conversation. Defaults to True.
        print_conversation (bool, optional): Whether to print the conversation. Defaults to False.
        max_tokens (int, optional): The maximum number of tokens in the completion response. Defaults to None.

    Returns:
        tuple: A tuple containing the updated conversation and the answer from the LM deploy API.
    """
    #if no previous conversation, send conver=[]. Do not automatically define conver above.
    client,model_name=InitializeOpenAIClient(base_url)

    conver=CreateConversation(img_file_list=img_file_list, text=text, conver=conver,size=size)

    # Print the conversation with truncated base64 data
    if print_conversation:
        for entry in conver:
            print(f"Role: {entry['role']}")
            for content in entry['content']:
                if content['type'] == 'text':
                    print(f"Text: {content['text']}")
                elif content['type'] == 'image_url':
                    image_url = content['image_url']['url']
                    truncated_url = truncate_base64(image_url)
                    # Extract image size from the base64 string
            
    # Create the request with the base64-encoded image data
    if max_tokens is None:
        response = client.chat.completions.create(
            model=model_name,
            messages=conver,
            temperature=0,
            top_p=1)
    else:
        response = client.chat.completions.create(
            model=model_name,
            messages=conver,
            max_tokens=max_tokens,
            temperature=0,
            top_p=1)


    # Print the response
    answer = response.choices[0].message.content

    conver.append({"role": "assistant","content": [{"type": "text", "text": answer}]})

    if prt:
        for i in range(len(img_file_list)):
            display(Image.open(img_file_list[i]))
        print('Text:',text)
        print('Answer:',answer)

    return conver, answer

BodyRegionTextV0=("The image I am sending is frontal projections of one CT scan. It is not a CT slice, instead, they have transparency and let you see through the entire human body, like an X-ray does. It looks like an AP (anterior-to-posterior) X-ray. Answer the questions below:\n"
"Q1- Look at the image carefully, tell me which body region it represents and where the image limits are. Present a complete list of all organs usually present in this body region (just list their names).\n"
"Q2- Based on your answer to Q1, is the %(organ)s usually present in this region and in your list? Answer ‘yes’ or ‘no’ using the template below, substituting  _ by Yes or No.:\n"
"Q2 = _\n")

BodyRegionTextV1=("The image I am sending is frontal projections of one CT scan. It is not a CT slice, instead, they have transparency and let you see through the entire human body, like an X-ray does. It looks like an AP (anterior-to-posterior) X-ray. Answer the questions below:\n"
"Q1- Look at the image carefully, tell me which body region it represents and which organs inside this region are touching the image's upper boundary and its lower boundary. Present a complete list of all organs usually present in the body region contained in the image (just list their names).\n"
"Q2- Based on your answer to Q1, is the %(organ)s usually present in this region and in your list? Answer ‘yes’ or ‘no’ using the template below, substituting  _ by Yes or No.:\n"
"Q2 = _\n")
#seem to give maybe more accurate answers
BodyRegionText=("The image I am sending is frontal projections of one CT scan. It is not a CT slice, instead, they have transparency and let you see through the entire human body, like an X-ray does. It looks like an AP (anterior-to-posterior) X-ray. Answer the questions below:\n"
"Q1- Look at the image carefully, tell me which body region it represents and which organs and bones inside this region are crossing the image's upper boundary and its lower boundary. Present a complete list of all organs usually present in the body region contained in the image (just list their names).\n"
"Q2- Based on your answer to Q1, is the %(organ)s usually present in this region and in your list? Answer ‘yes’ or ‘no’ using the template below, substituting  _ by Yes or No.:\n"
"Q2 = _\n")

#touching became crossing

BodyRegionTextHighlightedSkeleton=("The image I am sending is frontal projections of one CT scan. "
                "Is it not a CT slices but a transparent view of the entire body, similar to an X-ray that allows a see-through visualization of internal structures. It looks like an AP (anterior-to-posterior) X-ray."
                "The image focuses on the bones. \n"
"Q1- Look at the image carefully, tell me very precisely where in the body the image lower and upper limits are. Present a complete list of all organs usually present in the body area within the image (just list their names).\n"
"Q2- Which of these organs are partially contained in the image? In other words, which organs touch the image uper or lower limits.\n"
"Q3- Based on your answer to Q1, is the %(organ)s usually present (even if partially) in this area and in your list? Answer ‘yes’ or ‘no’ using the template below, substituting  _ by Yes or No.:\n"
"Q3 = _\n")

BodyRegionTextSkeleton=("The image I am sending is frontal projections of one CT scan, focusing on showing the skeleton. Look at it carefully, and answer the questions below:\n"
"Q1- Which bones are on the top of the image? Bones are on its bottom?\n"
"Q2- Which of the following landmarks are present in the image? Answer ‘yes’ or ‘no’ using the template below, substituting  _ by Yes or No:\n"
"skull = _"
"neck = _"
"trachea = _"
"ribs = _"
"lumbar spine = _"
"pelvis = _"
"femurs = _"
"Q3- Considering these landmarks and the bones on the image top and bottom, give me a complete list of all organs usually contained within this image limits (just list their names).\n"
"Q4- Based on your answer to Q3, is the %(organ)s usually present within this image limits? Answer ‘yes’ or ‘no’ using the template below, substituting  _ by Yes or No:\n"
"Q4 = _\n")

BodyRegionTextSkeletonV0=("The image I am sending is frontal projections of one CT scan. It is not a CT slice, instead, they have transparency and let you see through the entire human body, like an X-ray does. It highlights major bones, not soft tissue. Answer the questions below:\n"
"Q1- Look at the image carefully, tell me which body region it represents and where the image limits are. Present a complete list of all organs usually present in this body region (just list their names).\n"
"Q2- Based on your answer to Q1, is the %(organ)s usually present in this region and in your list? Answer ‘yes’ or ‘no’ using the template below, substituting  _ by Yes or No:\n"
"Q2 = _\n")


BodyRegionTextSkeletonV1 = ("I am sending you a figure with 2 images inside it, Image 1 (left) and Image 2 (right). The images are two frontal projections from the same CT scan. "
"They are not CT slices but a transparent view of the entire body, similar to an X-ray that allows a see-through visualization of internal structures. They look like AP (anterior-to-posterior) X-rays."
"Image 1 uses a window that displays soft tissue, while Image 2 focuses on the skeleton, but they show exactly the same body region. "
"Please carefully analyze the image and follow the steps below to ensure accuracy:\n\n"
"Step 1: Describe exactly which parts of the body are visible in the image, including where the image starts and ends. Focus on the visible boundaries, and if the image is cropped, mention where the cropping occurs. "
"Do not include parts of the body that are not visible in the image.\n\n"
"Step 2: Based on the visible region described in Step 1, provide a list of organs that are actually visible within the image boundaries. "
"Only include organs that are fully or partially visible in the image, and avoid general assumptions about what should be present.\n\n"
"Step 3: Based on your findings from Step 1 and Step 2, is the %(organ)s visible in the image? Answer ‘yes’ or ‘no’ using the template below, substituting  _ by 'Yes' or 'No':\n"
"Step 3 = _\n")

BodyRegionTextBad=("The image I am sending is a frontal projection of one CT scan. It is not a CT slice but a projection similar to an X-ray, allowing you to see through the body. Please answer the following questions carefully:\n"
                "Q1- Identify the body region shown in the image. Specifically, indicate if it includes the **upper chest**, **mid chest**, **lower chest**, **abdominal regions** and **pelvic regions**. List all organs that are typically found in the body region(s) you identify, including whether the **trachea** is normally visible in this region.\n"
                "Q2- Based on your answer to Q1, is the %(organ)s normally present in this body region? Provide a 'yes' or 'no' answer using the template below, substituting _ with Yes or No.:\n"
                "Q2 = _\n")


AorticArchTextSkeletonV0=("If you answered no to Step 3, skip Step 4. If you answered yes, continue:\n"
                "Step 4: Are heart and aoric arch fully contained within image bondaries? Answer ‘yes’ or ‘no’ using the template below, substituting  _ by 'yes' or 'no':\n"
                "Step 4 = _\n")

AorticArchTextSkeleton=("If you answered no to Q3, skip Q4 and Q5. If you answered yes, continue:\n"
                "Q4- What parts of the respiratory system are contained in the image region?\n"
                "Q5- Based on your answer to Q3, is the trachea usually present in this region and in your list? Answer ‘yes’ or ‘no’ using the template below, substituting  _ by Yes or No:\n"
                "Q5 = _\n")

AorticArchTextV5=("If you answered no to Q2, skip Q3 and Q4. If you answered yes, continue:\n"
                "Q3- What parts of the respiratory system are usually presentusually present in this region and in your list?\n"
                "Q4- Based on your answer to Q3, is the trachea usually present in this region and in your list? Answer ‘yes’ or ‘no’ using the template below, substituting  _ by Yes or No:\n"
                "Q4 = _\n")
#80% accuracy but there is a better one below!

AorticArchTextV4=("If you answered no to Q2, skip Q3 and Q4. If you answered yes, continue:\n"
                "Q3- The **trachea** is part of the respiratory system, and it is located **only in the upper chest**, near the midline of the body. It appears as a **vertical air-filled tube**, running from the neck down to the top of the lungs and connecting to the bronchi. Importantly, if **upper chest** organs and the **top of the lungs** are not visible in the image, the trachea cannot be present.\n"
                "Look at the image and determine if the **upper chest** is visible, specifically focusing on the area above the lungs. If only the lower parts of the lungs are visible, the trachea should not be visible. Identify which parts of the respiratory system are visible.\n"
                "Q4- Based on your answer to Q3, is the **trachea** visible in this image? Answer 'yes' or 'no' using the template below, substituting _ with Yes or No. Be sure to consider that the trachea cannot be present without the **upper chest** being visible, and it should appear as a distinct air-filled tube above the lungs:\n"
                "Q4 = _\n")

AorticArchTextV3=("If you answered no to Q2, skip Q3 and Q4. If you answered yes, continue:\n"
                "Q3- Look closely at the image. The **trachea** is part of the respiratory system and should appear as a **vertical air-filled tube in the upper chest**, located in the midline, above the lungs. It is positioned just below the neck and above the heart. If only the lower parts of the lungs are visible, the trachea is unlikely to be present. Based on this information, identify which parts of the respiratory system (if any) are visible in this image.\n"
                "Q4- Based on your answer to Q3, is the **trachea** visible in this image? Answer 'yes' or 'no' using the template below, substituting _ with Yes or No. Be sure to consider whether the upper chest is visible and whether the trachea’s position is above the lungs in this image:\n"
                "Q4 = _\n")
#bad

AorticArchTextV2=("Q3- If you answered no to Q2, skip this question (Q3). If you answered yes, continue:\n"
                "In this image, you should identify whether the **upper part of the lungs** is visible, not just the base. The **upper lungs** are located near the top of the ribcage.\n"
                "Pay attention to this distinction: If **only the base of the lungs** is visible, the aortic arch is not in the image.\n"
                "Remember, the **aortic arch** is located in the **upper chest**, near the top of the lungs. Therefore, the presence of the aortic arch depends on whether the upper portion of the lungs is visible in the scan.\n"
                "Now answer this: Are the **upper parts of the lungs** fully visible (not just the base)? If yes, the aortic arch should be visible.\n"
                "Based on your observation, fill in the template below by substituting _ with Yes or No:\n"
                "Q3 = _\n"
                "Please explain briefly whether the upper lungs are visible and why this leads to the conclusion about the aortic arch's presence.")

AorticArchTextV1=("Q3- If you answered no to Q2, ignore this question (Q3). If you answered yes, answer whis new question: the lungs should be two large dark regions in the image. Are the lungs present and fully visible, from top to bottom, in the image? Think carefully, and provide a complete answer with justification. \n"
                "Based on your answer to Q3, fill the template below substituting  _ by Yes or No.:\n"
                "Q3 = _\n")

AorticArchTextV0=("Q3- If you answered no to Q2, ignore this question (Q3). If you answered yes, describe the image's upper limit: does in include the aortic arch? The aortic arch is placed just above the heart, and it will be present if the top of the lungs are visible in the image. Think carefully, and provide a complete answer with justification.\n"
                "Based on your answer to Q3, fill the template below substituting  _ by Yes or No.:\n"
                "Q3 = _\n")

AorticArchTextV6=("Q3- If you answered no to Q2, ignore this question (Q3). If you answered yes, describe the image's upper limit: does it include the aortic arch? The aortic arch is placed just above the heart, and it will be presnt if the top of the lungs are visible in the image. Think carefully, and provide a complete answer with justification.\n"
                "Based on your answer to Q3, fill the template below substituting  _ by Yes or No.:\n"
                "Q3 = _\n")
#80% accuracy

AorticArchTextV7=("Q3- If you answered no to Q2, ignore this question (Q3). If you answered yes, describe the image's upper limit: does it include the aortic arch? The aortic arch is placed just above the heart.\n"
                "Carefully analyze the image and your answer to Q1 to respond Q3, justifying your answer. Moreover, consider the following tips:\n"
                "- The aortic arch is presnt only if the top of the lungs and the trachea are present in the image.\n" 
                "Based on your answer to Q3, fill the template below substituting  _ by Yes or No.:\n"
                "Q3 = _\n")

AorticArchText=(
    "If you answered 'no' to Q2, skip Q3, Q4 and Q5. If you answered 'yes', continue.\n"
    "Q3: Based on your answer to Q1, are the lungs present in the image? \n"
    "Answer 'yes' or 'no' using the template below, substituting _ with Yes or No.\n"
    "Q3 = _\n"
    "If you answered 'no' to Q3, skip Q4 and Q5. If you answered 'yes', continue.\n"
    "Q4: Repeat from you answer to Q1, which organs and bones are crossing the image upper boundary (none, brain, neck, lungs, heart or others)? \n"
    "Q5: Considering your answer to Q3 and Q4, does the image include the aortic arch? The aortic arch is placed just above the heart. It is not present if you listed the heart, lower ribs, or lower lungs in Q4. It is present if you listed none, brain or neck. \n"#It is probably not present if you listed the lungs. \n"
    "Answer 'yes' or 'no' using the template below, substituting _ with Yes or No.\n"
    "Q5 = _\n"
)
#90% localization accuracy

ComparisonText=("I am now sending you a figure with 4 images inside of it. They are all frontal projections of the same CT scan I sent before. "
                "The %(organ)s region in the images should be marked in red, using a red overlay. However, the red overlays may correctly or incorrectly mark the %(organ)s. "
                "The letters R (blue) and L (green) inside the images represent the right and left sides of the human body. Here is the description of each image (image titles are above the images):\n"
"Image 1: frontal projection of the CT scan using a wide window (2000 UI), tuned for better bone visualization. The image is superposed with %(organ)s overlay 1 (in red).\n"
"Image 2: frontal projection of a CT scan using a less wide window (400 UI), tuned for better abdominal organ visualization. The image is also superposed with %(organ)s overlay 1 (like image 1).\n"
"Image 3: frontal projection of the CT scan using a wide window (2000 UI). The image is superposed with %(organ)s overlay 2 (in red).\n"
"Image 4: frontal projection of a CT scan using a less wide window (400 UI). The image is also superposed with %(organ)s overlay 2 (like image 3).\n"
"Compare overlay 1 (shown in images 1 and 2) to overlay 2 (shown in images 3 and 4) and tell me which one is a better overlay for the %(organ)s.\n")

ComparisonTextContinued=("I am now sending you a figure with 4 images inside of it. They are all frontal projections of the same CT scan I sent before. "
                "The %(organ)s region in the images should be marked in red, using a red overlay. However, the red overlays may correctly or incorrectly mark the %(organ)s. "
                "The letters R (blue) and L (green) inside the images represent the right and left sides of the human body. Here is the description of each image (image titles are above the images):\n"
"Image 1: frontal projection of the CT scan using a wide window (2000 UI), tuned for better bone visualization. The image is superposed with %(organ)s overlay 1 (in red).\n"
"Image 2: frontal projection of a CT scan using a less wide window (400 UI), tuned for better abdominal organ visualization. The image is also superposed with %(organ)s overlay 1 (like image 1).\n"
"Image 3: frontal projection of the CT scan using a wide window (2000 UI). The image is superposed with %(organ)s overlay 2 (in red).\n"
"Image 4: frontal projection of a CT scan using a less wide window (400 UI). The image is also superposed with %(organ)s overlay 2 (like image 3).\n"
"Compare overlay 1 (shown in images 1 and 2) to overlay 2 (shown in images 3 and 4) and tell me which one is a better overlay for the %(organ)s.\n")

ComparisonText2Classes=("I am sending you a figure with 4 images inside of it. They are all frontal projections of the same CT scan. It is not a CT slice, instead, they have transparency and let you see through the entire human body, like an X-ray. "
                "The %(organ)s in the images should be marked using overlays. "
                "Each overlay should mark the right %(organ_singular)s in blue, and the left %(organ_singular)s in green. However, the two overlays may correctly or incorrectly mark the %(organ)s. "
                "One possible type of overlay error is overlapping the left and right %(organ_singular)s. These undesirable overlaps (if present) are marked in red in each overlay. "
                "The letters R (blue) and L (green) inside the images represent the right and left sides of the human body. Here is the description of each image (image titles are above the images):\n"
"Image 1 (top right): frontal projection of the CT scan using a wide window (2000 UI), tuned for better bone visualization. The image is superposed with %(organ)s overlay 1 \n"
"Image 2 (top left): frontal projection of a CT scan using a less wide window (400 UI), tuned for better abdominal organ visualization. The image is also superposed with %(organ)s overlay 1 (like image 1).\n"
"Image 3 (bottom left): frontal projection of the CT scan using a wide window (2000 UI). The image is superposed with %(organ)s overlay 2 (in red).\n"
"Image 4 (bottom right): frontal projection of a CT scan using a less wide window (400 UI). The image is also superposed with %(organ)s overlay 2 (like image 3).\n"
"Compare overlay 1 (images 1 and 2, top) to overlay 2 (images 3 and 4, bottom) and tell me which one is a better overlay for the %(organ)s.\n")


ComparisonText6Figs=("I am now sending you a figure with 6 images inside of it. They are all frontal projections of the same CT scan I sent before. "
                "The %(organ)s region in the images should be marked with colored overlays. However, the overlays may correctly or incorrectly mark the %(organ)s. "
                "Images 1 and 4 (column 1, left), show overlay 1 in red. Images 2 and 5 (column 2, middle), show overlay 2 in yellow. Images 3 and 6 (column 3, right), show the superposition of overlay 1 (red) and 2 (yellow), their intersection is in organge. "
                "The only difference from the images in the first row (1, 2 and 3) and in the second row (images 4, 5 and 6) is contrast, the first row more clearly shows the bones."
                "The letter R in blue inside the images represent the right side of the human body. The letter L in blue represents the left side of the human body. \n"
                "Compare overlay 1 (red) to overlay 2 (yellow) and tell me which one is a better overlay for the %(organ)s.\n")

ComparisonText2Figs=("I am sending you a figure that contains two images, Image 1 on the left and Image 2 on the right. "
                     "These images are frontal projections from the same CT scan, displayed with transparency similar to X-rays. "
                     "In these images, a red overlay has been applied to highlight the %(organ)s region. However, the accuracy of the red overlays in marking the %(organ)s may vary between the two images. "
                     "The letters R (blue) and L (green) are visual markers indicating the right and left sides of the human body, respectively. "
                     "Note that the left side of each image corresponds to the right side of the body, and the right side of the image corresponds to the left side of the body, like AP (anterior-to-posterior) X-rays. "
                     "Please compare Overlay 1 (Image 1) to Overlay 2 (Image 2) and determine which overlay provides a more accurate representation of the %(organ)s. "
                     "You will need to assess the overlays based on the following specific criteria related to the %(organ)s in the next part.")#GPT revised

ComparisonText2FigsV0=("I am now sending you a figure with 2 images inside of it. They are frontal projections of the same CT scan. "
                     "They are not a CT slices, instead, they have transparency and let you see through the entire human body, like an X-ray. "
                "The %(organ)s region in the images should be marked in red, using a red overlay. However, the red overlays may correctly or incorrectly mark the %(organ)s. "
                "The letters R (in blue) and L (in green) inside the images represent the right and left sides of the human body. "
                "The left side of each image represents the right side of the human body. The right side of each image represents the left side of the human body. "
                "Compare overlay 1 (shown in Image 1) to overlay 2 (shown in Images 2) and tell me which one is a better overlay for the %(organ)s.\n")


ComparisonText2FigsContinued=("I am sending you a figure with 2 images inside of it. The one on the left, Image 1, is the first image I sent you in this conversation, and the one on the right, Image 2, is the second one. " 
                            " As I previously explained, they are frontal projections of the same CT scan. "
                     "They are not a CT slices, instead, they have transparency and let you see through the entire human body, like an X-ray. "
                "The %(organ)s region in the images should be marked in red, using a red overlay. However, the red overlays may correctly or incorrectly mark the %(organ)s. "
                "The letters R (in blue) and L (in green) inside the images represent the right and left sides of the human body. "
                "The left side of each image represents the right side of the human body. The right side of each image represents the left side of the human body. "
                "Compare overlay 1, shown in Image 1, to overlay 2, shown in Images 2, and tell me which one is a better overlay for the %(organ)s.\n")


ComparisonText2Classes2Figs=("I am sending you a figure with 2 images inside of it. They are frontal projections of the same CT scan. "
                             " They are not a CT slices, they have transparency and let you see through the entire human body, like an X-ray. "
                             "The letters R (in blue) and L (in green) inside the images represent the right and left sides of the human body, which is NOT the right and left sides of the images. "
                "The %(organ)s in the images should be marked using overlays. "
                "Each overlay should mark the right %(organ_singular)s in blue, and the left %(organ_singular)s in green. However, the two overlays may correctly or incorrectly mark the %(organ)s. "
                "One possible error is the absence of one of the %(organ)s, in this case one of the collors will be missing. "
                "Another one is marking the left %(organ_singular)s in the right side of the human body or vice-versa. "
                "In these cases, it is possible to observe an undesired overlap between the left and right %(organ_singular)s, any overlap is marked in red. If one %(organ_singular)s is wrongly marked totally inside the other, we will have only red and blue or red and green overlays."
                "You may also encounter errors in the %(organ)s shape and position."
"Your task: Compare overlay 1 in image 1 to overlay 2 in image 2 and tell me which one is a better overlay for the %(organ)s.\n")

ComparisonText2Classes2FigsV2=("I am sending you a figure with 2 images inside of it. They are frontal projections of the same CT scan. "
                             " They are not a CT slices, they have transparency and let you see through the entire human body, like an X-ray. "
                             "The letters R (in blue) and L (in green) inside the images represent the right and left sides of the human body, which is NOT the right and left sides of the images. "
                "The %(organ)s in the images should be marked using overlays. "
                "Each overlay should mark the right %(organ_singular)s in blue, and the left %(organ_singular)s in green. However, the two overlays may correctly or incorrectly mark the %(organ)s. "
                "One possible error is the absence of one of the %(organ)s. Another one is marking the left %(organ_singular)s in the right side of the human body or vice-versa. "
                "In these cases, it is possible to observe an undesired overlap between the left and right %(organ_singular)s, any overlap is marked in red. You may also encounter errors in the %(organ)s shape and position."
"Your task: Compare overlay 1 in image 1 to overlay 2 in image 2 and tell me which one is a better overlay for the %(organ)s.\n")

ComparisonText2Classes2FigsV1=("I am sending you a figure with 2 images inside of it. They are frontal projections of the same CT scan. They are not a CT slices, they have transparency and let you see through the entire human body, like an X-ray. "
                "The %(organ)s in the images should be marked using overlays. "
                "Each overlay should mark the right %(organ_singular)s in blue, and the left %(organ_singular)s in green. However, the two overlays may correctly or incorrectly mark the %(organ)s. "
                "One possible type of overlay error is overlapping the left and right %(organ_singular)s. These undesirable overlaps (if present) are marked in red in each overlay. "
                "A possible label error is when the left %(organ_singular)s is marked in the right side of the body or vice versa. In these cases, there may be overlaps be"
                "The letters R (in blue) and L (in green) inside the images represent the right and left sides of the human body (which is not the right and left sides of the images). "
"Your task: Compare overlay 1 in image 1 to overlay 2 in image 2 and tell me which one is a better overlay for the %(organ)s.\n")


ComparisonText1Fig=("I am now sending you the same frontal projection of a CT as before, but now with 2 overlays. Overlay 1 is red, overlay 2 is yellow, and their superposition is orange."
                "The overlays try to mask the %(organ)s region in the image. However, the red or the yellow overlay may mark it better %(organ)s. "
                "The letters R (blue) and L (green) inside the images represent the right and left sides of the human body. Although the images represent the same CT scan, the overlays are different. \n"
                "Compare overlay 1 (red) to overlay 2 (yellow) and tell me which one is a better overlay for the %(organ)s.\n"
                "Remember to justify your answer, and remember that the orange region is the overlap between the 2 overlays.")

ComparisonText6FigsSimple=("I sending you a figure with 6 images inside of it. They are all frontal projections of the same CT scan. They look like X-rays."
                "The %(organ)s region in the images should be marked with colored overlays. However, the overlays may correctly or incorrectly mark the %(organ)s. "
                "Images 1 and 4 (column 1, left), show overlay 1 in red. Images 2 and 5 (column 2, middle), show overlay 2 in yellow. Images 3 and 6 (column 3, right), show the superposition of overlay 1 (red) and 2 (yellow), their intersection is in organge. "
                "The only difference from the images in the first row (1, 2 and 3) and in the second row (images 4, 5 and 6) is contrast, the first row more clearly shows the bones."
                "The letters R (blue) and L (green) inside the images represent the right and left sides of the human body. \n"
                "Compare overlay 1 (red) to overlay 2 (yellow) and tell me which one is a better overlay for the %(organ)s.\n")

LiverDescription=("When evaluating and comparing the overlays, consider the following anatomical information:\n"
"a) The liver is a large organ, with triangular or wedge-like shape.\n"
"b) The liver is located in the upper right quadrant of the abdomen (right is indicated with a blue R in the figures), just below the diaphragm. It spans across the midline, partially extending into the left upper quadrant of the abdomen. It spans across the midline, partially extending into the left upper quadrant. Looking at the bones images 1 and 3, you can easily identify if the liver should appear in the images or not. The liver is not in the pelvis.\n"
"c) The liver is a single structure.\n"
"d) The liver position is primarily under the rib cage.\n"
"e) If the CT scan does not include the %(organ)s region (for example, a scan showing just the pelvis), the correct overlay is actually the one showing no red region (or as little as possible).\n")

NoOrganText=("I am now sending you a figure with 4 images inside of it. They are all frontal projections of the same CT scan I sent before. "
                "The %(organ)s region in the images should be marked in red, using a red overlay. However, remember you concluded that the %(organ)s should not be present in the images. "
                "Compare overlay 1 (shown in images 1 and 2) to overlay 2 (shown in images 3 and 4) and tell me which one has the smallest red area (ideally none).\n")

NoOrganSimple=("I am sending you 2 images, 'Image 1' on the left, and 'Image 2' on the right. Which of them has the least ammount of red color?\n"
             "Answer only 'Image 1' or 'Image 2'.\n")

NoOrganSimple2Classes=("I am sending you 2 images, 'Image 1' on the left, and 'Image 2' on the right. Which of them has the least ammount of red, blue and green colors?\n"
             "Answer only 'Image 1' or 'Image 2'.\n")

CompareSummarize=("The text below represents a comparisons of 2 overlays, 'Overlay 1' and 'Overlay 2'. "
                "The overlays were positioned over 4 images, Image 1 and Image 2 showed Overlay 1, and Image 3 and Image 4 showed Overlay 2. "
                "A LVLM like you compared the 2 overlays by analyzing the 4 images. Its answer is the text below."
                "The text explains which overlay is better. I want you to answer me which overaly is better according to the text. Answer me with only 2 words: 'Overlay 1' or 'Overlay 2'. "
                "If the text does not mention any overlay or if it is blank, answer 'none'. The text is:\n")

CompareSummarize2Figs=("The text below represents a comparisons of 2 overlays, 'Overlay 1' and 'Overlay 2'. "
                " Image 1 showed Overlay 1, and Image 2 showed Overlay 2. "
                "A VLM like you compared the 2 overlays by analyzing the 2 images. Its answer is the text below."
                "The text explains which overlay (or image) is better. I want you to answer me which overaly is better according to the text. Answer me with only 2 words: 'Overlay 1' or 'Overlay 2'. "
                "The text is:\n")

CompareSummarize6Figs=("The text below represents a comparisons of 2 overlays, 'Overlay 1' and 'Overlay 2'. "
                " Image 1 and 4 showed Overlay 1 in red, and Images 2 and 5 showed Overlay 2 in yellow. Images 3 and 6 showed the superposition of both overlays. "
                "A LVLM like you compared the 2 overlays by analyzing the 6 images. Its answer is the text below."
                "The text explains which overlay (or image) is better. I want you to answer me which overaly is better according to the text. Answer me with only 2 words: 'Overlay 1' or 'Overlay 2'. "
                "In case the text does not mention any overlay or if it is blank, answer 'none'. The text is:\n")

CompareSummarize1Fig=("The text below represents a comparisons of 2 overlays, 'Overlay 1' and 'Overlay 2'. "
                " They were placed over one image, overlay 1 was red, overlay 2 was yellow, and their overlap was orange. "
                "A LVLM like you compared the 2 overlays. Its answer is the text below."
                "The text explains which overlay is better. I want you to answer me which overaly is better according to the text. Answer me with only 2 words: 'Overlay 1' or 'Overlay 2'. "
                "If the text does not mention any overlay or if it is blank, answer 'none'. The text is:\n")


KidneysDescription=("Answer each one of the following questions before drawing a conclusion of which overlay is better:\n"
                   "a) Consider that a person usually has two kidneys. Does Image 1 show two distinct kidney overlays? Does Image 2 show two distinct kidney overlays? If both show a single kidney, the patient may truly have only one kidney. If only one images shows a single kidney, the image showing two kidneys should be better. \n"
                   "b) Shape: each kidney has a bean-shaped structure, with a convex lateral surface and a concave medial surface. Are the shapes of the 2 kidneys in image 1 correct? Are the shapes in image 2 better, worse or similar? \n"
                   "c) Location: the kidneys are located in the retroperitoneal space, on either side of the spine, at the level of the lower ribs. Are the locations of the 2 kidneys in image 1 correct? Are the kidney locations in image 2 better, worse or similar? \n")

KidneysDescriptionV0=("When evaluating and comparing the overlays, consider the following anatomical information:\n"
                   "a) Number and location: a person usually has two kidneys.\n"
                   "b) Shape: each kidney has a bean-shaped structure, with a convex lateral surface and a concave medial surface. The concave side features an indentation called the renal hilum.\n"
                   "c) Location: the kidneys are located in the retroperitoneal space, on either side of the spine, at the level of the lower ribs.\n")

AdrenalGlandDescription=("When evaluating and comparing the overlays, consider the following anatomical information:\n"
                   "a) Number: a person usually has two adrenal glands, one on top of each kidney.\n"
                   "b) Location: the adrenal glands are located on the superior aspect of each kidney, in the retroperitoneal space.\n"
                   "c) Right Adrenal Gland Shape (right side of the body): triangular shape.\n"
                   "d) Left Adrenal Gland Shape (left side of the body): Generally crescent-shaped or semilunar. May appear as a curved line or elongated structure above the kidney.\n"
                   "e) Size: adrenal glands are relatively small compared to the kidneys.\n")

AortaDescriptionV0=("Answer each one of the following questions before drawing a conclusion of which overlay is better:\n"
                     "a) Shape: In a CT projection, the aorta should be marked as a long vertical red line, with a curve on its top, similar to the shape of a question mark (?). Does the shape of the red region in image 1 match the shape of the aorta? Does the shape of the red region in image 2 match the shape of the aorta? If the shape of the aorta is better represented in one of the images, that image should be better. \n"
                     "b) Location: after the small curve on its top (aortic arch), the aorta should be parallel to the spine. Usually it is a long vertical line in the midline of the body. The aorta can be curved if the spine is also curved. Is the aorta correctly positioned in image 1? Is it correctly positioned in image 2? \n"
                     "c) Completeness: the aorta should be visible from the heart to the pelvis. Is the aorta complete in image 1? Is it complete in image 2? \n")
#Acc: 2/3
AortaDescription = (
    "Analyze the two images to determine which overlay represents the aorta more accurately, based on the following criteria:\n"
    "1. Shape: The aorta should appear as a long vertical red line with a curve at the top, resembling a question mark (?).\n"
    "- In Image 1, does the shape of the red region resemble this description?\n"
    "- In Image 2, does the shape of the red region resemble this description?\n"
    "- Which image has a shape closer to the expected form of the aorta?\n"
    "2. Location: After the small curve at the top (aortic arch), the aorta should run parallel to the spine, which usually appears as a vertical line along the midline of the body.\n"
    "- Is the aorta correctly positioned in Image 1 relative to the spine?\n"
    "- Is the aorta correctly positioned in Image 2 relative to the spine?\n"
    "- Which image shows a better location for the aorta?\n"
    "3. Completeness: The aorta should be visible from the heart down to the pelvis (or down to the image bottom, if the pelvis is not visible).\n"
    "- Is the aorta fully visible and complete in Image 1?\n"
    "- Is the aorta fully visible and complete in Image 2?\n"
    "- Which image shows a more complete representation of the aorta?\n"
    "Based on the shape, location, and completeness, indicate which overlay (Image 1 or Image 2) provides a better representation of the aorta."
)
#Acc: 3/3
DescendingAortaDescriptionV0=("Answer each one of the following questions before drawing a conclusion of which overlay is better:\n"
                     "a) Shape: The aortic arch is not visible in this CT projection, Thus, the aorta should be a long vertical red line (tubular structure), clearly visible at the very top of the image. A common overlay error is not displaying the aorta at the very top of the image. Is the red overlay in Image 2 visible at the very top of the image? Is the red overlay in Image 2 visible at the very top of the image? \n"
                     "b) Location: the aorta should be parallel to the spine. Usually it is a long vertical line in the midline of the body. The aorta can be curved if the spine is also curved. Is the aorta correctly positioned in image 1? Is it correctly positioned in image 2? \n"
                     "c) Completeness: the aorta should be visible from the top of the image to the pelvis (if the pelvis is present in the ct). Is the aorta complete in image 1? Is it complete in image 2? \n")
#Accuracy:  62.5 ( 5 / 8 )

DescendingAortaDescriptionV1=("Answer the following questions before concluding which overlay is a better annotation for the aorta:\n"
                            "a) **Upper Edge**: The aortic arch is not visible in this CT projection. Therefore, the aorta should appear as a long, vertical red line (tubular structure), extending from the **top edge** of the image. A common mistake is the aorta overlay not reaching the very top of the image. Does the red overlay in Image 1 touch the **upper edge** of the image? Does the red overlay in Image 2 touch the **upper edge** of the image? Compare the two images and identify which one, if either, has the red overlay reaching the top edge more accurately.\n"
                            "b) **Location and Shape**: The aorta should be parallel to the spine, typically as a long vertical line in the midline of the body. The aorta may curve if the spine is also curved. Is the aorta correctly positioned in Image 1? Is it correctly positioned in Image 2? Compare both images and explain which overlay follows the correct position better.\n"
                            "c) **Completeness**: The aorta should extend from the top of the image to the pelvis (if the pelvis is visible in the scan). Is the aorta visible from the top edge to the pelvis in Image 1? Is it complete in Image 2? Compare the completeness of both images.")
#Accuracy:  80%, but errors were only in the trachea detection

DescendingAortaDescription=("Answer the following questions before concluding which overlay is a better annotation for the aorta:\n"
                            "a) **Upper Edge**: The aortic arch is not visible in this CT projection. Therefore, the aorta should appear as a long, vertical red line (tubular structure), extending from the **top edge** of the image. The overlay must display the aorta at the very top of the image. Does the red overlay in Image 1 touch the **upper edge** of the image? Does the red overlay in Image 2 touch the **upper edge** of the image? Compare the two images and identify which one, if either, has the red overlay reaching the top edge more accurately.\n"
                            "b) **Location and Shape**: The aorta should be parallel to the spine, typically as a long vertical line in the midline of the body. The aorta may curve if the spine is also curved. Is the aorta correctly positioned in Image 1? Is it correctly positioned in Image 2? Compare both images and explain which overlay follows the correct position better.\n"
                            "c) **Completeness**: The aorta should extend from the top of the image to the pelvis (if the pelvis is visible in the scan). Is the aorta visible from the top edge to the pelvis in Image 1? Is it complete in Image 2? Compare the completeness of both images.")
#Accuracy:  80%, but errors were only in the trachea detection

Descriptions={
              "aorta":AortaDescription,
              "descending aorta":DescendingAortaDescription,
              "liver":LiverDescription,
              "kidneys":KidneysDescription,
              "adrenal_glands":AdrenalGlandDescription}

def Prompt3MessagesLMDeploy(img1, img2, img3, 
                            base_url='http://0.0.0.0:23333/v1', size=512,
                            text1=BodyRegionText, 
                            textOrganPresent=ComparisonTextContinued, 
                            textOrganNotPresent=NoOrganSimple, 
                            summarize=CompareSummarize, organ='liver',
                            save_memory=False, window='bone'):
    
    organRegion=text1 % {'organ': organ.replace('_',' ')}
    if organ=='aorta':
        if window=='skeleton':
            organRegion+=AorticArchTextSkeleton
        else:
            organRegion+=AorticArchText

    if size>224:
        conversation, answer = SendMessageLmdeploy([img1], conver=[], text=organRegion,
                                                base_url=base_url, size=224)
    else:
        conversation, answer = SendMessageLmdeploy([img1], conver=[], text=organRegion,
                                                base_url=base_url, size=size)
    q='q2'
    AnswerNo=('no' in answer.lower()[answer.lower().rfind(q):answer.lower().rfind(q)+7])
    if organ=='aorta':
        if ('no' in answer.lower()[answer.lower().rfind('q3'):answer.lower().rfind('q3')+7]):#no lungs
             organ='descending aorta'
        else:
            if ('yes' in answer.lower()[answer.lower().rfind('q5'):answer.lower().rfind('q5')+7]):#aortic arch present
                organ='aorta'
            else:
                organ='descending aorta'
    
    if AnswerNo:
        #text2 = NoOrganText % {'organ': organ.replace('_',' ')}
        conversation, answer = SendMessageLmdeploy([img3],text=textOrganNotPresent, conver=[],
                                               base_url=base_url, size=size)
        if 'image 1' in answer.lower() and 'image 2' not in answer.lower():
            return 1
        elif 'image 2' in answer.lower() and 'image 1' not in answer.lower():
            return 2
        else:
            return 0.5
    else:   
        text2 = textOrganPresent % {'organ': organ.replace('_',' ')} 
        text2 += Descriptions[organ]

    if save_memory:
        conversation=[]
        imgs=[img2]
    else:
        imgs=[img1,img2]

    conversation, answer = SendMessageLmdeploy(imgs,text=text2, conver=conversation,
                                                base_url=base_url, size=size)
    
    if 'overlay' not in answer.lower():
        return 0.5

    conversation, answer = SendMessageLmdeploy([], text=summarize+answer, conver=[],
                                               base_url=base_url, size=size)

    if 'overlay 1' in answer.lower() and 'overlay 2' not in answer.lower():
        return 1
    elif 'overlay 2' in answer.lower() and 'overlay 1' not in answer.lower():
        return 2
    else:
        return 0.5


def MultiTurnMultiImageComparisonLMDeploy(clean, y1, y2, compImg, 
                            base_url='http://0.0.0.0:23333/v1', size=448,
                            text1=BodyRegionText, 
                            textOrganPresent=ComparisonText, 
                            textOrganNotPresent=NoOrganSimple, 
                            summarize=CompareSummarize, organ='liver',
                            save_memory=False):
    '''
    Idea: send one image, check for errors. Send another one, check for errors. Send both, compare.
    '''
    
    
    conversation, answer = SendMessageLmdeploy([clean], conver=[], text=text1 % {'organ': organ.replace('_',' ')},
                                                base_url=base_url, size=256)
    
    AnswerNo=('no' in answer.lower()[answer.lower().rfind('q2'):answer.lower().rfind('q2')+15])
    
    if AnswerNo:
        #text2 = NoOrganText % {'organ': organ.replace('_',' ')}
        conversation, answer = SendMessageLmdeploy([compImg],text=textOrganNotPresent, conver=[],
                                               base_url=base_url, size=size)
        if 'image 1' in answer.lower() and 'image 2' not in answer.lower():
            return 1
        elif 'image 2' in answer.lower() and 'image 1' not in answer.lower():
            return 2
        else:
            return 0.5
    else:   
        comp = textOrganPresent % {'organ': organ.replace('_',' ')} 
        comp += Descriptions[organ]

    textED=ZeroShotInstructions%{'organ':organ.replace('_',' ')}
    textED+=DescriptionsED[organ]
    textED+='Justify your answer but answer concisely.'

    conversation1, answer1 = SendMessageLmdeploy([y1], text='This is Image 1. '+textED, conver=[],
                                               base_url=base_url, size=size//2)
    
    conversation2, answer2 = SendMessageLmdeploy([y2], text='This is Image 2. '+textED, conver=[],
                                               base_url=base_url, size=size//2)
    
    conversation, answer = SendMessageLmdeploy([y1,y2,compImg],text=comp, conver=conversation1+conversation2,
                                                base_url=base_url, size=[size//4,size//4,size])
    
    if 'overlay' not in answer.lower():
        return 0.5

    conversation, answer = SendMessageLmdeploy([], text=summarize+answer, conver=[],
                                               base_url=base_url, size=size)

    if 'overlay 1' in answer.lower() and 'overlay 2' not in answer.lower():
        return 1
    elif 'overlay 2' in answer.lower() and 'overlay 1' not in answer.lower():
        return 2
    else:
        return 0.5

def SimpleMultiImageComparisonLMDeploy(clean, y1, y2, compImg, 
                            base_url='http://0.0.0.0:23333/v1', size=448,
                            text1=BodyRegionText, 
                            textOrganPresent=ComparisonText2FigsContinued, 
                            textOrganNotPresent=NoOrganSimple, 
                            summarize=CompareSummarize, organ='liver',
                            save_memory=False):
    '''
    Idea: send one image, check for errors. Send another one, check for errors. Send both, compare.
    '''
    
    
    conversation, answer = SendMessageLmdeploy([clean], conver=[], text=text1 % {'organ': organ.replace('_',' ')},
                                                base_url=base_url, size=256)
    
    AnswerNo=('no' in answer.lower()[answer.lower().rfind('q2'):answer.lower().rfind('q2')+15])
    
    if AnswerNo:
        #text2 = NoOrganText % {'organ': organ.replace('_',' ')}
        conversation, answer = SendMessageLmdeploy([compImg],text=textOrganNotPresent, conver=[],
                                               base_url=base_url, size=size)
        if 'image 1' in answer.lower() and 'image 2' not in answer.lower():
            return 1
        elif 'image 2' in answer.lower() and 'image 1' not in answer.lower():
            return 2
        else:
            return 0.5
    else:   
        comp = textOrganPresent % {'organ': organ.replace('_',' ')} 
        comp += Descriptions[organ]

    textED=ZeroShotInstructions%{'organ':organ.replace('_',' ')}
    textED+=DescriptionsED[organ]
    textED+='Justify your answer but answer concisely.'

    conversation=CreateConversation([clean], 'This is Image 0, a frontal projection of a CT scan, it has some transparency and looks like an X-ray. ',
                                    conver=[],size=size//4)
    
    #if no previous conversation, send conver=[]. Do not automatically define conver above.
    
    conversation.append({
            'role': 'assistant',
            'content': [{
                'type': 'text',
                'text': 'Indeed, this is a frontal projection of a CT scan. It is not a CT slice, instead, it has transparency and lets you see through the entire human body, like an X-ray. ',
            }],
        })
    
    conversation=CreateConversation([y1], 'This is Image 1, it is the same frontal projection of a CT scan I shown in image 0, but now there is a red overlay, overlay 1, which tries to display the {organ}. ',
                                    conver=conversation,size=size//4)
    
    conversation.append({
            'role': 'assistant',
            'content': [{
                'type': 'text',
                'text': 'Indeed, this is the same frontal projection of a CT scan, but now there is a red overlay, overlay 1.',
            }],
        })
    
    conversation=CreateConversation([y2], 'This is Image 2, it is the same frontal projection of a CT scan I shown in Image 0 and Image 1, but now there is another red overlay, overlay 2, which is different from overlay 1, but also tries to display the {organ}. ',
                                    conver=conversation,size=size//4)
    
    conversation.append({
            'role': 'assistant',
            'content': [{
                'type': 'text',
                'text': 'Indeed, this is the same frontal projection of a CT scan, but now there is a red overlay, overlay 2. Overlay 2 is different from overlay 1.',
            }],
        })
    
    conversation, answer = SendMessageLmdeploy([y1,y2,compImg],text=comp, conver=conversation,
                                                base_url=base_url, size=[size//4,size//4,size])
    
    if 'overlay' not in answer.lower():
        return 0.5

    conversation, answer = SendMessageLmdeploy([], text=summarize+answer, conver=[],
                                               base_url=base_url, size=size)

    if 'overlay 1' in answer.lower() and 'overlay 2' not in answer.lower():
        return 1
    elif 'overlay 2' in answer.lower() and 'overlay 1' not in answer.lower():
        return 2
    else:
        return 0.5
    

def SystematicComparison3MessagesLMDeploy4ImageSequence(pth,base_url='http://0.0.0.0:23333/v1', 
                                          size=512,
                            text1=BodyRegionText, 
                    textOrganPresent='auto', 
                    textOrganNotPresent=NoOrganSimple, 
                    summarize=CompareSummarize, organ='liver',
                    file_structure='original',
                    dice_check=False,pth1=None,pth2=None,save_memory=False):
        
        answers=[]
        outputs={}

        if textOrganPresent=='auto':
            textOrganPresent=ComparisonText2FigsContinued
            textOrganNotPresent=NoOrganSimple

        for target in os.listdir(pth):
            if '_ct_' not in target:
                continue
            clean=os.path.join(pth,target)
            twoImages=clean.replace('ct_window_bone','composite_image_2_figs')
            y1=clean.replace('ct_window_bone','overlay_window_bone').replace('.png','_y1.png')
            y2=clean.replace('ct_window_bone','overlay_window_bone').replace('.png','_y2.png')
            #print(target)
            #print('clean:',clean)
            #print('y1:',y1)
            #print('y2:',y2)
            #print('twoImages:',twoImages)


            print(target)

            if dice_check:
                #print('pth:',pth)
                pid=target[:target.rfind('_ct')]
                #print(os.listdir(pth1))
                #print('pid:',pid)
                dice=check_dice(y1,y2)
                #dice=check_dice_on_composite_2_figs(twoImages)
                print('2D dice coefficient between 2 projections on axis 1:',dice)
                if dice>0.9:
                    print('The projections are too similar for case {target}, skipping the comparison. Try another axis or ct compare slices (holes?).')
                    continue

            #consider that the correct answer is 2
            answer=SimpleMultiImageComparisonLMDeploy(
                            clean=clean,y1=y1,y2=y2,compImg=twoImages,
                            base_url=base_url,size=size,
                            text1=text1,
                            textOrganPresent=textOrganPresent,
                            textOrganNotPresent=textOrganNotPresent,
                            summarize=summarize,
                            organ=organ,save_memory=save_memory)
            
            print('Traget:',target,'Answer:',answer,'Label: Overlay 2')
            if answer==1:
                answers.append(0)
            elif answer==2:
                answers.append(1)
            outputs[target]=answer
        
            # Clean up
            del answer
            torch.cuda.empty_cache()
            gc.collect()
        acc=np.array(answers).sum()/len(answers)
        acc=np.round(100*acc,1)
        print('Accuracy: ',acc, '(',np.array(answers).sum(),'/',len(answers), ')')

        for k,v in outputs.items():
            print(k,v)






            
def SystematicComparison3MessagesLMDeploy(pth,base_url='http://0.0.0.0:23333/v1', 
                                          size=512,
                            text1=BodyRegionText, 
                    textOrganPresent='auto', 
                    textOrganNotPresent='auto', 
                    summarize=CompareSummarize, organ='liver',
                    file_structure='original',
                    dice_check=False,pth1=None,pth2=None,
                    save_memory=False):
        
        answers=[]
        outputs={}

        if textOrganPresent=='auto':
            if organ[-1]!='s':
                textOrganPresent=ComparisonText
                textOrganNotPresent=NoOrganSimple

            else:
                #textOrganPresent=ComparisonText2Classes%{'organ_singular':organ[:-1],'organ':organ}
                #textOrganNotPresent=NoOrganSimple2Classes
                textOrganPresent=ComparisonText
                textOrganNotPresent=NoOrganSimple

        for target in os.listdir(pth):
            if file_structure=='original':
                if 'overlay_axis_1' not in target or 'BestIs' in target:
                    continue
                anno=os.path.join(pth,target)
                clean=anno.replace('overlay','ct')
                twoImages=anno.replace('overlay_','2BoneImages')
            else:
                if 'ct_window_bone_axis_1' not in target:
                    continue
                clean=os.path.join(pth,target)
                anno=clean.replace('ct_window_bone','composite_image')
                twoImages=clean.replace('ct_window_bone','composite_image_2_figs')
                print('anno:',anno)

            print(target)

            if dice_check:
                #print('pth:',pth)
                pid=target[:target.rfind('_ct')]
                #print(os.listdir(pth1))
                #print('pid:',pid)
                #dice=check_dice(os.path.join(pth1,pid,pid+'_overlay_window_bone_axis_1_'+organ+'.png'),
                #                os.path.join(pth2,pid,pid+'_overlay_window_bone_axis_1_'+organ+'.png'))
                dice=check_dice_on_composite_2_figs(twoImages)
                print('2D dice coefficient between 2 projections on axis 1:',dice)
                if dice>0.9:
                    print('The projections are too similar for case {target}, skipping the comparison. Try another axis or ct compare slices (holes?).')
                    continue

            #consider that the correct answer is 2
            answer=Prompt3MessagesLMDeploy(
                            img1=clean,img2=anno,img3=twoImages,
                            base_url=base_url,size=size,
                            text1=text1,
                            textOrganPresent=textOrganPresent,
                            textOrganNotPresent=textOrganNotPresent,
                            summarize=summarize,
                            organ=organ, save_memory=save_memory)
            
            print('Traget:',target,'Answer:',answer,'Label: Overlay 2')
            if answer==1:
                answers.append(0)
            elif answer==2:
                answers.append(1)
            outputs[target]=answer
        
            # Clean up
            del answer
            torch.cuda.empty_cache()
            gc.collect()
        acc=np.array(answers).sum()/len(answers)
        acc=np.round(100*acc,1)
        print('Accuracy: ',acc, '(',np.array(answers).sum(),'/',len(answers), ')')

        for k,v in outputs.items():
            print(k,v)

def SystematicComparison3MessagesLMDeploy2Figs(pth,base_url='http://0.0.0.0:23333/v1', 
                                          size=512,
                            text1=BodyRegionText, 
                    textOrganPresent='auto', 
                    textOrganNotPresent=NoOrganSimple, 
                    summarize=CompareSummarize2Figs, organ='liver',
                    file_structure='original',
                    dice_check=False,pth1=None,pth2=None,save_memory=False,
                    window='bone'):
        
        if window=='skeleton':
            text1=BodyRegionTextHighlightedSkeleton
        answers=[]
        outputs={}

        if textOrganPresent=='auto':
            if organ[-1]!='s':
                textOrganPresent=ComparisonText2Figs
                textOrganNotPresent=NoOrganSimple
            else:
                #textOrganPresent=ComparisonText2Classes2Figs%{'organ_singular':organ[:-1],'organ':organ}
                #textOrganNotPresent=NoOrganSimple2Classes
                textOrganPresent=ComparisonText2Figs
                textOrganNotPresent=NoOrganSimple

        for target in os.listdir(pth):
            if file_structure=='original':
                if 'overlay_axis_1' not in target or 'BestIs' in target:
                    continue
                anno=os.path.join(pth,target)
                clean=anno.replace('overlay','ct')
                twoImages=anno.replace('overlay_','2BoneImages')
            else:
                if 'ct_window_bone_axis_1' not in target:
                    continue
                clean=os.path.join(pth,target)
                anno=clean.replace('ct_window_bone','composite_image')
                twoImages=clean.replace('ct_window_bone','composite_image_2_figs')

            print(target)

            if dice_check:
                #print('pth:',pth)
                pid=target[:target.rfind('_ct')]
                #print(os.listdir(pth1))
                #print('pid:',pid)
                dice=check_dice(clean.replace('ct_window_bone','overlay_window_bone').replace('.png','_y1.png'),
                                clean.replace('ct_window_bone','overlay_window_bone').replace('.png','_y2.png'))
                #dice=check_dice_on_composite_2_figs(twoImages)
                print('2D dice coefficient between 2 projections on axis 1:',dice)
                if dice>0.9:
                    print('The projections are too similar for case {target}, skipping the comparison. Try another axis or ct compare slices (holes?).')
                    continue

            if window=='skeleton':
                #clean=clean[:clean.rfind('ct_window_bone')]+'composite_ct_2_figs_axis_1_skeleton.png'
                #clean=clean.replace('ct_window_bone','composite')
                clean=clean[:clean.rfind('ct_window_bone')]+'highlighted_skeleton.png'
                #clean=clean.replace('ct_window_bone','ct_window_skeleton')
                print('anno:',anno)
                print('clean:',clean)

            #consider that the correct answer is 2
            answer=Prompt3MessagesLMDeploy(
                            img1=clean,img2=twoImages,img3=twoImages,
                            base_url=base_url,size=size,
                            text1=text1,
                            textOrganPresent=textOrganPresent,
                            textOrganNotPresent=textOrganNotPresent,
                            summarize=summarize,
                            organ=organ,save_memory=save_memory,
                            window=window)
            
            print('Traget:',target,'Answer:',answer,'Label: Overlay 2')
            if answer==1:
                answers.append(0)
            elif answer==2:
                answers.append(1)
            outputs[target]=answer
        
            # Clean up
            del answer
            torch.cuda.empty_cache()
            gc.collect()
        acc=np.array(answers).sum()/len(answers)
        acc=np.round(100*acc,1)
        print('Accuracy: ',acc, '(',np.array(answers).sum(),'/',len(answers), ')')

        for k,v in outputs.items():
            print(k,v)




organ_list=['adrenal_gland_left',
 'adrenal_gland_right',
 'aorta',
 'bladder',
 'celiac_trunk',
 'colon',
 'duodenum',
 'esophagus',
 'femur_left',
 'femur_right',
 'gall_bladder',
 'hepatic_vessel',
 'intestine',
 'kidney_left',
 'kidney_right',
 'liver',
 'lung_left',
 'lung_right',
 'pancreas',
 'portal_vein_and_splenic_vein',
 'postcava',
 'prostate',
 'rectum',
 'spleen',
 'stomach']

# Function to calculate Dice coefficient
def calculate_dice(mask1, mask2):
    intersection = torch.sum(mask1 & mask2)
    total = torch.sum(mask1) + torch.sum(mask2)
    dice = (2.0 * intersection) / (total+1e-6)
    return dice.item()

# Function to process the image
def check_dice_on_composite_2_figs(image_path):
    import cv2
    # 1- Load the image
    img = np.array(Image.open(image_path))
    
    # Convert image to grayscale to simplify background detection
    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Threshold to create a binary image (X-rays are darker, so we invert the threshold)
    _, binary_img = cv2.threshold(gray_img, 240, 255, cv2.THRESH_BINARY_INV)

    # Find contours to detect the X-ray areas
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Sort contours by area and get the largest two (assuming these are the X-rays)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:2]

    # Extract bounding boxes for the two largest contours
    xray_imgs = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        xray_imgs.append(img[y:y+h, x:x+w])  # Crop the X-ray based on bounding box

    # Ensure we have two sub-images (left and right X-rays)
    img1, img2 = xray_imgs

    # Create masks for each sub-image
    def create_mask(sub_img):
        # The red color in the image has high values for the red channel and low values for the other channels
        lower_red = np.array([150, 0, 0])  # Lower bound for red
        upper_red = np.array([255, 100, 100])  # Upper bound for red
        mask = cv2.inRange(sub_img, lower_red, upper_red)
        return mask // 255  # Normalize mask to binary (0 and 1)

    mask1 = create_mask(img1)
    mask2 = create_mask(img2)

    # Calculate Dice coefficient
    intersection = np.sum(mask1 * mask2)
    dice = (2. * intersection) / (np.sum(mask1) + np.sum(mask2))
    
    return dice

def check_dice(image_path1, image_path2):
    # Load images using PIL and convert to RGB
    img1 = Image.open(image_path1).convert('RGB')
    img2 = Image.open(image_path2).convert('RGB')

    # Convert images to NumPy arrays
    np_img1 = np.array(img1)
    np_img2 = np.array(img2)

    # Convert images to PyTorch tensors and move to GPU
    tensor1 = torch.from_numpy(np_img1).to('cuda')
    tensor2 = torch.from_numpy(np_img2).to('cuda')

    # Create binary mask where image is not greyscale
    mask1 = (tensor1[:, :, 0] != tensor1[:, :, 1]).to(torch.uint8)
    mask2 = (tensor2[:, :, 0] != tensor2[:, :, 1]).to(torch.uint8)

    # Calculate the Dice coefficient
    intersection = torch.sum(mask1 & mask2)
    total = torch.sum(mask1) + torch.sum(mask2)

    dice = (2.0 * intersection) / total

    return dice.item()

def project_and_compare(ct, y1, y2, base_url='http://0.0.0.0:23333/v1', 
                        size=512, organ=None, temp_dir='random',
                        text1=BodyRegionText, 
                        textOrganPresent=ComparisonText2Figs, 
                        textOrganNotPresent=NoOrganSimple, 
                        summarize=CompareSummarize2Figs,axis=1,
                        checkDice=True,window='bone'):
    
    if organ is None:
        for org in organ_list:
            if org in y1[y1.rfind('/')+1:]:
                organ=org
                break
        if organ is None:
            raise ValueError('Organ not found in annotation filenames. Please set it explicitly using the organ parameter.')
        print('Organ inferenced automatically from the annotation filename:', organ)

    # Project the CT scan
    if temp_dir=='random':
        temp_dir='./tmp'+str(random.randint(0,10000))
    os.makedirs(temp_dir, exist_ok=True)

    prj.overlay_projection_fast(pid='y1_bone', organ=organ, datapath=None, save_path=temp_dir,
                           ct_path=ct,mask_path=y1,
                           ct_only=False,window=window,axis=axis)
    
    prj.overlay_projection_fast(pid='y2_bone', organ=organ, datapath=None, save_path=temp_dir,
                           ct_path=ct,mask_path=y2,
                           ct_only=False,window=window,axis=axis)
    
    if checkDice:
        dice=check_dice(os.path.join(temp_dir,'y1_bone_overlay_window_'+window+'_axis_1_liver.png'),
                        os.path.join(temp_dir,'y2_bone_overlay_window_'+window+'_axis_1_liver.png'))
        print('2D dice coefficient between 2 projections on axis '+str(axis)+':',dice)
        if dice>0.9:
            print('The projections are too similar, skipping the comparison and returning 1.5. Try another axis or ct compare slices (holes?).')
            return 1.5

    prj.overlay_projection_fast(pid='ct', organ=organ, datapath=None, save_path=temp_dir,
                           ct_path=ct,mask_path=y1,
                           ct_only=True,window=window,axis=axis)
    
    prj.overlay_projection_fast(pid='y1_organs', organ=organ, datapath=None, save_path=temp_dir,
                           ct_path=ct,mask_path=y1,
                           ct_only=False,window='organs',axis=axis)

    
    prj.overlay_projection_fast(pid='y2_organs', organ=organ, datapath=None, save_path=temp_dir,
                           ct_path=ct,mask_path=y2,
                           ct_only=False,window='organs',axis=axis)
    
    #prj.create_composite_image(temp_dir, organ,axis=axis,window=window)
    prj.create_composite_image_2figs(temp_dir, organ,axis=axis,window=window)
    
    #API call to LLM
    ct=os.path.join(temp_dir,'ct_ct_window_'+window+'_axis_1_liver.png')
    fourImages=os.path.join(temp_dir,'composite_image_axis_1_liver.png')
    twoImages=os.path.join(temp_dir,'composite_image_2_figs_axis_1_liver.png')
    #consider that the correct answer is 2
    answer=Prompt3MessagesLMDeploy(
                    img1=ct,img2=twoImages,img3=twoImages,
                    base_url=base_url,size=size,
                    text1=text1,
                    textOrganPresent=textOrganPresent,
                    textOrganNotPresent=textOrganNotPresent,
                    summarize=summarize,
                    organ=organ)
    
    print('Answer:',answer)
    shutil.rmtree(temp_dir)
    return answer

    

def SystematicComparison3MessagesLMDeploy6Figs(pth,base_url='http://0.0.0.0:23333/v1', 
                                          size=512,
                            text1=BodyRegionText, 
                    textOrganPresent=ComparisonText6Figs+LiverDescription, 
                    textOrganNotPresent=NoOrganSimple, 
                    summarize=CompareSummarize6Figs, organ='liver'):
        answers=[]
        outputs={}

        for target in os.listdir(pth):
            if 'overlay_axis_1' not in target or 'BestIs' in target:
                continue
            print(target)

            anno=os.path.join(pth,target)
            clean=anno.replace('overlay','ct')
            twoImages=anno.replace('overlay_','2BoneImages')
            anno=anno.replace('overlay_','6Images')
            #consider that the correct answer is 2
            answer=Prompt3MessagesLMDeploy(
                            img1=clean,img2=anno,img3=twoImages,
                            base_url=base_url,size=size,
                            text1=text1,
                            textOrganPresent=textOrganPresent,
                            textOrganNotPresent=textOrganNotPresent,
                            summarize=summarize,
                            organ=organ)
            print('Traget:',target,'Answer:',answer,'Label: Overlay 2')
            if answer==1:
                answers.append(0)
            elif answer==2:
                answers.append(1)
            outputs[target]=answer
        
            # Clean up
            del answer
            torch.cuda.empty_cache()
            gc.collect()
        acc=np.array(answers).sum()/len(answers)
        acc=np.round(100*acc,1)
        print('Accuracy: ',acc, '(',np.array(answers).sum(),'/',len(answers), ')')

        for k,v in outputs.items():
            print(k,v)

def SystematicComparison3MessagesLMDeploy2FigsOld(pth,base_url='http://0.0.0.0:23333/v1', 
                                          size=512,
                            text1=BodyRegionText, 
                    textOrganPresent=ComparisonText2Figs+LiverDescription, 
                    textOrganNotPresent=NoOrganSimple, 
                    summarize=CompareSummarize2Figs, organ='liver',
                    mode='Tissue',save_memory=True):
        answers=[]
        outputs={}

        for target in os.listdir(pth):
            if 'overlay_axis_1' not in target or 'BestIs' in target:
                continue
            print(target)

            anno=os.path.join(pth,target)
            clean=anno.replace('overlay','ct')
            if mode=='bone':
                twoImages=anno.replace('overlay_','2BoneImages')
            else:
                twoImages=anno.replace('overlay_','2TissueImages')
            #consider that the correct answer is 2
            answer=Prompt3MessagesLMDeploy(
                            img1=clean,img2=twoImages,img3=twoImages,
                            base_url=base_url,size=size,
                            text1=text1,
                            textOrganPresent=textOrganPresent,
                            textOrganNotPresent=textOrganNotPresent,
                            summarize=summarize,
                            organ=organ,save_memory=save_memory)
            print('Traget:',target,'Answer:',answer,'Label: Overlay 2')
            if answer==1:
                answers.append(0)
            elif answer==2:
                answers.append(1)
            outputs[target]=answer
        
            # Clean up
            del answer
            torch.cuda.empty_cache()
            gc.collect()
        acc=np.array(answers).sum()/len(answers)
        acc=np.round(100*acc,1)
        print('Accuracy: ',acc, '(',np.array(answers).sum(),'/',len(answers), ')')

        for k,v in outputs.items():
            print(k,v)


def SystematicComparison3MessagesLMDeploy1Fig(pth,base_url='http://0.0.0.0:23333/v1', 
                                          size=512,
                            text1=BodyRegionText, 
                    textOrganPresent=ComparisonText1Fig+LiverDescription, 
                    textOrganNotPresent=NoOrganSimple, 
                    summarize=CompareSummarize1Fig, organ='liver',
                    mode='bone'):
        answers=[]
        outputs={}

        for target in os.listdir(pth):
            if 'overlay_axis_1' not in target or 'BestIs' in target:
                continue
            print(target)

            anno=os.path.join(pth,target)
            clean=anno.replace('overlay','ct')
            if mode=='bone':
                superpos=anno.replace('overlay_','superpositionBone_')
                twoImages=anno.replace('overlay_','2BoneImages')
            else:
                superpos=anno.replace('overlay_','superpositionTissue_')
                twoImages=anno.replace('overlay_','2TissueImages')

            
            #consider that the correct answer is 2
            answer=Prompt3MessagesLMDeploy(
                            img1=clean,img2=superpos,img3=twoImages,
                            base_url=base_url,size=size,
                            text1=text1,
                            textOrganPresent=textOrganPresent,
                            textOrganNotPresent=textOrganNotPresent,
                            summarize=summarize,
                            organ=organ)
            print('Traget:',target,'Answer:',answer,'Label: Overlay 2')
            if answer==1:
                answers.append(0)
            elif answer==2:
                answers.append(1)
            outputs[target]=answer
        
            # Clean up
            del answer
            torch.cuda.empty_cache()
            gc.collect()
        acc=np.array(answers).sum()/len(answers)
        acc=np.round(100*acc,1)
        print('Accuracy: ',acc, '(',np.array(answers).sum(),'/',len(answers), ')')

        for k,v in outputs.items():
            print(k,v)


SinglePrompt=("The images I am sending are frontal projections of one CT scan. They are not CT slices, instead, they have transparency and let you see through the entire human body, like an X-ray does. All 4 images are projections of the same 3D CT scan. \n"
              "Here is a full description of each image:\n"
"Image 1: frontal projection of the CT scan using a wide window (2000 UI),  tuned for better bone visualization. The image is superposed with liver overlay 1.\n"
"Image 2: frontal projection of a CT scan using a less wide window (400 UI),  tuned for better abdominal organ visualization. The image is also superposed with liver overlay 1 (like image 1).\n"
"Image 3: frontal projection of the CT scan using a wide window (2000 UI),  tuned for better bone visualization. The image is superposed with liver overlay 2.\n"
"Image 4: frontal projection of a CT scan using a less wide window (400 UI),  tuned for better abdominal organ visualization. The image is also superposed with liver overlay 2 (like image 3).\n"
"In all images, a blue R indicates the right side of the human body, and a green L indicates the left side. Answer the following questions in a stepwise manner:\n"
"Q1- Look carefully, tell me which body region the CT scan represents and where its limits are. Provide a detailed list of all organs usually contained in this region.\n"
"Q2- Based on your answer to question Q1, should the %(organ)s be present in this image? Answer with just ‘yes’ or ‘no’.. If your answer is ‘no’, fill the following template, substituting  _ by Yes or No, and add it to the end of your answer:\n"
"\n"
"Q2 = _\n"
"\n"
"- If your answer to Q2 is ‘yes’, continue to question 3 (below). \n"
"- If your answer to Q2 is ‘no’, ignore Q3 and tell me which overlay has smallest amount of red color. Then, fill the template substituting _ by 1 or 2 (do not fill the template if your answer to Q2 is yes): \n"
"'The best overlay is Overlay _.'\n"
"\n"
"Q3- The %(organ)s region in the images should be marked in red, using an overlay. However, the red overlays may correctly or incorrectly mark the %(organ)s. The letters R (blue) and L (green) inside the images represent the right and left sides of the human body. Compare overlay 1 (shown in images 1 and 2) to overlay 2 (shown in images 3 and 4) and tell me which one is a better overlay for the %(organ)s.\n")
DescribeLiver=("When evaluating and comparing the overlays, consider the following anatomical information:\n"
"a) The liver is a large organ, with triangular or wedge-like shape.\n"
"b) The liver is located in the upper right quadrant of the abdomen (right is indicated with a blue R in the figures), just below the diaphragm. It spans across the midline, partially extending into the left upper quadrant of the abdomen. It spans across the midline, partially extending into the left upper quadrant. Looking at the bones images 1 and 3, you can easily identify if the liver should appear in the images or not. The liver is not in the pelvis.\n"
"c) The liver is a single structure. \n"
"d) The liver position is primarily under the rib cage.\n"
"e) If the CT scan does not include the liver region (for example, a scan showing just the pelvis), the correct overlay is actually the one showing no red region (or as little as possible).\n")
SinglePromptSepImages=("The images I am sending are frontal projections of one CT scan. They are not CT slices, instead, they have transparency and let you see through the entire human body, like an X-ray does. In all images, a blue R indicates the right side of the human body, and a green L indicates the left side. All 3 images are projections of the same 3D CT scan. \n"
"Image 1: frontal projection of the CT scan using a wide window (2000 UI) for better bone visualization, you can use this image to better understand which body region the CT scan encompasses.\n"
"Image 2: frontal projection of a CT scan using a narrow window (400 UI) for better organ visualization. The image is also superposed with %(organ)s overlay 1, in red.\n"
"Image 3: the same frontal projection of a CT scan as image 1 and 2, but superposed with %(organ)s overlay 2.\n"
"Answer the following questions in a stepwise manner:\n"
"Q1- Look carefully, tell me which body region the CT scan represents and where its limits are. Provide a detailed list of all organs usually contained in this region.\n"
"Q2- Based on your answer to question Q1, should the %(organ)s be present in this image? Answer with just ‘yes’ or ‘no’.. If your answer is ‘no’, fill the following template, substituting  _ by Yes or No, and add it to the end of your answer:\n"
"\n"
"Q2 = _\n"
"\n"
"- If your answer to Q2 is ‘yes’, continue to question 3 (below). \n"
"- If your answer to Q2 is ‘no’, ignore Q3 and tell me which overlay has smallest amount of red color. Then, fill the template substituting _ by 1 or 2 (do not fill the template if your answer to Q2 is yes): \n"
"'The best overlay is Overlay _.'\n"
"\n"
"Q3- The %(organ)s region in the images should be marked in red, using an overlay. However, the red overlays may correctly or incorrectly mark the %(organ)s. The letters R (blue) and L (green) inside the images represent the right and left sides of the human body. Compare overlay 1 (shown in images 1 and 2) to overlay 2 (shown in images 3 and 4) and tell me which one is a better overlay for the %(organ)s.\n")
compareSummarizeSepImages=("The text below represents a comparisons of 2 overlays, 'Overlay 1' and 'Overlay 2'. "
                "A LVLM like you compared the 2 overlays by analyzing images. Its answer is the text below."
                "The text explains which overlay (or image) is better. I want you to answer me which overaly is better according to the text. Answer me with only 2 words: 'Overlay 1' or 'Overlay 2'. "
                "The text is:\n")

def Prompt2MessagesLMDeploy(img, base_url='http://0.0.0.0:23333/v1', size=512,
                    text1=SinglePrompt,
                    summarize=CompareSummarize, organ='liver'):
    if organ=='liver':
        organDescription=DescribeLiver

    _, answer = SendMessageLmdeploy([img], base_url=base_url, size=size ,conver=[],
                                    text=text1 % {'organ': organ.replace('_',' ')}+organDescription)
    
    if answer=='':
        return 0.5

    _, answer = SendMessageLmdeploy([], base_url=base_url, size=size, text=summarize+answer, conver=[])

    if 'overlay 2' in answer.lower() and 'overlay 1' not in answer.lower():
        return 2
    elif 'overlay 1' in answer.lower() and 'overlay 2' not in answer.lower():
        return 1
    else:
        return 0.5
    
def Prompt2MessagesMultiImageLMDeploy(img1,img2,img3, base_url='http://0.0.0.0:23333/v1', size=512,
                    text1=SinglePromptSepImages,
                    summarize=compareSummarizeSepImages, organ='liver'):
    
    if organ=='liver':
        organDescription=DescribeLiver

    _, answer = SendMessageLmdeploy([img1,img2,img3], base_url=base_url, size=size ,conver=[],
                                    text=text1 % {'organ': organ.replace('_',' ')}+organDescription)
    
    if answer=='':
        return 0.5

    _, answer = SendMessageLmdeploy([], base_url=base_url, size=size, text=summarize+answer, conver=[])

    if 'overlay 2' in answer.lower() and 'overlay 1' not in answer.lower():
        return 2
    elif 'overlay 1' in answer.lower() and 'overlay 2' not in answer.lower():
        return 1
    else:
        return 0.5
    
def SystematicComparison2MessagesLMDeployMultiImage(pth,base_url='http://0.0.0.0:23333/v1', 
                                          size=224,
                            text1=SinglePromptSepImages, 
                    summarize=compareSummarizeSepImages, organ='liver'):
    answers=[]

    outputs={}

    for target in os.listdir(pth):
        if 'ct_axis_1' not in target or 'BestIs' in target:
            continue
        print(target)

        ct=os.path.join(pth,target)
        overlay1=ct.replace('ct_','overlay_bad_tissue')
        overlay2=ct.replace('ct_','overlay_better_tissue')

        #consider that the correct answer is 2
        answer=Prompt2MessagesMultiImageLMDeploy(
                        img1=ct,
                        img2=overlay1,
                        img3=overlay2,
                        base_url=base_url,size=size,
                        text1=text1,
                        summarize=summarize,
                        organ=organ)
        print('Traget:',target,'Answer:',answer,'Label: Overlay 2')
        if answer==1:
            answers.append(0)
        elif answer==2:
            answers.append(1)
        outputs[target]=answer

        # Clean up
        del answer
        torch.cuda.empty_cache()
        gc.collect()
    acc=np.array(answers).sum()/len(answers)
    acc=np.round(100*acc,1)
    print('Accuracy: ',acc, '(',np.array(answers).sum(),'/',len(answers), ')')
    print('Answer:',answers)
    for k,v in outputs.items():
        print(k,v)
    
def SystematicComparison2MessagesLMDeploy(pth,base_url='http://0.0.0.0:23333/v1', 
                                          size=512,
                            text1=SinglePrompt, 
                    summarize=CompareSummarize, organ='liver'):
    answers=[]

    outputs={}

    for target in os.listdir(pth):
        if 'overlay_axis_1' not in target or 'BestIs' in target:
            continue
        print(target)

        anno=os.path.join(pth,target)
        #consider that the correct answer is 2
        answer=Prompt2MessagesLMDeploy(
                        img=anno,
                        base_url=base_url,size=size,
                        text1=text1,
                        summarize=summarize,
                        organ=organ)
        print('Traget:',target,'Answer:',answer,'Label: Overlay 2')
        if answer==1:
            answers.append(0)
        elif answer==2:
            answers.append(1)
        outputs[target]=answer

        # Clean up
        del answer
        torch.cuda.empty_cache()
        gc.collect()
    acc=np.array(answers).sum()/len(answers)
    acc=np.round(100*acc,1)
    print('Accuracy: ',acc, '(',np.array(answers).sum(),'/',len(answers), ')')
    print('Answer:',answers)
    for k,v in outputs.items():
        print(k,v)

def SystematicComparison2MessagesLMDeploySimple(pth,base_url='http://0.0.0.0:23333/v1', 
                                          size=512,
                            text1=SinglePrompt, 
                    summarize=CompareSummarize, organ='liver'):
    answers=[]

    outputs={}

    for target in os.listdir(pth):
        if 'overlay_axis_1' not in target or 'BestIs' in target:
            continue
        print(target)

        anno=os.path.join(pth,target)
        #consider that the correct answer is 2
        answer=Prompt2MessagesLMDeploy(
                        img=anno,
                        base_url=base_url,size=size,
                        text1=text1,
                        summarize=summarize,
                        organ=organ)
        print('Traget:',target,'Answer:',answer,'Label: Overlay 2')
        if answer==1:
            answers.append(0)
        elif answer==2:
            answers.append(1)
        outputs[target]=answer

        # Clean up
        del answer
        torch.cuda.empty_cache()
        gc.collect()
    acc=np.array(answers).sum()/len(answers)
    acc=np.round(100*acc,1)
    print('Accuracy: ',acc, '(',np.array(answers).sum(),'/',len(answers), ')')
    print('Answer:',answers)
    for k,v in outputs.items():
        print(k,v)
        



def Prompt3MessagesQwen(img1, img2, model, processor,process_vision_info,
                    text1=BodyRegionText, 
                    textOrganPresent=ComparisonText+LiverDescription, 
                    textOrganNotPresent=NoOrganText, summarize=CompareSummarize, organ='liver'):
    
    conversation, answer = SendMessageQwen([img1], model, processor, process_vision_info=process_vision_info, text=text1 % {'organ': organ.replace('_',' ')})

    
    AnswerNo=('no' in answer.lower()[answer.lower().rfind('q2'):answer.lower().rfind('q2')+15])
    
    if AnswerNo:
        text2 = textOrganNotPresent % {'organ': organ.replace('_',' ')}
    else:   
        text2 = textOrganPresent % {'organ': organ.replace('_',' ')}

    conversation, answer = SendMessageQwen([img1,img2], model, processor, process_vision_info=process_vision_info, text=text2, conversation=conversation)

    conversation, answer = SendMessageQwen([], model, processor, process_vision_info=process_vision_info, text=summarize+answer, conversation=[])

    if 'overlay 1' in answer.lower() and 'overlay 2' not in answer.lower():
        return 1
    elif 'overlay 2' in answer.lower() and 'overlay 1' not in answer.lower():
        return 2
    else:
        return 0.5
    

    
def SystematicComparison3MessagesQwen(pth,model,processor,process_vision_info,
                            text1=BodyRegionText, 
                    textOrganPresent=ComparisonText+LiverDescription, 
                    textOrganNotPresent=NoOrganText, 
                    summarize=CompareSummarize, organ='liver'):
        answers=[]

        for target in os.listdir(pth):
            if 'overlay_axis_1' not in target or 'BestIs' in target:
                continue
            print(target)

            anno=os.path.join(pth,target)
            clean=anno.replace('overlay','ct')
            #consider that the correct answer is 2
            answer=Prompt3MessagesQwen(
                            img1=clean,img2=anno,
                            model=model,processor=processor,
                            process_vision_info=process_vision_info,
                            text1=text1,
                            textOrganPresent=textOrganPresent,
                            textOrganNotPresent=textOrganNotPresent,
                            summarize=summarize,
                            organ=organ)
            print('Traget:',target,'Answer:',answer,'Label: Overlay 2')
            if answer==1:
                answers.append(0)
            elif answer==2:
                answers.append(1)
            
            # Clean up
            del answer
            torch.cuda.empty_cache()
            gc.collect()
        acc=np.array(answers).sum()/len(answers)
        acc=np.round(100*acc,1)
        print('Accuracy: ',acc, '(',np.array(answers).sum(),'/',len(answers), ')')
