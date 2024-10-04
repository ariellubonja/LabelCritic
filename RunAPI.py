import os
os.environ['TRANSFORMERS_CACHE'] = './HFCache'
os.environ['HF_HOME'] = './HFCache'
os.environ['CUDA_VISIBLE_DEVICES'] = '7'

import ErrorDetector as ed
import importlib 
import os
importlib.reload(ed)

annos='../compose_nnUnet_JHH/gall_bladder/'


ed.SystematicComparisonLMDeploySepFigures(pth=annos,size=512,organ='gall_bladder', dice_check=True,save_memory=True,solid_overlay='auto',
                                          multi_image_prompt_2=True,dual_confirmation=True,conservative_dual=False,dice_th=0.7,
                                          base_url='http://0.0.0.0:9001/v1')