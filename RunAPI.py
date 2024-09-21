import os
os.environ['TRANSFORMERS_CACHE'] = './HFCache'
os.environ['HF_HOME'] = './HFCache'
os.environ['CUDA_VISIBLE_DEVICES'] = '7'

import ErrorDetector as ed
import importlib
import os
importlib.reload(ed)

#annos='CompositeComparison4Images/'
annos='/mnt/sdc/pedro/ErrorDetection/projections_good_n_bad_liver_best_is_2'


#ed.SystematicComparison3MessagesLMDeploy2Figs(pth=annos,size=512)
#512 p; 10/12, 2 out-of-tokens. 83.33% accuracy.
#224p: 8/12, still out of token.
ed.SystematicComparison3MessagesLMDeploy(pth=annos,size=512,file_structure='new')
#40B model: 91.7% (11/12)
#26B model: 66.7%
#ed.SystematicComparison3MessagesLMDeploy(pth=annos,size=224)
#ed.SystematicComparison2MessagesLMDeploy(pth=annos,size=512) # This function will be called to run the error detection
#bad-50%
#76B: can follow the prompt very well, but Accuracy:  66.7 ( 8 / 12 ), 512p

#ed.SystematicComparison3MessagesLMDeploy1Fig(pth=annos,size=224,mode='tissue')
#72.7%, 224p, bone
#Accuracy:  88.9 ( 8 / 9 ), 224p, tissue
#Accuracy:  81.8 ( 9 / 11 ), 224p, bone
#Accuracy:  45, 512p, tissue
#ed.SystematicComparison3MessagesLMDeploy6Figs(pth=annos,size=512)
#80% 512p, 2 out-of-tokens

#ed.SystematicComparison2MessagesLMDeployMultiImage(pth='good_and_bad_liver_RL')
#Accuracy:  42.9 ( 3 / 7 )