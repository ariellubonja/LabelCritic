import argparse
import os
import ErrorDetector as ed
import importlib
import re

# Reload the module in case it has been updated
importlib.reload(ed)

# Set up argument parsing
parser = argparse.ArgumentParser(description='Run SystematicComparisonLMDeploySepFigures with the specified path.')
parser.add_argument('--path', help='Path to the annotations')
parser.add_argument('--port', help='VLLM port to use', default='8000')
parser.add_argument('--dice_th', help='VLLM port to use', default='0.75')
parser.add_argument('--organ_list', help='List of organs to process', default='auto')
parser.add_argument('--csv_path', help='path of casv to save results', default=None)
parser.add_argument('--continuing', action='store_true', help="Continues from interrupted run.")
    

# Parse the arguments
args = parser.parse_args()




all_organs=['aorta','liver','kidneys','spleen','pancreas','postcava','stomach','gall_bladder']

# Extract the organ from the path
path = args.path
if path[-1] != '/':
    path += '/'

if args.organ_list == 'auto':
    organs = [os.path.basename(os.path.normpath(path))]
    if organs[0] not in all_organs:
        organs = [file for file in os.listdir(path) if (('right' not in file) and ('left' not in file))]
elif args.organ_list == 'all':
    organs = [file for file in os.listdir(path) if (('right' not in file) and ('left' not in file))]
else:
    organs = re.findall(r'\w+', args.organ_list)


# Extract the organ from the path
path = args.path
if path[-1] != '/':
    path += '/'
#organ = os.path.basename(os.path.normpath(path))

base_url = 'http://0.0.0.0:8000/v1'.replace('8000', args.port)

for organ in organs:
    print('PROCESSING ORGAN: ', organ)
    # Call the function with the extracted organ and provided path
    ed.SystematicComparisonLMDeploySepFigures(
        pth=os.path.join(path,organ),
        size=512,
        organ=organ,
        dice_check=True,
        save_memory=True,
        solid_overlay='auto',
        multi_image_prompt_2='auto',
        dual_confirmation=True,
        conservative_dual=False,
        dice_th=float(args.dice_th),
        base_url=base_url,
        csv_file=args.csv_path,
        restart=(not args.continuing)
    )