import argparse
import os
import ErrorDetector as ed
import importlib

# Reload the module in case it has been updated
importlib.reload(ed)

# Set up argument parsing
parser = argparse.ArgumentParser(description='Run SystematicComparisonLMDeploySepFigures with the specified path.')
parser.add_argument('--path', help='Path to the annotations')
parser.add_argument('--port', help='VLLM port to use', default='8000')

# Parse the arguments
args = parser.parse_args()

# Extract the organ from the path
path = args.path
if path[-1] != '/':
    path += '/'
organ = os.path.basename(os.path.normpath(path))

base_url = 'http://0.0.0.0:8000/v1'.replace('8000', args.port)

# Call the function with the extracted organ and provided path
ed.SystematicComparisonLMDeploySepFigures(
    pth=path,
    size=512,
    organ=organ,
    dice_check=True,
    save_memory=True,
    solid_overlay='auto',
    multi_image_prompt_2='auto',
    dual_confirmation=True,
    conservative_dual=False,
    dice_th=0.75,
    base_url=base_url
)