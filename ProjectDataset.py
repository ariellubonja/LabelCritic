import AnnotationVLM.projection as pj
import importlib
importlib.reload(pj)

import ast

with open('ErrorLists/bad_nnUNet250Epochs.txt', 'r') as file:
    data = file.read()
bad_nn = ast.literal_eval(data)

with open('ErrorLists/bad_old_labels.txt', 'r') as file:
    data = file.read()
bad_old = ast.literal_eval(data)

names_nn={}
organs=list(bad_nn['DSC<=0.5'].keys())
for organ in organs:
    names_nn[organ]=[]
for key in bad_nn:
    for organ in organs:
        names_nn[organ]+=bad_nn[key][organ]
        
names_old={}
for organ in organs:
    names_old[organ]=[]
for key in bad_old:
    for organ in organs:
        if organ in bad_old[key]:
            names_old[organ]+=bad_old[key][organ]

# Get unique patient IDs
ground_truth={}
for organ in organs:
    ground_truth[organ]=set(names_old[organ]+names_nn[organ])

for organ in ground_truth:
    #unio of left and right
    if 'right' in organ:
        ground_truth[organ]=ground_truth[organ] | ground_truth[organ.replace('right','left')]
    if 'left' in organ:
        ground_truth[organ]=ground_truth[organ] | ground_truth[organ.replace('left','right')]

for organ in organs:
    pj.project_files(pth='/mnt/sdc/pedro/ErrorDetection/revised_cropped/',destin='/mnt/sdc/pedro/ErrorDetection/revised_cropped_projection/'+organ,
                    file_list=ground_truth[organ], organ=organ)
    pj.project_files(pth='/mnt/sdc/pedro/ErrorDetection/cropped_nnunet_results_250Epch/',destin='/mnt/sdc/pedro/ErrorDetection/nnUNet_results_250Epochs_cropped_projection/'+organ,
                 file_list=ground_truth[organ], organ=organ)
    if organ in list(bad_old['DSC<=0.5'].keys()):
        pj.project_files(pth='../AbdomenAtlasBeta/',destin='/mnt/sdc/pedro/ErrorDetection/beta_labels_projection/'+organ,
                    file_list=ground_truth[organ], organ=organ)
    
pj.composite_dataset(output_dir='compose_nnUnet_ProGT', good_path='/mnt/sdc/pedro/ErrorDetection/revised_cropped_projection/',
                     bad_path='/mnt/sdc/pedro/ErrorDetection/nnUNet_results_250Epochs_cropped_projection/',)
pj.composite_dataset(output_dir='compose_BetaGT_ProGT', good_path='/mnt/sdc/pedro/ErrorDetection/revised_cropped_projection/',
                     bad_path='/mnt/sdc/pedro/ErrorDetection/beta_labels_projection/')

for organ in organs:
    if 'left' in organ:
        if organ in list(bad_old['DSC<=0.5'].keys()):
            pj.join_left_and_right_dataset('compose_BetaGT_ProGT/'+organ+'/',
                                        'compose_BetaGT_ProGT/'+organ.replace('left','right')+'/',
                                        'compose_BetaGT_ProGT/'+organ.replace('_left','s')+'/',)
        pj.join_left_and_right_dataset('compose_nnUnet_ProGT/'+organ+'/',
                                       'compose_nnUnet_ProGT/'+organ.replace('left','right')+'/',
                                       'compose_nnUnet_ProGT/'+organ.replace('_left','s')+'/',)