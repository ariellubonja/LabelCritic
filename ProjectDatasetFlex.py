import os
import argparse
import ast
import importlib
import json
import AnnotationVLM.projection as pj
importlib.reload(pj)

def parse_organs_arg(organs_str):
    # Check if the string starts with '[' and ends with ']'
    if organs_str.startswith('[') and organs_str.endswith(']'):
        # Remove the square brackets and split the string by commas
        organs_list = organs_str[1:-1].split(',')
        # Strip any extra spaces from the organ names
        organs_list = [organ.strip() for organ in organs_list]
        return organs_list
    # Return as a single element list if it's not a list format
    return None


def main():
    parser = argparse.ArgumentParser(description='Process paths for data projection and composition.')

    # Arguments with defaults
    parser.add_argument('--good_folder', default='/mnt/sdc/pedro/ErrorDetection/revised_cropped/',
                        help='Path to the good samples directory (revised_cropped_projection).')
    parser.add_argument('--bad_folder', default='/mnt/sdc/pedro/ErrorDetection/cropped_nnunet_results_250Epch/',
                        help='Path to the first bad samples directory.')
    parser.add_argument('--bad_folder2', default='none',#/mnt/sdc/pedro/ErrorDetection/beta_labels_projection/',
                        help='Path to the second bad samples directory, or "none" if not used.')
    parser.add_argument('--output_dir1', default='compose_nnUnet_ProGT',
                        help='Output directory for the first composite dataset.')
    parser.add_argument('--output_dir2', default='compose_BetaGT_ProGT',
                        help='Output directory for the second composite dataset (ignored if bad_folder2 is "none").')
    parser.add_argument('--organ', default='none')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--num_processes', default='10')
    parser.add_argument('--file_list', default=None)
    parser.add_argument('--restart', action='store_true',default=False)

    args = parser.parse_args()

    good_folder = args.good_folder
    bad_folder = args.bad_folder
    bad_folder2 = args.bad_folder2
    output_dir1 = args.output_dir1
    output_dir2 = args.output_dir2

    

    #print(os.listdir(bad_folder))
    
    organs=parse_organs_arg(args.organ)
    if organs is None:
        if args.organ=='kidneys':
            organs=['kidney_left','kidney_right']
        elif '00' in os.listdir(bad_folder)[0]:
            if args.organ=='none':
                organs=[item[:item.rfind('.nii.gz')] for item in os.listdir(bad_folder+'/'+os.listdir(bad_folder)[0]+'/segmentations')]
            else:
                organs=[args.organ]
        else:
            organs=os.listdir(bad_folder)

    print('Organs:',organs)

    if bad_folder2.lower() != 'none':
        if '00' not in os.listdir(bad_folder2)[0]:
            if args.organ=='none':
                organs2=[item[:item.rfind('.nii.gz')] for item in os.listdir(bad_folder2+'/'+os.listdir(bad_folder2)[0]+'/segmentations')]
            else:
                organs2=[args.organ]
        else:
            organs2=[args.organ]

    order=['spleen','stomach','pancreas','gall_bladder']
    tmp=[]
    for organ in order:
        if organ in organs:
            tmp.append(organ)
    for organ in organs:
        if organ not in order:
            tmp.append(organ)
    organs=tmp

    print(organs)
    #raise ValueError

    if args.file_list is not None:
        with open(args.file_list, 'r') as file:
            file_list_loaded = json.load(file)
        #print(file_list)
        #raise ValueError
        if bad_folder2.lower() != 'none':
            raise ValueError('File list not supported for multiple bad folders.')
        

    file_list={}
    file_list2={}
    for organ in organs:
        if '00' in os.listdir(bad_folder)[0]:
            file_list[organ]=os.listdir(bad_folder)
        else:
            file_list[organ]=os.listdir(os.path.join(bad_folder,organ))
        if bad_folder2.lower() != 'none':
            if '00' in os.listdir(bad_folder)[0]:
                file_list2[organ]=os.listdir(bad_folder2)
            else:
                file_list2[organ]=os.listdir(os.path.join(bad_folder2,organ))


    #get intersection between file list and file_list_loaded
    if args.file_list is not None:
        for organ in organs:
            file_list[organ]=list(set(file_list[organ])&set(file_list_loaded[organ]))
            
    for organ in organs:
        if 'right' in organ:
            file_list[organ]=list(set(file_list[organ]+file_list[organ.replace('right','left')]))
            if bad_folder2.lower() != 'none':
                file_list2[organ]=list(set(file_list2[organ]+file_list2[organ.replace('right','left')]))
        if 'left' in organ:
            file_list[organ]=list(set(file_list[organ]+file_list[organ.replace('left','right')]))
            if bad_folder2.lower() != 'none':
                file_list2[organ]=list(set(file_list2[organ]+file_list2[organ.replace('left','right')]))
    


    # Define projection paths
    good_projection_path = good_folder.rstrip('/') + '_projection'
    if 'T9' in good_projection_path:
        good_projection_path ='./'+good_projection_path[good_projection_path.rfind('T9/')+3:]
    print(good_projection_path)
    bad_projection_path = bad_folder.rstrip('/') + '_projection'
    if bad_folder2.lower() != 'none':
        bad2_projection_path = bad_folder2.rstrip('/') + '_projection'

    print(file_list)

    # Ensure projection directories exist
    os.makedirs(good_projection_path, exist_ok=True)
    os.makedirs(bad_projection_path, exist_ok=True)
    if bad_folder2.lower() != 'none':
        os.makedirs(bad2_projection_path, exist_ok=True)

    # Project files
    for organ in organs:
        if 'all_classes' in organ:
            continue
        #print(file_list[organ])
        if organ != 'none':
            if '00' not in os.listdir(bad_folder)[0]:
                src = os.path.join(good_folder, organ)
            else:
                src = good_folder
            destination = os.path.join(good_projection_path, organ)
        else:
            src = good_folder
            destination = good_projection_path
        print(src, destination)
        pj.project_files(
            pth=src,
            destin=destination,
            file_list=file_list[organ],
            organ=organ,
            device=args.device,
            num_processes=int(args.num_processes),
            skip_existing=(not args.restart)
        )
        if organ != 'none':
            if '00' not in os.listdir(bad_folder)[0]:
                src = os.path.join(bad_folder, organ)
            else:
                src = bad_folder
            destination = os.path.join(bad_projection_path, organ)
        else:
            src = bad_folder
            destination = bad_projection_path
        pj.project_files(
            pth=src,
            destin=destination,
            file_list=file_list[organ],
            organ=organ,
            device=args.device,
            num_processes=int(args.num_processes),
            skip_existing=(not args.restart)
        )
        if bad_folder2.lower() != 'none' and organ in list(bad_old['DSC<=0.5'].keys()):
            if organ != 'none':
                if '00' not in os.listdir(bad_folder)[0]:
                    src = os.path.join(bad_folder2, organ)
                else:
                    src = bad_folder2
                destination = os.path.join(bad2_projection_path, organ)
            else:
                src = bad_folder2
                destination = bad2_projection_path
            pj.project_files(
                pth=src,
                destin=destination,
                file_list=file_list2[organ],
                organ=organ,
                device=args.device,
            num_processes=int(args.num_processes),
            skip_existing=(not args.restart)
            )

        
        # Composite datasets
        pj.composite_dataset(
            output_dir=output_dir1,
            good_path=good_projection_path,
            bad_path=bad_projection_path,
            organ=organ
        )
        if bad_folder2.lower() != 'none':
            pj.composite_dataset(
                output_dir=output_dir2,
                good_path=good_projection_path,
                bad_path=bad2_projection_path,
            organ=organ
            )

        # Join left and right datasets
    for organ in organs:
        if 'left' in organ:
            pj.join_left_and_right_dataset(
                os.path.join(output_dir1, organ),
                os.path.join(output_dir1, organ.replace('left', 'right')),
                os.path.join(output_dir1, organ.replace('_left', 's'))
            )
            if bad_folder2.lower() != 'none' and organ in list(bad_old['DSC<=0.5'].keys()):
                pj.join_left_and_right_dataset(
                    os.path.join(output_dir2, organ),
                    os.path.join(output_dir2, organ.replace('left', 'right')),
                    os.path.join(output_dir2, organ.replace('_left', 's'))
                )

if __name__ == '__main__':
    main()