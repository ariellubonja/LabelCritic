import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import cv2
import random 
import scipy.ndimage as ndimage
import ast
import torch
from PIL import Image

def plot_organ_projection(list_of_array, organ_name, pid, axis=2,
                           pngpath=None, th=0.5, ct=False, save=True,
                           window='organs'):
    if axis == 2:
        projection = np.zeros((list_of_array[0][:,:,0].shape), dtype='float')
    elif axis == 1:
        projection = np.zeros((list_of_array[0][:,0,:].shape), dtype='float')
    elif axis == 0:
        projection = np.zeros((list_of_array[0][0,:,:].shape), dtype='float')
    else:
        raise ValueError('Axis should be 0, 1, or 2')

    for i in range(len(list_of_array)):
        x=list_of_array[i]
        if ct:
            if window=='organs':
                x=np.where(x>250,250,x)
                x=np.where(x<-150,-150,x)
                x=x+150
                x=x/400
            elif window=='bone':
                x=np.where(x>1500,1500,x)
                x=np.where(x<-500,-500,x)
                x=x+500
                x=x/2000
            else:
                raise ValueError('Window should be organs or bone')
        organ_projection = np.sum(x, axis=axis) * 1.0
        organ_projection /= np.max(organ_projection)
        projection += organ_projection
    
    projection /= np.max(projection)

    if th>0:
        projection=np.where(projection>0,projection/(1/(1-th))+th,0)

    projection *= 255.0
    projection = np.rot90(projection)

    if save:
        if not os.path.exists(pngpath):
            os.makedirs(pngpath)
        cv2.imwrite(os.path.join(pngpath, pid + '_axis_'+str(axis)+'.png'), projection)

        print('Organ projection of ' + organ_name + ' for patient ' + pid + ' is saved to ' + os.path.join(pngpath, pid + '.png'))

    return projection

def plot_organ_projection_3_axis(list_of_array, organ_name, pid,
                                  pngpath=None, th=0.5, ct=False, save=True,
                                  window='organs'):
    projections=[]
    for axis in range(3):
        projections.append(plot_organ_projection(list_of_array, organ_name, pid, axis, 
                                                 pngpath, th=th, ct=ct,save=save,
                                                 window=window))
    return projections


def get_orientation_transform(nii):
    """
    Compute the transformation needed to reorient the image to LAS standard orientation.
    """
    current_orientation = nib.orientations.io_orientation(nii.affine)
    standard_orientation = nib.orientations.axcodes2ornt(('L', 'A', 'S'))
    transform = nib.orientations.ornt_transform(current_orientation, standard_orientation)
    return transform

def apply_transform(data, transform):
    """
    Apply the orientation transformation to the image data.
    """
    return nib.orientations.apply_orientation(data, transform)

def resample_image(image, original_spacing, target_spacing=(1, 1, 1),order=1):
    """
    Resample the image to the target spacing.

    Parameters:
    image (nibabel.Nifti1Image): Input image to resample.
    target_spacing (tuple): Target spacing in x, y, z directions.

    Returns:
    numpy.ndarray: Resampled image data.
    """
    # Get original spacing
    resize_factor = np.array(original_spacing) / np.array(target_spacing)
    new_shape = np.round(image.shape * resize_factor).astype(int)

    # Resample image
    try:image=image.get_fdata()
    except:pass
    resampled_image = ndimage.zoom(image, resize_factor, order=order)

    return resampled_image   

import torch
import torch.nn.functional as F

def resample_image_cuda(image, original_spacing, target_spacing=(1, 1, 1), order=1):
    """
    Resample the image to the target spacing using GPU acceleration with PyTorch.

    Parameters:
    image (numpy.ndarray or nibabel.Nifti1Image): Input image to resample.
    original_spacing (tuple or list): Original spacing in x, y, z directions.
    target_spacing (tuple): Target spacing in x, y, z directions.
    order (int): Interpolation order (0=nearest, 1=linear)

    Returns:
    numpy.ndarray: Resampled image data.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Get image data
    try:
        image = image.get_fdata()
    except AttributeError:
        pass  # Image is already a NumPy array

    # Convert image to torch tensor and move to device
    image = torch.from_numpy(image).float().to(device)

    # Add batch and channel dimensions: (N, C, D, H, W)
    image = image.unsqueeze(0).unsqueeze(0)

    # Reverse the spacings to match the image axes (D, H, W)
    resize_factor = np.array(original_spacing)[::-1] / np.array(target_spacing)[::-1]

    # Compute new shape
    new_shape = np.round(np.array(image.shape[2:]) * resize_factor).astype(int)

    # Choose interpolation mode based on order
    if order == 0:
        mode = 'nearest'
    elif order == 1:
        mode = 'trilinear'
    else:
        raise ValueError('Order must be 0 (nearest) or 1 (linear)')

    # Perform interpolation
    if mode == 'nearest':
        resampled_image = F.interpolate(
            image,
            size=new_shape.tolist(),
            mode=mode,
            recompute_scale_factor=False
        )
    else:
        resampled_image = F.interpolate(
            image,
            size=new_shape.tolist(),
            mode=mode,
            align_corners=False,
            recompute_scale_factor=False
        )

    # Remove batch and channel dimensions
    resampled_image = resampled_image.squeeze(0).squeeze(0)

    # Convert back to numpy array
    resampled_image = resampled_image.cpu().numpy()

    return resampled_image

def load_ct_and_mask(pid, organ, datapath,
                     ct_path=None, mask_path=None):
    """
    Load and reorient the CT scan and its corresponding mask to the standard RAS orientation.

    Parameters:
        pid (str): Patient ID.
        organ (str): Name of the organ for the mask file.
        datapath (str): Path to the dataset.
        ct_path (str): Path to the CT scan.
        mask_path (str): Path to the mask file.

    Returns:
        ct (np.ndarray): Reoriented CT scan data.
        mask (np.ndarray): Reoriented mask data.
    """
    # Load the CT scan
    if ct_path is None:
        ct_path = os.path.join(datapath, pid, 'ct.nii.gz')
    ct_nii = nib.load(ct_path)
    spacing=ct_nii.header.get_zooms()
    resampled = resample_image(ct_nii, spacing, target_spacing=(1, 1, 1),order=1)
    ct_nii = nib.Nifti1Image(resampled, affine=ct_nii.affine, header=ct_nii.header)

    # Calculate the orientation transformation based on the CT scan
    transform = get_orientation_transform(ct_nii)

    # Apply the transformation to the CT scan data
    ct = apply_transform(ct_nii.get_fdata(), transform)

    # Load the mask using the same transformation
    if mask_path is None:
        mask_path = os.path.join(datapath, pid, 'segmentations', organ + '.nii.gz')
    mask_nii = nib.load(mask_path)
    resampled = resample_image(mask_nii, spacing, target_spacing=(1, 1, 1),order=0)
    if not np.array_equal(resampled, resampled.astype(bool)):
        resampled = np.where(resampled > 0.5, 1, 0)
    mask_nii = nib.Nifti1Image(resampled, affine=mask_nii.affine, header=mask_nii.header)

    # Apply the same transformation to the mask data
    mask = apply_transform(mask_nii.get_fdata(), transform).astype(np.uint8)

    return ct, mask



def overlay_projection(pid, organ, datapath,save_path,th=0.5,
                       mask_only=False,ct_only=False,
                       clahe=False,he=False,window='organs',
                       ct_path=None, mask_path=None):

    ct, mask=load_ct_and_mask(pid, organ, datapath,
                              ct_path=ct_path, mask_path=mask_path)

    ct_projections=plot_organ_projection_3_axis([ct], organ, pid, th=0,
                                                 ct=True, save=False,
                                                 window=window)
    if clahe or he:
        for i in range(len(ct_projections)):
            img=ct_projections[i]
            flag=False
            #print('max img:',np.max(img))
            #print('min img:',np.min(img))
            if np.max(img)>1:
                img=img/255
                flag=True
            if clahe:
                ct_projections[i] = exposure.equalize_adapthist(img, clip_limit=0.01, 
                                                                nbins=256, 
                                                                kernel_size=(16,16))
            if he:
                ct_projections[i] = exposure.equalize_hist(img)

            if flag:
                ct_projections[i]=ct_projections[i]*255

    mask_projections=plot_organ_projection_3_axis([mask], organ, pid, th=th,
                                                   ct=False, save=False)

    if mask_only:
        ct_projections=[0*proj for proj in ct_projections]
    if ct_only:
        mask_projections=[0*proj for proj in mask_projections]


    for i in range(len(ct_projections)):
        ct_proj=ct_projections[i]
        mask_proj=mask_projections[i]

        print('max ct proj:',np.max(ct_proj))
        print('min ct proj:',np.min(ct_proj))  
        print('max mask proj:',np.max(mask_proj))
        print('min mask proj:',np.min(mask_proj))    

        ct_proj = np.expand_dims(ct_proj, axis=-1)
        ct_proj = np.tile(ct_proj, (1, 1, 3))

        if not mask_only:
            overlay = ct_proj
            ct_proj[:,:,0] = ct_proj[:,:,0]-mask_proj
            ct_proj[:,:,1] = ct_proj[:,:,1]-mask_proj
        else:
            overlay = np.expand_dims(mask_proj, axis=-1)
            overlay = np.tile(overlay, (1, 1, 3))
            overlay[:,:,0] = 0
            overlay[:,:,1] = 0
            

        
        #overlay = overlay.astype(np.uint8)

        print('overlay shape:',overlay.shape)
        print('max overlay:',np.max(overlay))
        print('min overlay:',np.min(overlay))

        print('shape of overlay:',overlay.shape)

        if not os.path.exists(save_path):
            os.makedirs(save_path)
        name='_overlay'
        if mask_only:
            name='_mask'
        if ct_only:
            name='_ct'
        cv2.imwrite(os.path.join(save_path, pid + name + '_axis_'+str(i)+'.png'), 
                    overlay)

        print('Organ projection of ' + save_path + ' for patient ' + pid + \
            ' is saved to ' + os.path.join(save_path, pid + '.png'))
        

def plot_organ_projection_cuda(list_of_array, organ_name, pid, axis=2,
                               pngpath=None, th=0.5, ct=False, save=True,
                               window='organs'):
    # Stack tensors along a new dimension
    stacked_x = torch.stack(list_of_array, dim=0)  # Shape: (N, D1, D2, D3)

    # Apply windowing if ct is True
    if ct:
        if window == 'organs':
            upper_limit = 250.0
            lower_limit = -150.0
            offset = 150.0
            divisor = 400.0
        elif window == 'bone':
            upper_limit = 1500.0
            lower_limit = -500.0
            offset = 500.0
            divisor = 2000.0
        else:
            raise ValueError('Window should be "organs" or "bone"')

        # Apply the windowing to the stacked tensor
        stacked_x = torch.clamp(stacked_x, min=lower_limit, max=upper_limit)
        stacked_x = (stacked_x + offset) / divisor

    # Sum over the specified axis (adjusted for batch dimension)
    organ_projection = torch.sum(stacked_x, dim=axis+1)

    # Normalize each projection in the batch
    max_vals = organ_projection.view(organ_projection.size(0), -1).max(dim=1)[0] + 1e-8
    organ_projection = organ_projection / max_vals[:, None, None]

    # Sum the normalized projections across the batch dimension
    projection = torch.sum(organ_projection, dim=0)

    # Normalize the final projection
    projection = projection / (torch.max(projection) + 1e-8)

    # Apply threshold if specified
    if th > 0:
        projection = torch.where(projection > 0,
                                 projection / (1 / (1 - th)) + th,
                                 torch.tensor(0.0).type_as(projection))

    # Scale to 255 for image representation
    projection *= 255.0

    # Rotate the projection by 90 degrees counter-clockwise
    projection = torch.rot90(projection, k=1, dims=(0, 1))

    # Save the projection if required
    if save:
        projection_np = projection.detach().cpu().numpy()
        if pngpath is not None and not os.path.exists(pngpath):
            os.makedirs(pngpath)
        filename = f"{pid}_axis_{axis}.png"
        filepath = os.path.join(pngpath, filename) if pngpath else filename
        cv2.imwrite(filepath, projection_np)
        print(f'Organ projection of {organ_name} for patient {pid} is saved to {filepath}')

    return projection



def overlay_projection_fast(pid, organ, datapath, save_path, th=0.5,
                            mask_only=False, ct_only=False, window='organs',
                            ct_path=None, mask_path=None, axis=1, device='cuda:0',
                            precision=32):
    """
    Generate and save overlay projections of CT and mask images.

    Parameters:
    - pid (str): Patient ID.
    - organ (str): Organ name.
    - datapath (str): Path to the data directory.
    - save_path (str): Directory to save the overlay images.
    - th (float): Threshold for mask projection.
    - mask_only (bool): If True, only the mask is projected.
    - ct_only (bool): If True, only the CT is projected.
    - window (str): Windowing option for CT images ('organs' or 'bone').
    - ct_path (str): Path to the CT image file.
    - mask_path (str): Path to the mask image file.
    - axis (int): Axis along which to project (0, 1, or 2).
    - device (str): Device to run the computations on ('cuda:0' or 'cpu').

    Returns:
    - None
    """
    # Load CT and mask images
    ct, mask = load_ct_and_mask(pid, organ, datapath,
                                ct_path=ct_path, mask_path=mask_path)

    # Convert to torch tensors and move to device
    # Ensure arrays have positive strides by making a copy
    ct = ct.copy()
    mask = mask.copy()

    ct = torch.from_numpy(ct).to(device).float()
    mask = torch.from_numpy(mask).to(device).float()
    

    # Generate projections
    if not mask_only:
        ct_projection = plot_organ_projection_cuda([ct], organ, pid, axis=axis,
                                                   th=0, ct=True, save=False,
                                                   window=window)
    if not ct_only:
        mask_projection = plot_organ_projection_cuda([mask], organ, pid, axis=axis,
                                                     th=th, ct=False, save=False)

    # Handle cases where only CT or only mask is desired
    if mask_only:
        # Create a zero projection for CT
        ct_projection = torch.zeros_like(mask_projection, device=device)
    if ct_only:
        # Create a zero projection for mask
        mask_projection = torch.zeros_like(ct_projection, device=device)

    # Ensure projections are on the same device and have the same shape
    ct_projection = ct_projection.to(device)
    mask_projection = mask_projection.to(device)

    # Print max and min values for debugging
    #print('max ct proj:', ct_projection.max().item())
    #print('min ct proj:', ct_projection.min().item())
    #print('max mask proj:', mask_projection.max().item())
    #print('min mask proj:', mask_projection.min().item())

    # Prepare overlay
    if mask_only:
        # Overlay only the mask in the blue channel
        overlay = torch.stack([
            torch.zeros_like(mask_projection),  # R channel
            torch.zeros_like(mask_projection),  # G channel
            mask_projection  # B channel
        ], dim=2)
    elif ct_only:
        # Overlay only the CT in all channels
        overlay = ct_projection.unsqueeze(-1).repeat(1, 1, 3)
    else:
        # Overlay CT and subtract mask from R and G channels
        overlay = torch.stack([
            ct_projection - mask_projection,  # R channel
            ct_projection - mask_projection,  # G channel
            ct_projection  # B channel
        ], dim=2)

    # Clamp values to [0, 255] and convert to uint8
    overlay = torch.clamp(overlay, 0, 255).byte()

    # Move to CPU and convert to NumPy array
    overlay_np = overlay.cpu().numpy()

    # Print overlay information
    #print('overlay shape:', overlay_np.shape)
    #print('max overlay:', overlay_np.max())
    #print('min overlay:', overlay_np.min())

    # Save the overlay image
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    name = '_overlay'
    if mask_only:
        name = '_mask'
    if ct_only:
        name = '_ct'
    filename = os.path.join(save_path, f"{pid}{name}_axis_{axis}_{organ}.png")
    cv2.imwrite(filename, overlay_np)
    print(f'Organ projection of {organ} for patient {pid} is saved to {filename}')

def create_composite_image(pth, organ, axis=1):
    y1_bone = os.path.join(pth, f'y1_bone_overlay_axis_{axis}_{organ}.png')
    y1_organs = os.path.join(pth, f'y1_organs_overlay_axis_{axis}_{organ}.png')
    y2_bone = os.path.join(pth, f'y2_bone_overlay_axis_{axis}_{organ}.png')
    y2_organs = os.path.join(pth, f'y2_organs_overlay_axis_{axis}_{organ}.png')

    # Load the images
    image_paths = [y1_bone, y1_organs, y2_bone, y2_organs]
    images = [Image.open(path) for path in image_paths]

    # Get image dimensions in pixels
    img_width, img_height = images[0].size

    # DPI (dots per inch)
    dpi = 100  # Adjust as needed

    # Calculate subplot size in inches
    subplot_width = img_width / dpi
    subplot_height = img_height / dpi

    # Title font size in points and inches
    title_font_size_pts = 12  # Default matplotlib font size
    title_font_size_inch = title_font_size_pts / 72  # Convert points to inches

    # Desired spacing between subplots (1.5 times title font size)
    desired_spacing_inch = 1.5 * title_font_size_inch

    # Compute wspace and hspace as fractions of subplot size
    wspace = desired_spacing_inch / subplot_width
    hspace = desired_spacing_inch / subplot_height

    # Total figure size in inches
    fig_width = 2 * subplot_width + desired_spacing_inch
    fig_height = 2 * subplot_height + desired_spacing_inch

    # Create figure and axes
    fig, axes = plt.subplots(2, 2, figsize=(fig_width, fig_height), dpi=dpi)
    axes = axes.flatten()

    # Use your original titles
    titles = ["Image 1", "Image 2", "Image 3", "Image 4"]

    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img)
        ax.set_title(title)
        ax.axis('off')

        # Add 'R' and 'L' labels with some padding
        padding = 10  # Adjust as needed
        ax.text(padding, img_height / 2, 'R', fontsize=title_font_size_pts, ha='left', va='center',
                color='blue', fontweight='bold')
        ax.text(img_width - padding, img_height / 2, 'L', fontsize=title_font_size_pts, ha='right', va='center',
                color='green', fontweight='bold')

    # Adjust subplot spacing
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=wspace, hspace=hspace)

    # Save the figure
    save_path = os.path.join(pth, f'composite_image_axis_{axis}_{organ}.png')
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)

    print(f'Composite image saved to {save_path}')
    plt.close()
    
def create_composite_image_2figs(pth, organ, axis=1):
    y1_bone = os.path.join(pth, f'y1_bone_overlay_axis_{axis}_{organ}.png')
    y2_bone = os.path.join(pth, f'y2_bone_overlay_axis_{axis}_{organ}.png')

    # Load the images
    image_paths = [y1_bone, y2_bone]
    images = [Image.open(path) for path in image_paths]

    # Get image dimensions in pixels
    img_width, img_height = images[0].size

    # DPI (dots per inch)
    dpi = 100  # Adjust as needed

    # Calculate subplot size in inches
    subplot_width = img_width / dpi
    subplot_height = img_height / dpi

    # Title font size in points and inches
    title_font_size_pts = 12  # Default matplotlib font size
    title_font_size_inch = title_font_size_pts / 72  # Convert points to inches

    # Desired spacing between subplots (1.5 times title font size)
    desired_spacing_inch = 1.5 * title_font_size_inch

    # Compute wspace as a fraction of subplot width
    wspace = desired_spacing_inch / subplot_width

    # Total figure size in inches
    fig_width = 2 * subplot_width + desired_spacing_inch
    fig_height = subplot_height + desired_spacing_inch  # Adjust height if needed

    # Create figure and axes
    fig, axes = plt.subplots(1, 2, figsize=(fig_width, fig_height), dpi=dpi)
    axes = axes.flatten()

    # Titles for each subplot
    titles = ["Image 1", "Image 2"]

    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img)
        ax.set_title(title)
        ax.axis('off')

        # Add 'R' and 'L' labels with some padding
        padding = 10  # Adjust as needed
        ax.text(padding, img_height / 2, 'R', fontsize=title_font_size_pts, ha='left', va='center',
                color='blue', fontweight='bold')
        ax.text(img_width - padding, img_height / 2, 'L', fontsize=title_font_size_pts, ha='right', va='center',
                color='green', fontweight='bold')

    # Adjust subplot spacing
    plt.subplots_adjust(left=0, right=1, top=0.95, bottom=0.05, wspace=wspace)

    # Save the figure
    save_path = os.path.join(pth, f'composite_image_2_figs_axis_{axis}_{organ}.png')
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)

    print(f'Composite image saved to {save_path}')
    plt.close()

# to improve speed: project ct just once. Then you project the masks using lower precision and overlay them.