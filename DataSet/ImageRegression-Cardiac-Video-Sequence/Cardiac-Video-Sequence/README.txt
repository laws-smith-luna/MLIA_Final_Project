################################

Due to privacy restrictions on sharing the original CMR images, we can only provide the segmented contours rather than the original images.

The task (a) on "First train a segmentation model [U-Net: Convolutional Networks for Biomedical
Image Segmentation, https://arxiv.org/pdf/1505.04597] to segment heart contours from all
Images" is no longer needed. 

################################



#Use the code below to load the 2DxT video sequences of images and associated ground truth label TOS for the image regression task

import numpy as np
data = np.load('2023-11-15-cine-myo-masks-and-TOS.npy', allow_pickle=True)

mask_volume_0 = data[0]['cine_lv_myo_masks_cropped'] # Get the myocardium mask of slice 0. It should be a (H, W, n_frames) volume
TOS_0 = data[0]['TOS'] # Get the TOS curve of slice 0. It should be a (126, ) 1D array
