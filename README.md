# Medical_segmentation
In repository there is the second project made during Advanced Image Processing laboratory classes at the Poznan University of Technology. Main target of project was brain segmentation from sequence T1 type of MRI.
# Requirements
All libraries are installed automatically by colab or by instructions in .ipynb file.
# Dataset
Dataset consisted of scans in medical \*.nii format. I split whole dataset into train and validation sets. From three-dimentional matrix (simple scan) I extracted 2D slices from y axis, rescale them to 256x256 and saved them as images (png format). These images were used as input to network. Loaded images from dataset was normalized.
# Network
I used Unet architecture from segmentation_models with params: 
- backbone='resnet50',
- 'encoder_weights='imagenet',
- input_shape=(256, 256, 3).
# Result
Average Dice score for test dataset: 0.9785.
