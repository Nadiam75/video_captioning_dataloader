# video_captioning_dataloader

## requirements
- cv2
- csv
- numpy
- pandas
- pytorch
- sklearn



## data

5 videos of varying sizes are uploaded in the repository

## usage

by changing DISPLAY_FRAMES and SAVE_FRAMES into True or False you can chooses to save the sampled frames in a folder or to display them.



## The weaknesses:


### 1. data augmentation is not implemented

**Solution:**

[This library provides a useful tool for augmenting videos](https://github.com/okankop/vidaug)


### 2. The input resolution must be constant.

**Solution:**

this can be solved by Albumentations library using which we can resize every input frame into a constant Height and Width and perform various kind of transforms on the sampled video frames.



## contact:
nadia.meskar@yahoo.com
zahra_meskar@ee.sharif.edu
