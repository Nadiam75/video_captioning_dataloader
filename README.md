# video_captioning_dataloader

According to [PyTorch Documentation](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html),
the code for processing data samples can get messy and hard to maintain; we ideally want our dataset code to be decoupled from our model training code for better readability and modularity. PyTorch provides two data primitives: torch.utils.data.DataLoader and torch.utils.data.Dataset that allow you to use pre-loaded datasets as well as your own data. Dataset stores the samples and their corresponding labels, and DataLoader wraps an iterable around the Dataset to enable easy access to the samples.
Also, we can benefit from PyTorch domain libraries which provide a number of pre-loaded datasets (such as FashionMNIST) that subclass torch.utils.data.Dataset and implement functions specific to the particular data. They can be used to prototype and benchmark your model.

## requirements
- cv2
- csv
- numpy
- pandas
- pytorch
- sklearn



## data

a .csv file contains the path to 5 videos and their corresponding caption.
by iterating through this file and  each video is loaded.
at the next stage, a dataframe is created in which each row contains an array of size (N*H*W*C) and a caption.
A class named 'MovieCaptioningDataset' is created which inherits its properties from torch.utils.data's Dataset class.
An instance of this class is generated for the training and validation data containing the model's input and output.
## usage

by changing DISPLAY_FRAMES and SAVE_FRAMES into True or False you can chooses to save the sampled frames in a folder or to display them.

Also to run the .py file :
``` shell
python dataloader.py
```

The .ipynb version has also been uploaded in the repository.

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
