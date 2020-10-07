[//]: # (Image References)

[image1]: ./images/key_pts_example.png "Facial Keypoint Detection"

# Facial keypoints detection using CNN and Pytorch

## Project Overview

In this project, I combine my knowledge of computer vision techniques and deep learning architectures to build a facial keypoint detection system. Nowadays, facial keypoints detection has become a very popular topic and its applications include 
- Snapchat, 
- Facial tracking, 
- Facial pose recognition, 
- Facial filters, and 
- Emotion recognition. 

The objective of facial keypoints detection is to find the facial keypoints in a given face, which is very challenging due to very different facial features from person to person. I modified **NaimishNet architecture** to obtain a CNN model this is able to look at any image, detect faces, and predict the locations of facial keypoints on each face.

The Notebook I submitted for revision to Udacity's reviewers are two Python notebooks. The third one is a fun project.

__Notebook 2__ : Defining and Training a Convolutional Neural Network (CNN) to Predict Facial Keypoints

__Notebook 3__ : Facial Keypoint Detection Using Haar Cascades and your Trained CNN

__Notebook 4__ : Fun Filters and Keypoint Uses


### Image Augmentation
Introduce generalization(randomness) to detect and learn structures


### Transform Steps
I used the following transformation techniques to avoid unnecessary impacts on keypoints
- **Rescale** to 250 for width and height
- **RandomCrop** to 224
- **Normalize & Convert ToTensor**


### Architecture & Performance
I modified the NaimishNet (original)[https://arxiv.org/pdf/1710.00977.pdf] architecture. 

|Conv Layer      | Number of Filters | Filter Shape     | Stride     | Padding     |
| :---           |    :----:         |          :---:   | :---:      |---:         |
|1               | 32                |        5 x 5     |1	     |0 	   |
|2               | 64                |        5 x 5     |1	     |0	           |
|3               | 128               |        3 x 3     |1	     |0	           |
|4               | 256               |        3 x 3     |1	     |0	           |

Activation : ReLU
MaxPooling2d1 : Use a pool shape of (2 x 2) 4x

My modified NaimishNet has four **Conv2D** layers (as shown in the above table), **MaxPooling**, **BatchNorm** in every layer, **ReLU** Activation, and three **fully-connected** (FC) layers. I used **Dropout** at a rate of 0.5 on the the first FC layer and tuned it down to 0.4 on the second FC layer.


BatchSize, Epochs, Loss & Optimization Functions (Ran it on **1 CPU**)
- **BatchSize** : 32 for train dataset and 20 for test dataset
- **Epochs**   : 7 (can train longer for better performance)
- **Loss**     : SmoothL1Loss (significant loss reduction since earlier epochs)
- **Optimizer** : Adam as suggested in the original article
	

### Worklist
For project instructions, please refer to https://github.com/udacity/P1_Facial_Keypoints


LICENSE: This project is licensed under the terms of the MIT license.
