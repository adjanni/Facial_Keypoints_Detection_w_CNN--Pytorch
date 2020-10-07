## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        """
        Modified by Axel Tcheheumeni
        Modified NaimishNet has layers below:
        Conv Layer | Number of Filters | Filter Shape
         ---------------------------------------------
        1          |        32         |    (5,5)
        2          |        64         |    (5,5)
        3          |        128        |    (3,3)
        4          |        256        |    (3,3)
        ---------------------------------------------
        Activation : RELU
        MaxPooling2d1 : Use a pool shape of (2,2) 4x
        
        I did not use dropout in each convolutional layer as explained in this article [1]. Instead, I used bacth normalisation between my
        convolution layers to regularise my model and to make the model stable during the training [2][3]. 
        
        In the fully-connected layer, I start with a dropout rate of 0.5 and tune it down to 0.4 on the second FC layer.
        
        [1] https://towardsdatascience.com/dropout-on-convolutional-layers-is-weird-5c6ab14f19b2
        [2] https://mc.ai/intuit-and-implement-batch-normalization/
        [3] https://www.kdnuggets.com/2018/09/dropout-convolutional-networks.html
        
        """
            
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        
        
        ######## Convolutional layers the '#->' means the  size of the output image at each step 
        # output size = (W-F)/S +1
        
        # First conv
        self.conv1 = nn.Conv2d(1, 32, 5) #-> (32, 220, 220)   
        # maxpool layer
        # pool with jernel_size = 2, stride = 2
        self.pool1 = nn.MaxPool2d(2, 2) #-> (32, 110, 110)
        #self.drop1 = nn.Dropout(p = 0.1)
        
        # Second conv
        self.conv2 = nn.Conv2d(32, 64, 5) #-> (64, 106, 106)
        #MPL
        self.pool2 = nn.MaxPool2d(2, 2) #-> (64, 53, 53)
        #self.drop2 = nn.Dropout(p = 0.2)
        
        # Third conv
        self.conv3 = nn.Conv2d(64, 128, 3) #-> (128, 51, 51)
        #MPL 
        self.pool3 = nn.MaxPool2d(2, 2) #-> (128, 25, 25)
        #self.drop3 = nn.Dropout(p = 0.3)
        
        # Fourth conv
        self.conv4 = nn.Conv2d(128, 256, 3) #-> (256, 23, 23)
        #MPL 
        self.pool4 = nn.MaxPool2d(2, 2) #-> (256, 11, 11)
        #self.drop4 = nn.Dropout(p = 0.4)
        
        
        ###### Batch normalisation - to ease the training process of deeper model
        #- For any hidden layers, can we norwalise the value of an activation so as to      train the weight and biase faster...We will normalise the value before applying relu
        self.bn_1 = nn.BatchNorm2d(32)
        self.bn_2 = nn.BatchNorm2d(64)
        self.bn_3 = nn.BatchNorm2d(128)
        self.bn_4 = nn.BatchNorm2d(256)
        
        
        ########  Create three fully-connected linear layers + two dropout to avoid   overfitting
        # The 1st layer will be of size 512*6*6 nodes and will connect to the second layer of 512*6 nodes. This layer will connect to 
        # the last one of 256*6 nodes.
        self.fc1 = nn.Linear(256 * 11 * 11, 1500)        
        self.drop6 = nn.Dropout(0.5)
        
        self.fc2 = nn.Linear(1500, 1500)
        self.drop5 = nn.Dropout(0.4)
        
        self.fc3 = nn.Linear(1500, 68 * 2)
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
      
        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        
        # First - Conv --> MPL --> BN --> Activation
        x = self.pool1(self.conv1(x))
        x = F.relu(self.bn_1(x))
        
        # Second - Conv --> MPL --> BN --> Activation
        x = self.pool2(self.conv2(x))
        x = F.relu(self.bn_2(x))
        
        # Third - Conv --> MPL --> BN --> Activation
        x = self.pool3(self.conv3(x))
        x = F.relu(self.bn_3(x))
        
         # Fourth - Conv --> MPL --> BN --> Activation
        x = self.pool4(self.conv4(x))
        x = F.relu(self.bn_4(x))
        
        
        ## prep for linear layer
        # Flattenen layer
        x = x.view(x.size(0), -1)
        
        ## three linear layers with dropout in between
        x = F.relu(self.fc1(x))
        x = self.drop5(x)
        
        x = F.relu(self.fc2(x))
        x = self.drop6(x)
        
        x = self.fc3(x)        
        
        # a modified x, having gone through all the layers of your model, should be returned
        
        return x
