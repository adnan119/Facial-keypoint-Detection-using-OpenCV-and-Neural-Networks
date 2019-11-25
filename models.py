## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.conv1_bn = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(2, 2)
        #self.convdrop1 = nn.Dropout(p=0.1)
        #new size of the featuremap = ((224-5)/1 +1)/2 = 220/2 = 110
        
        self.conv2 = nn.Conv2d(32,64,4)
        self.conv2_bn = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        #self.convdrop2 = nn.Dropout(p=0.2)
        #new size = ((110 - 4) + 1)/2 = 107/2 = 53(rounded down)
        
        self.conv3 = nn.Conv2d(64,128,3)
        self.conv3_bn = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2, 2)
        #self.convdrop3 = nn.Dropout(p=0.3)
        #new size = ((53-3)+1)/2 = 51/2 = 25(rounded down)
        
        self.conv4 = nn.Conv2d(128,256,2)
        self.conv4_bn = nn.BatchNorm2d(256)
        self.pool = nn.MaxPool2d(2, 2)
        #self.convdrop4 = nn.Dropout(p=0.4)
        #new size = ((25-2)+1)/2 = 24/2 = 12
        
        self.conv5 = nn.Conv2d(256,512,1)
        self.conv5_bn = nn.BatchNorm2d(512)
        self.pool = nn.MaxPool2d(2, 2)
        #self.convdrop5 = nn.Dropout(p=0.5)
        #new size = ((12-1)+1)/2 = 12/2 = 6
        
        self.fc1 = nn.Linear(512*6*6, 1024)
        self.fc1_bn = nn.BatchNorm1d(1024)
        #self.fc1_drop = nn.Dropout(p=0.1)
        
        self.fc2 = nn.Linear(1024, 256)
        self.fc2_bn = nn.BatchNorm1d(256)
        #self.fc2_drop = nn.Dropout(p=0.4)
        
        self.out = nn.Linear(256,136)
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        x = self.pool(F.relu(self.conv1_bn(self.conv1(x))))
        x = self.pool(F.relu(self.conv2_bn(self.conv2(x))))
        x = self.pool(F.relu(self.conv3_bn(self.conv3(x))))
        x = self.pool(F.relu(self.conv4_bn(self.conv4(x))))
        x = self.pool(F.relu(self.conv5_bn(self.conv5(x))))
        
        x = x.view(x.size(0), -1)
        
        x = F.relu(self.fc1_bn(self.fc1(x)))
        #x = self.fc1_drop(x)
        x = F.relu(self.fc2(x))
        #x = self.fc2_drop(x)
        x = self.out(x)
        # a modified x, having gone through all the layers of your model, should be returned
        return x
