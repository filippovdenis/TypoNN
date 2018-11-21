import numpy as np
import torch
import torch.nn as nn

class  CharCNN(nn.Module):
    
    def __init__(self, num_features):
        super(CharCNN, self).__init__()
        
        self.num_features = num_features
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 256, kernel_size=(self.num_features, 7), stride=1),
            nn.ReLU()
        )

        self.maxpool1 = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3))

        self.conv2 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=(1, 7), stride=1),
            nn.ReLU()
        )
        self.maxpool2 = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3))

        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=(1, 3), stride=1),
            nn.ReLU()
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=(1, 3), stride=1),
            nn.ReLU()
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=(1, 3), stride=1),
            nn.ReLU()
        )

        self.conv6 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=(1, 3), stride=1),
            nn.ReLU()
        )

        self.maxpool6 = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3))

        self.fc1 = nn.Sequential(
            nn.Linear(1536, 256),
            nn.ReLU(),
            nn.Dropout(p=0.5)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(p=0.5)
        )
        self.fc3 =nn.Linear(256, 3)
        self.softmax = nn.LogSoftmax()
            # nn.LogSoftmax()

        # self.inference_log_softmax = InferenceBatchLogSoftmax()

    def forward(self, x):
        debug=True
        x = x.unsqueeze(1)
        if debug:
            print('x.size()', x.size())

        x = self.conv1(x)
        if debug:
            print('x after conv1', x.size())

        x = self.maxpool1(x)
        if debug:
            print('x after maxpool1', x.size())
                     

        x = self.conv2(x)
        if debug:
            print('x after conv2', x.size())



        x = self.conv3(x)
        if debug:
            print('x after conv3', x.size())


        x = self.conv4(x)
        if debug:
            print('x after conv4', x.size())
            
        x = self.maxpool2(x)
        if debug:
            print('x after maxpool2', x.size())            


        x = x.view(x.size(0), -1)
        if debug:
            print('Collapse x:, ', x.size())

        x = self.fc1(x)
        if debug:
            print('FC1: ', x.size())

        x = self.fc2(x)
        if debug:
            print('FC2: ', x.size())

        x = self.fc3(x)
        if debug:
            print('x: ', x.size())

        x = self.softmax(x)
        # x = self.inference_log_softmax(x)

        return x