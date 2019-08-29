import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

import torchvision
import resnet


class DeepMAR_res34(nn.Module):
    def __init__(self, classNum):
        super(DeepMAR_res34, self).__init__()

        self.base = resnet.resnet34(pretrained=True)
        self.num_att = classNum

        #print ((self.base))
        #exit()
        
        self.classifier = nn.Linear(512, self.num_att)
        init.normal(self.classifier.weight, std=0.001)
        init.constant(self.classifier.bias, 0)

    def forward(self, x):
        x = self.base(x)
        x = F.avg_pool2d(x, x.shape[2:])
        x = x.view(x.size(0), -1)

        #if self.drop_pool5:
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.classifier(x)
        return x