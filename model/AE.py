import numpy as np
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from .Memory import *

class cnn_layer(nn.Module):
    def __init__(self, nin, nout):
        super(cnn_layer, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(nin, nout, 3, 1, 1),
            nn.BatchNorm2d(nout),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        return self.main(x)

class Encoder(torch.nn.Module):
    def __init__(self, in_channel, dim):
        super(Encoder, self).__init__()
        self.block1 = cnn_layer(in_channel, 64)
        self.block2 = cnn_layer(64, 128)
        self.block3 = cnn_layer(128, 128)
        self.block4 = cnn_layer(128, 256)
        self.block5 = cnn_layer(256, 256)
        self.block6 = cnn_layer(256, 512)
        self.block7 = nn.Sequential(
            nn.Conv2d(512, dim, kernel_size=4),
            nn.BatchNorm2d(dim),
            nn.Tanh(),
        )

        self.mp = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(self.mp(x))
        x = self.block3(self.mp(x))
        x = self.block4(self.mp(x))
        x = self.block5(self.mp(x))
        x = self.block6(self.mp(x))
        x = self.block7(self.mp(x))
        return x

    
    
class Decoder(torch.nn.Module):
    def __init__(self, out_channel, dim):
        super(Decoder, self).__init__()
        self.block1 = nn.Sequential(
            nn.ConvTranspose2d(dim, 512, kernel_size=4),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
        )
        self.block2 = cnn_layer(512, 256)
        self.block3 = cnn_layer(256, 256)
        self.block4 = cnn_layer(256, 128)
        self.block5 = cnn_layer(128, 128)
        self.block6 = cnn_layer(128, 64)
        self.block7 = nn.Sequential(
            nn.ConvTranspose2d(64, out_channel, kernel_size=3, stride=1, padding=1),
            nn.Tanh(),
        )

        self.up = nn.UpsamplingNearest2d(scale_factor=2)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(self.up(x))
        x = self.block3(self.up(x))
        x = self.block4(self.up(x))
        x = self.block5(self.up(x))
        x = self.block6(self.up(x))
        x = self.block7(self.up(x))
        return x
    


class convAE(torch.nn.Module):
    def __init__(self, channel=3, dim=32):
        super(convAE, self).__init__()

        self.encoder = Encoder(in_channel=channel, dim=dim)
        self.decoder = Decoder(out_channel=channel, dim=dim)
        

    def forward(self, x,train=True):

        
        if train:
            fea = self.encoder(x)
            output = self.decoder(fea)
            
            return output
        #test
        else:
            fea = self.encoder(x)
            output = self.decoder(fea)
            
            return output
        
                                          



    