# Created on 8/15/22 at 3:14 PM 

# Author: Jenny Sun

'''this scrip contains models that fit drift and boundary, single boundary
    model split from the the beggining'''

# Created on 10/12/21 at 10:50 PM

# Author: Jenny Sun
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import math
from models_sinc_spatial import *




# torch.manual_seed(2022)
# np.random.seed(2022)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

class SincDriftBoundAttChoice_full(nn.Module):
    '''predicts drift and boundary
    attention layer after sinc and after conv
    prediction splits before sinc layer'''

    # filter_length = 251
    num_filters = 32
    filter_length = 131
    t_length = 500
    pool_window_ms =250   # set the pool window in time unit
    stride_window_ms = 100   # set the stride window in time unit
    pool_window = int(np.rint(((t_length- filter_length +1) * pool_window_ms)/ (1000)))
    stride_window =  int(np.rint(((t_length- filter_length +1) * stride_window_ms)/ (1000)))
    if pool_window % 2 == 0 :
        pool_window -= 1
    if stride_window % 2 == 0 :
        stride_window -= 1
    output_size = int(np.floor(((t_length- filter_length +1) - pool_window)/stride_window +1))
    num_chan = 98
    spatialConvDepth = 1
    attentionLatent = 6
    def __init__(self, dropout):
        super(SincDriftBoundAttChoice_full, self).__init__()
        self.b0 = nn.BatchNorm2d(1, momentum=0.99)
        self.sinc_cnn2d_drift = sinc_conv(self.num_filters, self.filter_length, 500)
        self.sinc_cnn2d_bound = sinc_conv(self.num_filters, self.filter_length, 500)
        self.sinc_cnn2d_choice = sinc_conv(self.num_filters, self.filter_length, 500)

        self.b_drift =  nn.BatchNorm2d(self.num_filters, momentum=0.99)
        self.b_bound = nn.BatchNorm2d(self.num_filters, momentum=0.99)
        self.b_choice = nn.BatchNorm2d(self.num_filters, momentum=0.99)

        self.separable_conv_drift = SeparableConv2d(self.num_filters, self.num_filters, depth = self.spatialConvDepth, kernel_size= (self.num_chan,1))
        self.separable_conv_bound= SeparableConv2d(self.num_filters, self.num_filters, depth = self.spatialConvDepth, kernel_size= (self.num_chan,1))
        self.separable_conv_choice = SeparableConv2d(self.num_filters, self.num_filters, depth = self.spatialConvDepth, kernel_size= (self.num_chan,1))

        # self.separable_conv_point = SeparableConv2d_pointwise(32*3, 32*3, depth = 1, kernel_size= (1,8))

        # self.fc2 = torch.nn.Linear(495,495)
        self.b2_drift = nn.BatchNorm2d(self.num_filters*self.spatialConvDepth, momentum=0.99)
        self.b2_bound = nn.BatchNorm2d(self.num_filters*self.spatialConvDepth, momentum=0.99)
        self.b2_choice = nn.BatchNorm2d(self.num_filters*self.spatialConvDepth, momentum=0.99)

        # self.pool1 = nn.MaxPool2d((1, 151), stride=(1, 45))  # spatial activation of 100ms and with a stride of 13ms
        # self.pool1 = nn.AvgPool2d((1, 107), stride=(1, 45))  # spatial activation of 100ms and with a stride of 13ms
        self.pool1_drift = nn.AvgPool2d((1, self.pool_window), stride=(1, self.stride_window))  # spatial activation of 100ms and with a stride of 13ms
        self.pool1_bound = nn.AvgPool2d((1, self.pool_window), stride=(1, self.stride_window))  # spatial activation of 100ms and with a stride of 13ms
        self.pool1_choice = nn.AvgPool2d((1, self.pool_window), stride=(1, self.stride_window))  # spatial activation of 100ms and with a stride of 13ms

        self.dropout1_drift = torch.nn.Dropout(p=dropout)
        self.dropout1_bound= torch.nn.Dropout(p=dropout)
        self.dropout1_choice = torch.nn.Dropout(p=dropout)

        self.fc_drift = torch.nn.Linear(self.num_filters*self.spatialConvDepth*self.output_size,1)
        self.fc_bound = torch.nn.Linear(self.num_filters*self.spatialConvDepth*self.output_size,1)
        self.fc_choice = torch.nn.Linear(self.num_filters*self.spatialConvDepth*self.output_size,1)

        # self.fc5 = torch.nn.Linear(20, 1)
        # self.LeakyRelu = nn.LeakyReLU(0.2)
        self.gradients = None


    # attention layers
        self.gap0_drift = torch.nn.AdaptiveAvgPool2d(1)

        ### Fully Connected Multi-Layer Perceptron (FC-MLP)
        self.mlp0_drift = torch.nn.Sequential(
            torch.nn.Linear(self.num_filters, self.num_filters // self.attentionLatent, bias=False),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(self.num_filters // self.attentionLatent, self.num_filters, bias=False),
            torch.nn.Sigmoid()
        )


        self.gap0_bound = torch.nn.AdaptiveAvgPool2d(1)

        ### Fully Connected Multi-Layer Perceptron (FC-MLP)
        self.mlp0_bound = torch.nn.Sequential(
            torch.nn.Linear(self.num_filters, self.num_filters // self.attentionLatent, bias=False),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(self.num_filters // self.attentionLatent, self.num_filters, bias=False),
            torch.nn.Sigmoid()
        )

        self.gap0_choice = torch.nn.AdaptiveAvgPool2d(1)

        ### Fully Connected Multi-Layer Perceptron (FC-MLP)
        self.mlp0_choice = torch.nn.Sequential(
            torch.nn.Linear(self.num_filters, self.num_filters // self.attentionLatent, bias=False),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(self.num_filters // self.attentionLatent, self.num_filters, bias=False),
            torch.nn.Sigmoid()
        )


    # hook for the gradients of the activations
    def activations_filterhook(self, grad):
        self.gradients_filter = grad
    def activations_temporalhook(self, grad):
        self.gradients_temp = grad
    def forward(self, x):
        batch_n = x.shape[0]
        x = x.view(batch_n, 1, self.num_chan,self.t_length)
        x = self.b0(x)
        x = torch.squeeze(x)
        if batch_n > 1:
            x = torch.squeeze(x)
        else:
            x = x.view(batch_n,self.num_chan,self.t_length)
        x0_drift = self.sinc_cnn2d_drift(x)
        x0_bound = self.sinc_cnn2d_bound(x)
        x0_choice = self.sinc_cnn2d_choice(x)

        # start attention
        b_drift, c_drift, _, _ = x0_drift.size()
        y0_drift = self.gap0_drift(x0_drift).view(b_drift, c_drift)
        y0_drift = self.mlp0_drift(y0_drift).view(b_drift, c_drift, 1, 1)
        score_new_drift = x0_drift * y0_drift.expand_as(x0_drift)
        # end attention

        # start attention for choice
        b_bound, c_bound, _, _ = x0_bound.size()
        y0_bound = self.gap0_bound(x0_bound).view(b_bound, c_bound)
        y0_bound = self.mlp0_bound(y0_bound).view(b_bound, c_bound, 1, 1)
        score_new_bound = x0_bound * y0_bound.expand_as(x0_bound)
        # end attention


        # start attention for choice
        b_choice, c_choice, _, _ = x0_choice.size()
        y0_choice = self.gap0_choice(x0_choice).view(b_choice, c_choice)
        y0_choice = self.mlp0_choice(y0_choice).view(b_choice, c_choice, 1, 1)
        score_new_choice = x0_choice * y0_choice.expand_as(x0_choice)
        # end attention



        # spatial convulation layer
        score0_drift_ = self.b_drift(score_new_drift)
        score0_drift = self.separable_conv_drift(score0_drift_) # output is [n, 64,1,1870)
        # score0 = self.separable_conv_point(score0) # output is [n, 64,1,1870)

        # spatial convulation layer for bound
        score0_bound_ = self.b_bound(score_new_bound)
        score0_bound = self.separable_conv_bound(score0_bound_)  # output is [n, 64,1,1870)

        # spatial convulation layer for choice
        score0_choice_ = self.b_choice(score_new_choice)
        score0_choice = self.separable_conv_choice(score0_choice_) # output is [n, 64,1,1870)


        # relu  layers
        score_drift = self.b2_drift(score0_drift)
        score_drift = F.relu(score_drift)

        score_drift = self.pool1_drift(score_drift)
        score_drift = self.dropout1_drift(score_drift)  # output is [n, 64,1,,37)


        score_bound = self.b2_bound(score0_bound)
        score_bound = F.relu(score_bound)

        score_bound = self.pool1_bound(score_bound)
        score_bound = self.dropout1_bound(score_bound)  # output is [n, 64,1,,37)


        score_choice = self.b2_choice(score0_choice)
        score_choice = F.relu(score_choice)

        score_choice = self.pool1_choice(score_choice)
        score_choice = self.dropout1_choice(score_choice)  # output is [n, 64,1,,37)


        # output layer to choice
        score2_choice = score_choice.view(-1,self.num_filters*self.spatialConvDepth*self.output_size)
        score3_choice = self.fc_choice(score2_choice)   # choice
        score3_choice = F.sigmoid(score3_choice)

        # output layer for drift
        score2_drift = score_drift.view(-1,self.num_filters*1*self.output_size)  # output [batch size, 64*17)
        score3_drift= self.fc_drift(score2_drift)# output is [batch size, 1]



        # ouput layer for bound
        score2_bound =score_bound.view(-1,self.num_filters*self.spatialConvDepth*self.output_size)
        score4_bound = F.softplus(self.fc_bound(score2_bound))

        return score3_drift,score4_bound,score3_choice

    def get_activations_gradient_filter(self):
        return self.gradients_filter
    def get_activations_gradient_temp(self):
        return self.gradients_temp





