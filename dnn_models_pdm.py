# Created on 10/12/21 at 10:50 PM 

# Author: Jenny Sun
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import sys
from torch.autograd import Variable
import math

torch.manual_seed(2022)
np.random.seed(2022)
# random.seed(2021)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def flip(x, dim):
    xsize = x.size()
    dim = x.dim() + dim if dim < 0 else dim
    x = x.contiguous()
    x = x.view(-1, *xsize[dim:])
    x = x.view(x.size(0), x.size(1), -1)[:, getattr(torch.arange(x.size(1)-1,
                      -1, -1), ('cpu','cuda')[x.is_cuda])().long(), :]
    return x.view(xsize)


def sinc(band,t_right):
    y_right= torch.sin(2*math.pi*band*t_right)/(2*math.pi*band*t_right)
    y_left= flip(y_right,0)

    y=torch.cat([y_left,Variable(torch.ones(1)).cuda(),y_right])

    return y


def act_fun(act_type):
    if act_type == "relu":
        return nn.ReLU()

    if act_type == "tanh":
        return nn.Tanh()

    if act_type == "sigmoid":
        return nn.Sigmoid()

    if act_type == "leaky_relu":
        return nn.LeakyReLU(0.2)

    if act_type == "elu":
        return nn.ELU()

    if act_type == "softmax":
        return nn.LogSoftmax(dim=1)

    if act_type == "linear":
        return nn.LeakyReLU(1)  # initializzed like this, but not used in forward!


class SincConv_fast(nn.Module):
    """Sinc-based convolution
    Parameters
    ----------
    in_channels : `int`
        Number of input channels. Must be 1.
    out_channels : `int`
        Number of filters.
    kernel_size : `int`
        Filter length.
    sample_rate : `int`, optional
        Sample rate. Defaults to 16000.
    Usage
    -----
    See `torch.nn.Conv1d`
    Reference
    ---------
    Mirco Ravanelli, Yoshua Bengio,
    "Speaker Recognition from raw waveform with SincNet".
    https://arxiv.org/abs/1808.00158
    """

    @staticmethod
    def to_mel(hz):
        return 2595 * np.log10(1 + hz / 700)

    @staticmethod
    def to_hz(mel):
        return 700 * (10 ** (mel / 2595) - 1)

    def __init__(self, out_channels, kernel_size, sample_rate=1000, in_channels=1,
                 stride=1, padding=0, dilation=1, bias=False, groups=1, min_low_hz=150, min_band_hz=150):

        super(SincConv_fast, self).__init__()

        if in_channels != 1:
            # msg = (f'SincConv only support one input channel '
            #       f'(here, in_channels = {in_channels:d}).')
            msg = "SincConv only support one input channel (here, in_channels = {%i})" % (in_channels)
            raise ValueError(msg)

        self.out_channels = out_channels

        self.kernel_size = kernel_size

        # Forcing the filters to be odd (i.e, perfectly symmetrics)
        if kernel_size % 2 == 0:
            self.kernel_size = self.kernel_size + 1

        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        if bias:
            raise ValueError('SincConv does not support bias.')
        if groups > 1:
            raise ValueError('SincConv does not support groups.')

        self.sample_rate = sample_rate
        self.min_low_hz = min_low_hz
        self.min_band_hz = min_band_hz

        # initialize filterbanks such that they are equally spaced in Mel scale
        low_hz = 3
        high_hz = self.sample_rate / 2 - (self.min_low_hz + self.min_band_hz)

        mel = np.linspace(self.to_mel(low_hz),
                          self.to_mel(high_hz),
                          self.out_channels + 1)
        hz = self.to_hz(mel)

        # filter lower frequency (out_channels, 1)
        self.low_hz_ = nn.Parameter(torch.Tensor(hz[:-1]).view(-1, 1))

        # filter frequency band (out_channels, 1)
        self.band_hz_ = nn.Parameter(torch.Tensor(np.diff(hz)).view(-1, 1))

        # Hamming window
        # self.window_ = torch.hamming_window(self.kernel_size)
        n_lin = torch.linspace(0, (self.kernel_size / 2) - 1,
                               steps=int((self.kernel_size / 2)))  # computing only half of the window
        self.window_ = 0.54 - 0.46 * torch.cos(2 * math.pi * n_lin / self.kernel_size);

        # (1, kernel_size/2)
        n = (self.kernel_size - 1) / 2.0
        self.n_ = 2 * math.pi * torch.arange(-n, 0).view(1,
                                                         -1) / self.sample_rate  # Due to symmetry, I only need half of the time axes

    def forward(self, waveforms):
        """
        Parameters
        ----------
        waveforms : `torch.Tensor` (batch_size, 1, n_samples)
            Batch of waveforms.
        Returns
        -------
        features : `torch.Tensor` (batch_size, out_channels, n_samples_out)
            Batch of sinc filters activations.
        """

        self.n_ = self.n_.to(waveforms.device)

        self.window_ = self.window_.to(waveforms.device)

        low = self.min_low_hz + torch.abs(self.low_hz_)

        high = torch.clamp(low + self.min_band_hz + torch.abs(self.band_hz_), self.min_low_hz, self.sample_rate / 2)
        band = (high - low)[:, 0]

        f_times_t_low = torch.matmul(low, self.n_)
        f_times_t_high = torch.matmul(high, self.n_)

        band_pass_left = ((torch.sin(f_times_t_high) - torch.sin(f_times_t_low)) / (
                    self.n_ / 2)) * self.window_  # Equivalent of Eq.4 of the reference paper (SPEAKER RECOGNITION FROM RAW WAVEFORM WITH SINCNET). I just have expanded the sinc and simplified the terms. This way I avoid several useless computations.
        band_pass_center = 2 * band.view(-1, 1)
        band_pass_right = torch.flip(band_pass_left, dims=[1])

        band_pass = torch.cat([band_pass_left, band_pass_center, band_pass_right], dim=1)
        # b = torch.zeros((band_pass.shape[0],120,band_pass.shape[1]))
        # for i in range(0,120):
        #     b[:,i,:] = band_pass / (2*band[:,None])
        band_pass = torch.cat([band_pass_left, band_pass_center, band_pass_right], dim=1)

        band_pass = band_pass / (2 * band[:, None])


        self.filters = (band_pass).view(
            self.out_channels, 1,1, self.kernel_size)
        batch_n = waveforms.shape[0]
        return F.conv2d(waveforms.view(batch_n,1,120,1000), self.filters, stride=1,
                        padding=self.padding, dilation=self.dilation,
                        bias=None, groups=1)


class sinc_conv(nn.Module):

    def __init__(self, N_filt, Filt_dim, fs):
        super(sinc_conv, self).__init__()

        # Mel Initialization of the filterbanks
        # low_freq_mel = 80
        # high_freq_mel = (2595 * np.log10(1 + (fs / 2) / 700))  # Convert Hz to Mel
        low_freq_mel = 0
        high_freq_mel = 30
        # high_freq_mel = 63
        # mel_points = np.linspace(low_freq_mel, high_freq_mel, N_filt)  # Equally spaced in Mel scale
        # f_cos = (700 * (10 ** (mel_points / 2595) - 1))  # Convert Mel to Hz
        mel_points = np.random.uniform(low_freq_mel,high_freq_mel, N_filt)
        # b1 = mel_points
        # b2 = np.zeros_like(b1)
        # b2[0:-1] = np.diff(b1)
        # b2[-1] = b2[-2]

        # mel_points = np.linspace(low_freq_mel, high_freq_mel, N_filt)  # Equally spaced in Mel scale
        b1 = mel_points
        b2 = np.zeros_like(b1) + 2

        self.freq_scale = fs * 1.0
        self.filt_b1 = nn.Parameter(torch.from_numpy(b1 / self.freq_scale))
        self.filt_band = nn.Parameter(torch.from_numpy(b2 / self.freq_scale))

        self.N_filt = N_filt
        self.Filt_dim = Filt_dim
        self.fs = fs

    def forward(self, x):
        filters = Variable(torch.zeros((self.N_filt, self.Filt_dim))).cuda()
        N = self.Filt_dim
        t_right = Variable(torch.linspace(1, (N - 1) / 2, steps=int((N - 1) / 2)) / self.fs).cuda()

        min_freq = 1.0
        min_band = 2.0;

        filt_beg_freq = torch.abs(self.filt_b1) +  min_freq / self.freq_scale
        filt_end_freq = torch.clamp(filt_beg_freq + (torch.abs(self.filt_band) + min_band / self.freq_scale), 3/self.freq_scale, (self.fs/2)/self.freq_scale)
        n = torch.linspace(0, N, steps=N)

        # Filter window (hamming)
        window = 0.54 - 0.46 * torch.cos(2 * math.pi * n / N);
        window = Variable(window.float().cuda())

        for i in range(self.N_filt):
            low_pass1 = 2 * filt_beg_freq[i].float() * sinc(filt_beg_freq[i].float() * self.freq_scale, t_right)
            low_pass2 = 2 * filt_end_freq[i].float() * sinc(filt_end_freq[i].float() * self.freq_scale, t_right)
            band_pass = (low_pass2 - low_pass1)

            band_pass = band_pass / torch.max(band_pass)   #normalize to one

            filters[i, :] = band_pass.cuda() * window
        batch_n = x.shape[0]
        x = x.cuda()
        # return filters
        out = F.conv2d(x.view(batch_n,1,x.shape[1],x.shape[-1]), filters.view(self.N_filt, 1, 1,self.Filt_dim))

        return out



        # self.filters = (band_pass).view(
        #     self.out_channels, 1,1, self.kernel_size)
        #
        # return F.conv2d(waveforms.view(batch_n,1,120,1000), self.filters, stride=1,
        #                 padding=self.padding, dilation=self.dilation,
        #                 bias=None, groups=1)




#
# class MLP(nn.Module):
#     def __init__(self, options):
#         super(MLP, self).__init__()
#
#         self.input_dim = int(options['input_dim'])
#         self.fc_lay = options['fc_lay']
#         self.fc_drop = options['fc_drop']
#         self.fc_use_batchnorm = options['fc_use_batchnorm']
#         self.fc_use_laynorm = options['fc_use_laynorm']
#         self.fc_use_laynorm_inp = options['fc_use_laynorm_inp']
#         self.fc_use_batchnorm_inp = options['fc_use_batchnorm_inp']
#         self.fc_act = options['fc_act']
#
#         self.wx = nn.ModuleList([])
#         self.bn = nn.ModuleList([])
#         self.ln = nn.ModuleList([])
#         self.act = nn.ModuleList([])
#         self.drop = nn.ModuleList([])
#
#         # input layer normalization
#         if self.fc_use_laynorm_inp:
#             self.ln0 = LayerNorm(self.input_dim)
#
#         # input batch normalization
#         if self.fc_use_batchnorm_inp:
#             self.bn0 = nn.BatchNorm1d([self.input_dim], momentum=0.05)
#
#         self.N_fc_lay = len(self.fc_lay)
#
#         current_input = self.input_dim
#
#         # Initialization of hidden layers
#
#         for i in range(self.N_fc_lay):
#
#             # dropout
#             self.drop.append(nn.Dropout(p=self.fc_drop[i]))
#
#             # activation
#             self.act.append(act_fun(self.fc_act[i]))
#
#             add_bias = True
#
#             # layer norm initialization
#             self.ln.append(LayerNorm(self.fc_lay[i]))
#             self.bn.append(nn.BatchNorm1d(self.fc_lay[i], momentum=0.05))
#
#             if self.fc_use_laynorm[i] or self.fc_use_batchnorm[i]:
#                 add_bias = False
#
#             # Linear operations
#             self.wx.append(nn.Linear(current_input, self.fc_lay[i], bias=add_bias))
#
#             # weight initialization
#             self.wx[i].weight = torch.nn.Parameter(
#                 torch.Tensor(self.fc_lay[i], current_input).uniform_(-np.sqrt(0.01 / (current_input + self.fc_lay[i])),
#                                                                      np.sqrt(0.01 / (current_input + self.fc_lay[i]))))
#             self.wx[i].bias = torch.nn.Parameter(torch.zeros(self.fc_lay[i]))
#
#             current_input = self.fc_lay[i]
#
#     def forward(self, x):
#
#         # Applying Layer/Batch Norm
#         if bool(self.fc_use_laynorm_inp):
#             x = self.ln0((x))
#
#         if bool(self.fc_use_batchnorm_inp):
#             x = self.bn0((x))
#
#         for i in range(self.N_fc_lay):
#
#             if self.fc_act[i] != 'linear':
#
#                 if self.fc_use_laynorm[i]:
#                     x = self.drop[i](self.act[i](self.ln[i](self.wx[i](x))))
#
#                 if self.fc_use_batchnorm[i]:
#                     x = self.drop[i](self.act[i](self.bn[i](self.wx[i](x))))
#
#                 if self.fc_use_batchnorm[i] == False and self.fc_use_laynorm[i] == False:
#                     x = self.drop[i](self.act[i](self.wx[i](x)))
#
#             else:
#                 if self.fc_use_laynorm[i]:
#                     x = self.drop[i](self.ln[i](self.wx[i](x)))
#
#                 if self.fc_use_batchnorm[i]:
#                     x = self.drop[i](self.bn[i](self.wx[i](x)))
#
#                 if self.fc_use_batchnorm[i] == False and self.fc_use_laynorm[i] == False:
#                     x = self.drop[i](self.wx[i](x))
#
#         return x



class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, depth, kernel_size, bias=False):
        super(SeparableConv2d, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, out_channels*depth, kernel_size=kernel_size, groups=in_channels, bias=bias)
        # self.pointwise = nn.Conv2d(out_channels*depth, out_channels, kernel_size=1, bias=bias)

    def forward(self, x):
        out = self.depthwise(x)
        # out = self.pointwise(out)
        return out


class SeparableConv2d_pointwise(nn.Module):
    def __init__(self, in_channels, out_channels, depth, kernel_size, bias=False):
        super(SeparableConv2d_pointwise, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, out_channels*depth, kernel_size=kernel_size, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(out_channels*depth, out_channels, kernel_size=1, bias=bias)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out




class Sinc_Conv2d(nn.Module):
    def __init__(self):
        super(Sinc_Conv2d, self).__init__()
        self.b0 = nn.BatchNorm2d(1, momentum=0.99)

        self.sinc_cnn2d = sinc_conv(32, 131, 500)

        self.b =  nn.BatchNorm2d(32, momentum=0.99)
        self.separable_conv = SeparableConv2d(32, 32, depth = 2, kernel_size= (98,1))
        # self.dropout0 = torch.nn.Dropout(p=.25)
        # self.fc2 = torch.nn.Linear(495,495)
        self.b2 = nn.BatchNorm2d(64, momentum=0.99)
        self.pool1 = nn.AvgPool2d((1, 215), stride=(1, 45))  # spatial activation of 100ms and with a stride of 13ms
        # self.pool1 = nn.AvgPool2d((1, 151), stride=(1, 45))  # spatial activation of 100ms and with a stride of 13ms

        self.dropout1 = torch.nn.Dropout(p=.5)
        self.fc3 = torch.nn.Linear(64*37,1)


    def forward(self, x):
        batch_n = x.shape[0]
        x = x.view(batch_n, 1, 81,2000).cuda()
        x = self.b0(x.cuda())
        x = torch.squeeze(x.cuda())
        x = self.sinc_cnn2d(x)
        # x = F.relu(x)
        # x = self.fc1(x)
        x = self.b(x)
        x = self.separable_conv(x)
        # x = self.dropout0(x)
        # x = F.relu(x)
        # x = self.fc2(x)
        x = self.b2(x)
        x = F.elu(x,alpha=1)
        # x = F.relu(x)
        x = self.pool1(x)
        x = self.dropout1(x)
        x = x.view(-1,64*37)
        x=self.fc3(x)
        # x = F.softmax(x, dim=1)
        return x
# # # # # # #
# # # # # # # #
# batch_n = data.shape[0]
# x = data.view(batch_n, 1, 81, 2000).cuda()
# b0 = nn.BatchNorm2d(1, momentum=0.99).cuda()
# x = b0(x)
# x = torch.squeeze(x.cuda())
# sinc_cnn2d = sinc_conv(32, 131, 500) # (input - kernel + 2P) /S + 1
# x = sinc_cnn2d(x)
# #
# # # fc1 = torch.nn.Linear(495, 495).cuda()
# # # fc1 = torch.nn.Linear(495, 495).cuda()
# # # x = fc1(x)
# # #
# b =  nn.BatchNorm2d(32, momentum=0.99).cuda()
# x = b(x)
# #
# separable_conv = SeparableConv2d(32, 32, depth=2, kernel_size=(81, 1)).cuda()
# x = separable_conv(x)
# # # fc2 = torch.nn.Linear(495, 495).cuda()
# # # x = fc2(x)
# # b2 = nn.BatchNorm2d(64, momentum=0.99).cuda()
# # x = b2(x)
# # x = F.elu(x,alpha=1)
# pool1 = nn.AvgPool2d((1,215), stride=(1,45)).cuda() #spatial activation of 100ms and with a stride of 13ms
# x = pool1(x)
# # dropout1 = torch.nn.Dropout(p = .5).cuda()
# x = x.view(-1,61*39)
# x = dropout1(x)
# #
# fc3 = torch.nn.Linear(61*39,1).cuda()
# x = fc3(x)
# # # #
# #
# x=fc1(x)
# # b = nn.BatchNorm2d(32, momentum=0.99)
# # x = b(x)
# # x = separable_conv(x)
# # x = fc2(x)
# # b2 = nn.BatchNorm2d(64, momentum=0.99)
# # x = b2(x)
# # x = F.elu(x, alpha=1)
# # x = pool1(x)
# # x = dropout1(x)
# # x = x.view(-1,64*11)
# # x=fc3(x)



class Sinc_Conv2d_new(nn.Module):
    # class global variables
    # filter_length = 251
    filter_length = 131
    t_length = 500
    pool_window = int(np.rint(((t_length- filter_length +1) * 250)/ (t_length*2)))
    stride_window =  int(np.rint(((t_length- filter_length +1) * 100)/ (t_length*2)))
    if pool_window % 2 == 0 :
        pool_window -= 1
    if stride_window % 2 == 0 :
        stride_window -= 1
    output_size = int(np.floor(((t_length- filter_length +1) - pool_window)/stride_window +1))
    num_chan = 98

    def __init__(self, dropout):
        super(Sinc_Conv2d_new, self).__init__()
        self.b0 = nn.BatchNorm2d(1, momentum=0.99)
        self.sinc_cnn2d = sinc_conv(32, self.filter_length, 500)
        # self.fc1 = torch.nn.Linear(495,495)
        self.b =  nn.BatchNorm2d(32, momentum=0.99)
        self.separable_conv = SeparableConv2d(32, 32, depth = 2, kernel_size= (self.num_chan,1))
        # self.fc2 = torch.nn.Linear(495,495)
        self.b2 = nn.BatchNorm2d(64, momentum=0.99)
        # self.pool1 = nn.MaxPool2d((1, 151), stride=(1, 45))  # spatial activation of 100ms and with a stride of 13ms
        # self.pool1 = nn.AvgPool2d((1, 107), stride=(1, 45))  # spatial activation of 100ms and with a stride of 13ms
        self.pool1 = nn.AvgPool2d((1, self.pool_window), stride=(1, self.stride_window))  # spatial activation of 100ms and with a stride of 13ms

        self.dropout1 = torch.nn.Dropout(p=dropout)
        self.fc3 = torch.nn.Linear(64*self.output_size,1)
        # self.fc3 = torch.nn.Linear(64*14,1)
        self.gradients = None

    # hook for the gradients of the activations
    def activations_filterhook(self, grad):
        self.gradients_filter = grad
    def activations_temporalhook(self, grad):
        self.gradients_temp = grad
    def forward(self, x):
        batch_n = x.shape[0]
        x = x.view(batch_n, 1, self.num_chan,self.t_length).cuda()
        x = self.b0(x.cuda())
        x = torch.squeeze(x.cuda())
        if batch_n > 1:
            x = torch.squeeze(x.cuda())
        else:
            x = x.view(batch_n,self.num_chan,self.t_length)
        x = self.sinc_cnn2d(x)
        h = x.register_hook(self.activations_filterhook)
        score = self.b(x)
        score = self.separable_conv(score) # output is [n, 64,1,1870)
        score = self.b2(score)
        score = F.relu(score)
        # score = F.elu(score,alpha=1)
        score = self.pool1(score)
        score = self.dropout1(score)  # output is [n, 64,1,,37)
        h2 = score.register_hook(self.activations_temporalhook)
        score2 = score.view(-1,64*self.output_size)  # output [batch size, 64*17)
        # score2 = score.view(-1,64*14)
        score2=self.fc3(score2) # output is [batch size, 1]
        # score4 = self.fc4(score2)
        # x = F.softmax(x, dim=1)
        return score2

    def get_activations_gradient_filter(self):
        return self.gradients_filter
    def get_activations_gradient_temp(self):
        return self.gradients_temp




class Sinc_Conv2d_ddm(nn.Module):
    # class global variables
    # filter_length = 251
    filter_length = 131
    t_length = 500
    pool_window = int(np.rint(((t_length- filter_length +1) * 250)/ (t_length*2)))
    stride_window =  int(np.rint(((t_length- filter_length +1) * 100)/ (t_length*2)))
    if pool_window % 2 == 0 :
        pool_window -= 1
    if stride_window % 2 == 0 :
        stride_window -= 1
    output_size = int(np.floor(((t_length- filter_length +1) - pool_window)/stride_window +1))
    num_chan = 98

    def __init__(self, dropout):
        super(Sinc_Conv2d_ddm, self).__init__()
        self.b0 = nn.BatchNorm2d(1, momentum=0.99)
        self.sinc_cnn2d = sinc_conv(32, self.filter_length, 500)
        # self.sinc_cnn2d = sinc_conv(32, self.filter_length, 500)
        # self.fc1 = torch.nn.Linear(495,495)
        self.b =  nn.BatchNorm2d(32, momentum=0.99)
        self.separable_conv = SeparableConv2d(32, 32, depth = 2, kernel_size= (self.num_chan,1))
        # self.fc2 = torch.nn.Linear(495,495)
        self.b2 = nn.BatchNorm2d(64, momentum=0.99)
        # self.pool1 = nn.MaxPool2d((1, 151), stride=(1, 45))  # spatial activation of 100ms and with a stride of 13ms
        # self.pool1 = nn.AvgPool2d((1, 107), stride=(1, 45))  # spatial activation of 100ms and with a stride of 13ms
        self.pool1 = nn.AvgPool2d((1, self.pool_window), stride=(1, self.stride_window))  # spatial activation of 100ms and with a stride of 13ms

        self.dropout1 = torch.nn.Dropout(p=dropout)
        self.fc3 = torch.nn.Linear(64*self.output_size,1)
        # self.fc3 = torch.nn.Linear(64*14,1)
        self.gradients = None

    # hook for the gradients of the activations
    def activations_filterhook(self, grad):
        self.gradients_filter = grad
    def activations_temporalhook(self, grad):
        self.gradients_temp = grad
    def forward(self, x):
        batch_n = x.shape[0]
        x = x.view(batch_n, 1, self.num_chan,self.t_length).cuda()
        x = self.b0(x.cuda())
        x = torch.squeeze(x.cuda())
        if batch_n > 1:
            x = torch.squeeze(x.cuda())
        else:
            x = x.view(batch_n,self.num_chan,self.t_length)
        x = self.sinc_cnn2d(x)
        h = x.register_hook(self.activations_filterhook)
        score = self.b(x)
        score = self.separable_conv(score) # output is [n, 64,1,1870)
        score = self.b2(score)
        score = F.relu(score)
        # score = F.elu(score,alpha=1)
        score = self.pool1(score)
        score = self.dropout1(score)  # output is [n, 64,1,,37)
        h2 = score.register_hook(self.activations_temporalhook)
        score2 = score.view(-1,64*self.output_size)  # output [batch size, 64*17)
        # score2 = score.view(-1,64*14)
        score2=self.fc3(score2)
        # output is [batch size, 1]
        # score4 = self.fc4(score2)
        # x = F.softmax(x, dim=1)
        return score2

    def get_activations_gradient_filter(self):
        return self.gradients_filter
    def get_activations_gradient_temp(self):
        return self.gradients_temp


#
# #
# class Sinc_Conv2d_ddm_2param(nn.Module):
#     # class global variables
#     # filter_length = 251
#     filter_length = 131
#     t_length = 500
#     pool_window = int(np.rint(((t_length- filter_length +1) * 250)/ (t_length*2)))
#     stride_window =  int(np.rint(((t_length- filter_length +1) * 100)/ (t_length*2)))
#     if pool_window % 2 == 0 :
#         pool_window -= 1
#     if stride_window % 2 == 0 :
#         stride_window -= 1
#     output_size = int(np.floor(((t_length- filter_length +1) - pool_window)/stride_window +1))
#     num_chan = 98
#
#     def __init__(self, dropout):
#         super(Sinc_Conv2d_ddm_2param, self).__init__()
#         self.b0 = nn.BatchNorm2d(1, momentum=0.99)
#         self.sinc_cnn2d = sinc_conv(32, self.filter_length, 500)
#         # self.fc1 = torch.nn.Linear(495,495)
#         self.b =  nn.BatchNorm2d(32, momentum=0.99)
#         self.separable_conv = SeparableConv2d(32, 32, depth = 2, kernel_size= (self.num_chan,1))
#         # self.fc2 = torch.nn.Linear(495,495)
#         self.b2 = nn.BatchNorm2d(64, momentum=0.99)
#         # self.pool1 = nn.MaxPool2d((1, 151), stride=(1, 45))  # spatial activation of 100ms and with a stride of 13ms
#         # self.pool1 = nn.AvgPool2d((1, 107), stride=(1, 45))  # spatial activation of 100ms and with a stride of 13ms
#         self.pool1 = nn.AvgPool2d((1, self.pool_window), stride=(1, self.stride_window))  # spatial activation of 100ms and with a stride of 13ms
#
#         self.dropout1 = torch.nn.Dropout(p=dropout)
#         self.fc3 = torch.nn.Linear(64*self.output_size,1)
#         self.fc4 =  torch.nn.Linear(64*self.output_size,1)
#         # self.fc5 = torch.nn.Linear(20, 1)
#         # self.LeakyRelu = nn.LeakyReLU(0.2)
#         self.gradients = None
#
#     # hook for the gradients of the activations
#     def activations_filterhook(self, grad):
#         self.gradients_filter = grad
#     def activations_temporalhook(self, grad):
#         self.gradients_temp = grad
#     def forward(self, x):
#         batch_n = x.shape[0]
#         x = x.view(batch_n, 1, self.num_chan,self.t_length).cuda()
#         x = self.b0(x.cuda())
#         x = torch.squeeze(x.cuda())
#         if batch_n > 1:
#             x = torch.squeeze(x.cuda())
#         else:
#             x = x.view(batch_n,self.num_chan,self.t_length)
#         x = self.sinc_cnn2d(x)
#         h = x.register_hook(self.activations_filterhook)
#         score = self.b(x)
#         score = self.separable_conv(score) # output is [n, 64,1,1870)
#         score = self.b2(score)
#         score = F.relu(score)
#         # score = F.elu(score,alpha=1)
#         score = self.pool1(score)
#         score = self.dropout1(score)  # output is [n, 64,1,,37)
#         h2 = score.register_hook(self.activations_temporalhook)
#         score2 = score.view(-1,64*self.output_size)  # output [batch size, 64*17)
#         score3= self.fc3(score2)# output is [batch size, 1]
#         # score3 = -F.relu(-score3)
#         # score3 = F.sigmoid(score3)* -6
#         # score3 = torch.abs(score3)
#         # score3 = torch.abs(score3)
#
#         # score4 = self.fc4(score2)
#         score4 = F.sigmoid(self.fc4(score2))    # right now uses sigmoid
#         score4 = torch.exp(score4)
#         # score4 = torch.clamp(score4, 0.37, 2.65)
#         # score4 = F.relu(self.fc5(score4))
#         # score5 = torch.exp(score4)
#         # score4 = torch.max(torch.zeros_like(self.fc4(score2))+0.2, score4)
#         # score4 = torch.clamp(score4, min = 0.2, max = 3)
#         # score5 = torch.mean(score4,axis=0)
#         return score3, score4
#
#     def get_activations_gradient_filter(self):
#         return self.gradients_filter
#     def get_activations_gradient_temp(self):
#         return self.gradients_temp

class Sinc_Conv2d_ddm_2param_split(nn.Module):
    # class global variables
    # filter_length = 251
    filter_length = 131
    t_length = 500
    pool_window = int(np.rint(((t_length - filter_length + 1) * 250) / (t_length * 2)))
    stride_window = int(np.rint(((t_length - filter_length + 1) * 100) / (t_length * 2)))
    if pool_window % 2 == 0:
        pool_window -= 1
    if stride_window % 2 == 0:
        stride_window -= 1
    output_size = int(np.floor(((t_length - filter_length + 1) - pool_window) / stride_window + 1))
    num_chan = 98

    def __init__(self, dropout):
        super(Sinc_Conv2d_ddm_2param_split, self).__init__()
        self.b0 = nn.BatchNorm2d(1, momentum=0.99)
        self.sinc_cnn2d = sinc_conv(32, self.filter_length, 500)
        self.sinc_cnn2d_2 = sinc_conv(32, self.filter_length, 500)

        # self.fc1 = torch.nn.Linear(495,495)
        self.b = nn.BatchNorm2d(32, momentum=0.99)
        self.b_2 = nn.BatchNorm2d(32, momentum=0.99)

        self.separable_conv = SeparableConv2d(32, 32, depth=2, kernel_size=(self.num_chan, 1))
        self.separable_conv_2 = SeparableConv2d(32, 32, depth=2, kernel_size=(self.num_chan, 1))

        self.separable_conv_point = SeparableConv2d_pointwise(64, 64, depth = 1, kernel_size= (1,8))
        self.separable_conv_point_2 = SeparableConv2d_pointwise(64, 64, depth = 1, kernel_size= (1,8))


        # self.fc2 = torch.nn.Linear(495,495)
        self.b2 = nn.BatchNorm2d(64, momentum=0.99)
        self.b2_2 = nn.BatchNorm2d(64, momentum=0.99)

        # self.pool1 = nn.MaxPool2d((1, 151), stride=(1, 45))  # spatial activation of 100ms and with a stride of 13ms
        # self.pool1 = nn.AvgPool2d((1, 107), stride=(1, 45))  # spatial activation of 100ms and with a stride of 13ms
        self.pool1 = nn.AvgPool2d((1, self.pool_window), stride=(
        1, self.stride_window))  # spatial activation of 100ms and with a stride of 13ms
        self.pool1_2 = nn.AvgPool2d((1, self.pool_window), stride=(
        1, self.stride_window))  # spatial activation of 100ms and with a stride of 13ms
        self.dropout1 = torch.nn.Dropout(p=dropout)
        self.dropout1_2 = torch.nn.Dropout(p=0.8)

        self.fc3 = torch.nn.Linear(64 * self.output_size, 1)
        self.fc4 = torch.nn.Linear(64 * self.output_size, 1)
        # self.fc5 = torch.nn.Linear(20, 1)
        # self.LeakyRelu = nn.LeakyReLU(0.2)
        self.gradients = None

    # hook for the gradients of the activations
    def activations_filterhook(self, grad):
        self.gradients_filter = grad

    def activations_temporalhook(self, grad):
        self.gradients_temp = grad

    def forward(self, x):
        batch_n = x.shape[0]
        x = x.view(batch_n, 1, self.num_chan, self.t_length).cuda()
        x = self.b0(x.cuda())
        x = torch.squeeze(x.cuda())
        if batch_n > 1:
            x = torch.squeeze(x.cuda())
        else:
            x = x.view(batch_n, self.num_chan, self.t_length)
        score0 = self.sinc_cnn2d(x)
        score_ = self.sinc_cnn2d_2(x)
        # x2 = self.sinc_cnn2d_2(x)
        h = score0.register_hook(self.activations_filterhook)

        score0 = self.b(score0)
        score_ = self.b_2(score_)


        score0 = self.separable_conv(score0)  # output is [n, 64,1,1870)
        score_ = self.separable_conv_2(score_)  # output is [n, 64,1,1870)

        score0 = self.separable_conv_point(score0)  # output is [n, 64,1,1870)
        score_ = self.separable_conv_point_2(score_)  # output is [n, 64,1,1870)


        score0 = self.b2(score0)
        score_ = self.b2_2(score_)


        score0 = F.relu(score0)
        score_ = F.relu(score_)
        # score = F.elu(score,alpha=1)
        score0 = self.pool1(score0)
        score0 = self.dropout1(score0)  # output is [n, 64,1,,37)
        score_ = self.pool1_2(score_)
        score_ = self.dropout1_2(score_)  # output is [n, 64,1,,37)
        h2 = score0.register_hook(self.activations_temporalhook)
        score0 = score0.view(-1, 64 * self.output_size)  # output [batch size, 64*17)
        score_ = score_.view(-1, 64 * self.output_size)  # output [batch size, 64*17)

        score3 = self.fc3(score0)  # output is [batch size, 1]
        # score3 = F.sigmoid(score3) * -6
        # score3 = -F.relu(-score3)
        # score3 = F.sigmoid(score3)* -6
        # score3 = torch.abs(score3)
        # score3 = torch.abs(score3)

        # score4 = self.fc4(score2)
        score4 = F.sigmoid(self.fc4(score_))*3  # right now uses sigmoid
        # score4 = torch.exp(score4)
        # score4 = torch.clamp(score4, 0.37, 2.65)
        # score4 = F.relu(self.fc5(score4))
        # score5 = torch.exp(score4)
        # score4 = torch.max(torch.zeros_like(self.fc4(score2))+0.2, score4)
        # score4 = torch.clamp(score4, min = 0.2, max = 3)
        # score5 = torch.mean(score4,axis=0)
        return score3, score4, (x,x)

    def get_activations_gradient_filter(self):
        return self.gradients_filter

    def get_activations_gradient_temp(self):
        return self.gradients_temp


class Sinc_Conv2d_ddm_2param(nn.Module):
    # class global variables
    # filter_length = 251
    filter_length = 131
    t_length = 500
    pool_window = int(np.rint(((t_length- filter_length +1) * 250)/ (t_length*2)))
    stride_window =  int(np.rint(((t_length- filter_length +1) * 100)/ (t_length*2)))
    if pool_window % 2 == 0 :
        pool_window -= 1
    if stride_window % 2 == 0 :
        stride_window -= 1
    output_size = int(np.floor(((t_length- filter_length +1) - pool_window)/stride_window +1))
    num_chan = 98

    def __init__(self, dropout):
        super(Sinc_Conv2d_ddm_2param, self).__init__()
        self.b0 = nn.BatchNorm2d(1, momentum=0.99)
        self.sinc_cnn2d = sinc_conv(32, self.filter_length, 500)
        # self.fc1 = torch.nn.Linear(495,495)
        self.b =  nn.BatchNorm2d(32, momentum=0.99)
        self.separable_conv = SeparableConv2d(32, 32, depth = 2, kernel_size= (self.num_chan,1))
        # self.fc2 = torch.nn.Linear(495,495)
        self.b2 = nn.BatchNorm2d(64, momentum=0.99)
        # self.pool1 = nn.MaxPool2d((1, 151), stride=(1, 45))  # spatial activation of 100ms and with a stride of 13ms
        # self.pool1 = nn.AvgPool2d((1, 107), stride=(1, 45))  # spatial activation of 100ms and with a stride of 13ms
        self.pool1 = nn.AvgPool2d((1, self.pool_window), stride=(1, self.stride_window))  # spatial activation of 100ms and with a stride of 13ms

        self.dropout1 = torch.nn.Dropout(p=dropout)
        self.fc3 = torch.nn.Linear(64*self.output_size,1)
        self.fc4 =  torch.nn.Linear(64*self.output_size,1)
        # self.fc5 = torch.nn.Linear(20, 1)
        # self.LeakyRelu = nn.LeakyReLU(0.2)
        self.gradients = None

    # hook for the gradients of the activations
    def activations_filterhook(self, grad):
        self.gradients_filter = grad
    def activations_temporalhook(self, grad):
        self.gradients_temp = grad
    def forward(self, x):
        batch_n = x.shape[0]
        x = x.view(batch_n, 1, self.num_chan,self.t_length).cuda()
        x = self.b0(x.cuda())
        x = torch.squeeze(x.cuda())
        if batch_n > 1:
            x = torch.squeeze(x.cuda())
        else:
            x = x.view(batch_n,self.num_chan,self.t_length)
        x = self.sinc_cnn2d(x)
        h = x.register_hook(self.activations_filterhook)
        score = self.b(x)
        score = self.separable_conv(score) # output is [n, 64,1,1870)
        score = self.b2(score)
        score = F.relu(score)
        # score = F.elu(score,alpha=1)
        score = self.pool1(score)
        score = self.dropout1(score)  # output is [n, 64,1,,37)
        h2 = score.register_hook(self.activations_temporalhook)
        score2 = score.view(-1,64*self.output_size)  # output [batch size, 64*17)
        score3= F.relu(self.fc3(score2))# output is [batch size, 1]
        # score4 = self.fc4(score2)
        score4 = F.tanh(self.fc4(score2))
        score4 = torch.exp(score4)
        # score4 = F.relu(self.fc5(score4))
        # score5 = torch.exp(score4)
        # score4 = torch.max(torch.zeros_like(self.fc4(score2))+0.2, score4)
        # score4 = torch.clamp(score4, min = 0.2, max = 3)
        # score5 = torch.mean(score4,axis=0)
        return -score3, score4

    def get_activations_gradient_filter(self):
        return self.gradients_filter
    def get_activations_gradient_temp(self):
        return self.gradients_temp





class Sinc_Conv2d_dual(nn.Module):
    # class global variables
    # filter_length = 251
    filter_length = 131
    t_length = 400
    num_chan = 98
    post_convlength = t_length -filter_length + 1


    def __init__(self, dropout):
        super(Sinc_Conv2d_dual, self).__init__()
        self.b0 = nn.BatchNorm2d(1, momentum=0.99)
        self.sinc_cnn2d = sinc_conv(32, self.filter_length, 500)
        # self.fc1 = torch.nn.Linear(495,495)
        self.b =  nn.BatchNorm2d(32, momentum=0.99)
        self.separable_conv = SeparableConv2d(32, 32, depth = 2, kernel_size= (self.num_chan,1))
        # self.fc2 = torch.nn.Linear(495,495)
        self.b2 = nn.BatchNorm2d(64, momentum=0.99)
        # self.pool1 = nn.MaxPool2d((1, 151), stride=(1, 45))  # spatial activation of 100ms and with a stride of 13ms
        # self.pool1 = nn.AvgPool2d((1, 107), stride=(1, 45))  # spatial activation of 100ms and with a stride of 13ms
        self.pool1 = nn.AvgPool2d((1, 107), stride=(1, 45))  # spatial activation of 100ms and with a stride of 13ms

        self.dropout1 = torch.nn.Dropout(p=dropout)
        self.fc3 = torch.nn.Linear(64*17,1)
        self.fc4 = torch.nn.Linear(64*17,2)

        # self.fc3 = torch.nn.Linear(64*14,1)
        self.gradients = None

    # hook for the gradients of the activations
    def activations_filterhook(self, grad):
        self.gradients_filter = grad
    def activations_temporalhook(self, grad):
        self.gradients_temp = grad
    def forward(self, x):
        batch_n = x.shape[0]
        x = x.view(batch_n, 1, self.num_chan,self.t_length).cuda()
        x = self.b0(x.cuda())
        x = torch.squeeze(x.cuda())
        if batch_n > 1:
            x = torch.squeeze(x.cuda())
        else:
            x = x.view(batch_n,self.num_chan,self.t_length)
        x = self.sinc_cnn2d(x)
        h = x.register_hook(self.activations_filterhook)
        score = self.b(x)
        score = self.separable_conv(score) # output is [n, 64,1,1870)
        score = self.b2(score)
        score = F.relu(score)
        # score = F.elu(score,alpha=1)
        score = self.pool1(score)
        score = self.dropout1(score)  # output is [n, 64,1,,37)
        h2 = score.register_hook(self.activations_temporalhook)
        # score2 = score.view(-1,64*5)  # output [batch size, 64*17)
        score2 = score.view(-1,64*17)
        score3=self.fc3(score2) # output is [batch size, 1]
        score4 = self.fc4(score2)
        # x = F.softmax(x, dim=1)
        return score3, score4

    def get_activations_gradient_filter(self):
        return self.gradients_filter
    def get_activations_gradient_temp(self):
        return self.gradients_temp




#
# loss = nn.MSEloss()
# lossCE = nn.CrossEntropyLoss()
#
# loss.backward() + lossCE.backward()


#
class Sinc_Conv2d_attention_post(nn.Module):
    # class global variables
    # filter_length = 251
    filter_length = 131
    t_length = 500
    pool_window = int(np.rint(((t_length- filter_length +1) * 250)/ (t_length*2)))
    stride_window =  int(np.rint(((t_length- filter_length +1) * 100)/ (t_length*2)))
    if pool_window % 2 == 0 :
        pool_window -= 1
    if stride_window % 2 == 0 :
        stride_window -= 1
    output_size = int(np.floor(((t_length- filter_length +1) - pool_window)/stride_window +1))
    num_chan = 98

    def __init__(self, dropout):
        super(Sinc_Conv2d_attention_post, self).__init__()
        self.b0 = nn.BatchNorm2d(1, momentum=0.99)
        self.sinc_cnn2d = sinc_conv(32, self.filter_length, 500)
        # self.fc1 = torch.nn.Linear(495,495)
        self.b =  nn.BatchNorm2d(32, momentum=0.99)
        self.separable_conv = SeparableConv2d(32, 32, depth = 2, kernel_size= (self.num_chan,1))
        # self.separable_conv_point = SeparableConv2d_pointwise(64, 64, depth = 1, kernel_size= (1,8))

        # self.fc2 = torch.nn.Linear(495,495)
        self.b2 = nn.BatchNorm2d(32*2, momentum=0.99)
        # self.pool1 = nn.MaxPool2d((1, 151), stride=(1, 45))  # spatial activation of 100ms and with a stride of 13ms
        # self.pool1 = nn.AvgPool2d((1, 107), stride=(1, 45))  # spatial activation of 100ms and with a stride of 13ms
        self.pool1 = nn.AvgPool2d((1, self.pool_window), stride=(1, self.stride_window))  # spatial activation of 100ms and with a stride of 13ms

        self.dropout1 = torch.nn.Dropout(p=dropout)
        self.fc3 = torch.nn.Linear(32*2*self.output_size,1)
        self.fc4 =  torch.nn.Linear(32*2*self.output_size,1)
        # self.fc5 = torch.nn.Linear(20, 1)
        # self.LeakyRelu = nn.LeakyReLU(0.2)
        self.gradients = None

    # #
    # #
    # # GLobal Average Pooling
    # ### Global Average Pooling
    #     self.gap = torch.nn.AdaptiveAvgPool2d((98,1))
    #
    #     ### Fully Connected Multi-Layer Perceptron (FC-MLP)
    #     self.mlp = torch.nn.Sequential(
    #         torch.nn.Linear(98, 98 // 8, bias=False),
    #         torch.nn.ReLU(inplace=True),
    #         torch.nn.Linear(98 // 8, 98, bias=False),
    #         torch.nn.Sigmoid()
    #     )
        # GLobal Average Pooling
        ### Global Average Pooling
        self.gap0 = torch.nn.AdaptiveAvgPool2d(1)

        ### Fully Connected Multi-Layer Perceptron (FC-MLP)
        self.mlp0 = torch.nn.Sequential(
            torch.nn.Linear(64, 64 // 4, bias=False),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(64 // 4, 64, bias=False),
            torch.nn.Sigmoid()
        )



    # hook for the gradients of the activations
    def activations_filterhook(self, grad):
        self.gradients_filter = grad
    def activations_temporalhook(self, grad):
        self.gradients_temp = grad
    def forward(self, x):
        batch_n = x.shape[0]
        x = x.view(batch_n, 1, self.num_chan,self.t_length).cuda()
        x = self.b0(x.cuda())
        x = torch.squeeze(x.cuda())
        if batch_n > 1:
            x = torch.squeeze(x.cuda())
        else:
            x = x.view(batch_n,self.num_chan,self.t_length)
        x = self.sinc_cnn2d(x)

        # # start attention
        # b, c, h, _ = x0.size()
        # y1 = self.gap(x0).view(b, c,h)
        # y1 = self.mlp(y1).view(b, c, h, 1)
        # x1 = x0 * y1.expand_as(x0)
        # # end attention

        h = x.register_hook(self.activations_filterhook)
        score0 = self.b(x)
        score0 = self.separable_conv(score0) # output is [n, 64,1,1870)
        # score = self.separable_conv_point(score) # output is [n, 64,1,1870)


        # start attention
        b, c, _, _ = score0.size()
        y0 = self.gap0(score0).view(b, c)
        y0 = self.mlp0(y0).view(b, c, 1, 1)
        x0 = score0 * y0.expand_as(score0)
        # end attention



        score = self.b2(x0)
        score = F.relu(score)
        # score = F.elu(score,alpha=1)
        score = self.pool1(score)
        score = self.dropout1(score)  # output is [n, 64,1,,37)
        h2 = score.register_hook(self.activations_temporalhook)
        score2 = score.view(-1,32*2*self.output_size)  # output [batch size, 64*17)
        score3= F.relu(self.fc3(score2))# output is [batch size, 1]


        # score4 = self.fc4(score2)
        score4 = F.sigmoid(self.fc4(score2))    # right now uses sigmoid
        score4 = torch.exp(score4)
        # score4 = torch.clamp(score4, 0.37, 2.65)
        # score4 = F.relu(self.fc5(score4))
        # score5 = torch.exp(score4)
        # score4 = torch.max(torch.zeros_like(self.fc4(score2))+0.2, score4)
        # score4 = torch.clamp(score4, min = 0.2, max = 3)
        # score5 = torch.mean(score4,axis=0)
        return -score3,score4,(score0,y0)

    def get_activations_gradient_filter(self):
        return self.gradients_filter
    def get_activations_gradient_temp(self):
        return self.gradients_temp






class Sinc_Conv2d_choice(nn.Module):
    # class global variables
    # filter_length = 251
    filter_length = 131
    t_length = 500
    pool_window = int(np.rint(((t_length- filter_length +1) * 250)/ (t_length*2)))
    stride_window =  int(np.rint(((t_length- filter_length +1) * 100)/ (t_length*2)))
    if pool_window % 2 == 0 :
        pool_window -= 1
    if stride_window % 2 == 0 :
        stride_window -= 1
    output_size = int(np.floor(((t_length- filter_length +1) - pool_window)/stride_window +1))
    num_chan = 98

    def __init__(self, dropout):
        super(Sinc_Conv2d_choice, self).__init__()
        self.b0 = nn.BatchNorm2d(1, momentum=0.99)
        self.sinc_cnn2d = sinc_conv(32, self.filter_length, 500)
        # self.fc1 = torch.nn.Linear(495,495)
        self.b =  nn.BatchNorm2d(32, momentum=0.99)
        self.separable_conv = SeparableConv2d(32, 32, depth = 3, kernel_size= (self.num_chan,1))
        self.separable_conv_point = SeparableConv2d_pointwise(32*3, 32*3, depth = 1, kernel_size= (1,8))

        # self.fc2 = torch.nn.Linear(495,495)
        self.b2 = nn.BatchNorm2d(32*3, momentum=0.99)
        # self.pool1 = nn.MaxPool2d((1, 151), stride=(1, 45))  # spatial activation of 100ms and with a stride of 13ms
        # self.pool1 = nn.AvgPool2d((1, 107), stride=(1, 45))  # spatial activation of 100ms and with a stride of 13ms
        self.pool1 = nn.AvgPool2d((1, self.pool_window), stride=(1, self.stride_window))  # spatial activation of 100ms and with a stride of 13ms

        self.dropout1 = torch.nn.Dropout(p=dropout)
        self.fc3 = torch.nn.Linear(32*3*self.output_size,1)
        self.fc4 =  torch.nn.Linear(32*3*self.output_size,1)
        self.fc5 =  torch.nn.Linear(32*3*self.output_size,1)

        # self.fc5 = torch.nn.Linear(20, 1)
        # self.LeakyRelu = nn.LeakyReLU(0.2)
        self.gradients = None

    # #
    # #
    # # GLobal Average Pooling
    # ### Global Average Pooling
    #     self.gap = torch.nn.AdaptiveAvgPool2d((98,1))
    #
    #     ### Fully Connected Multi-Layer Perceptron (FC-MLP)
    #     self.mlp = torch.nn.Sequential(
    #         torch.nn.Linear(98, 98 // 8, bias=False),
    #         torch.nn.ReLU(inplace=True),
    #         torch.nn.Linear(98 // 8, 98, bias=False),
    #         torch.nn.Sigmoid()
    #     )
    #     GLobal Average Pooling
        ## Global Average Pooling
        self.gap0 = torch.nn.AdaptiveAvgPool2d(1)

        ### Fully Connected Multi-Layer Perceptron (FC-MLP)
        self.mlp0 = torch.nn.Sequential(
            torch.nn.Linear(32, 32 // 6, bias=False),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(32 // 6, 32, bias=False),
            torch.nn.Sigmoid()
        )



    # hook for the gradients of the activations
    def activations_filterhook(self, grad):
        self.gradients_filter = grad
    def activations_temporalhook(self, grad):
        self.gradients_temp = grad
    def forward(self, x):
        batch_n = x.shape[0]
        x = x.view(batch_n, 1, self.num_chan,self.t_length).cuda()
        x = self.b0(x.cuda())
        x = torch.squeeze(x.cuda())
        if batch_n > 1:
            x = torch.squeeze(x.cuda())
        else:
            x = x.view(batch_n,self.num_chan,self.t_length)
        x0 = self.sinc_cnn2d(x)
        h = x0.register_hook(self.activations_filterhook)

        # start attention
        b, c, _, _ = x0.size()
        y1 = self.gap0(x0).view(b, c)
        y1 = self.mlp0(y1).view(b, c, 1, 1)
        x1 = x0 * y1.expand_as(x0)
        # end attention

        score0 = self.b(x1)
        score0 = self.separable_conv(score0) # output is [n, 64,1,1870)
        score = self.separable_conv_point(score0) # output is [n, 64,1,1870)

        #
        # # start attention
        # b, c, _, _ = score0.size()
        # y0 = self.gap0(score0).view(b, c)
        # y0 = self.mlp0(y0).view(b, c, 1, 1)
        # x0 = score0 * y0.expand_as(score0)
        # # end attention


        score = self.b2(score)
        score = F.relu(score)
        # score = F.elu(score,alpha=1)
        score = self.pool1(score)
        score = self.dropout1(score)  # output is [n, 64,1,,37)
        h2 = score.register_hook(self.activations_temporalhook)
        score2 = score.view(-1,32*3*self.output_size)  # output [batch size, 64*17)
        score3= self.fc3(score2)# output is [batch size, 1]


        # score4 = self.fc4(score2)
        score4 = F.softplus(self.fc4(score2))
        score5 = F.sigmoid(self.fc5(score2))
        # right now uses sigmoid


        # score4 = torch.exp(score4)
        # score4 = torch.clamp(score4, 0.37, 2.65)
        # score4 = F.relu(self.fc5(score4))
        # score5 = torch.exp(score4)
        # score4 = torch.max(torch.zeros_like(self.fc4(score2))+0.2, score4)
        # score4 = torch.clamp(score4, min = 0.2, max = 3)
        # score5 = torch.mean(score4,axis=0)
        return score3,score4,score5

    def get_activations_gradient_filter(self):
        return self.gradients_filter
    def get_activations_gradient_temp(self):
        return self.gradients_temp



#
#
class Sinc_Conv2d_attention_pre(nn.Module):
    # class global variables
    # filter_length = 251
    filter_length = 131
    t_length = 500
    pool_window = int(np.rint(((t_length- filter_length +1) * 250)/ (t_length*2)))
    stride_window =  int(np.rint(((t_length- filter_length +1) * 100)/ (t_length*2)))
    if pool_window % 2 == 0 :
        pool_window -= 1
    if stride_window % 2 == 0 :
        stride_window -= 1
    output_size = int(np.floor(((t_length- filter_length +1) - pool_window)/stride_window +1))
    num_chan = 98

    def __init__(self, dropout):
        super(Sinc_Conv2d_attention_pre, self).__init__()
        self.b0 = nn.BatchNorm2d(1, momentum=0.99)
        self.sinc_cnn2d = sinc_conv(32, self.filter_length, 500)
        # self.fc1 = torch.nn.Linear(495,495)
        self.b =  nn.BatchNorm2d(32, momentum=0.99)
        self.separable_conv = SeparableConv2d(32, 32, depth = 3, kernel_size= (self.num_chan,1))
        self.separable_conv_point = SeparableConv2d_pointwise(32*3, 32*3, depth = 1, kernel_size= (1,8))

        # self.fc2 = torch.nn.Linear(495,495)
        self.b2 = nn.BatchNorm2d(32*3, momentum=0.99)
        # self.pool1 = nn.MaxPool2d((1, 151), stride=(1, 45))  # spatial activation of 100ms and with a stride of 13ms
        # self.pool1 = nn.AvgPool2d((1, 107), stride=(1, 45))  # spatial activation of 100ms and with a stride of 13ms
        self.pool1 = nn.AvgPool2d((1, self.pool_window), stride=(1, self.stride_window))  # spatial activation of 100ms and with a stride of 13ms

        self.dropout1 = torch.nn.Dropout(p=dropout)
        self.fc3 = torch.nn.Linear(32*3*self.output_size,1)
        self.fc4 =  torch.nn.Linear(32*3*self.output_size,1)
        # self.fc5 = torch.nn.Linear(20, 1)
        # self.LeakyRelu = nn.LeakyReLU(0.2)
        self.gradients = None

    # #
    # #
    # GLobal Average Pooling
    ### Global Average Pooling
        # self.gap = torch.nn.AdaptiveAvgPool2d((98,1))
        #
        # ### Fully Connected Multi-Layer Perceptron (FC-MLP)
        # self.mlp = torch.nn.Sequential(
        #     torch.nn.Linear(96, 96 // 8, bias=False),
        #     torch.nn.ReLU(inplace=True),
        #     torch.nn.Linear(96 // 8, 96, bias=False),
        #     torch.nn.Sigmoid()
        # )
        # GLobal Average Pooling
        ## Global Average Pooling
        self.gap0 = torch.nn.AdaptiveAvgPool2d(1)

        ### Fully Connected Multi-Layer Perceptron (FC-MLP)
        self.mlp0 = torch.nn.Sequential(
            torch.nn.Linear(32*3, 32*3 // 6, bias=False),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(32*3 // 6, 32*3, bias=False),
            torch.nn.Sigmoid()
        )


    # hook for the gradients of the activations
    def activations_filterhook(self, grad):
        self.gradients_filter = grad
    def activations_temporalhook(self, grad):
        self.gradients_temp = grad
    def forward(self, x):
        batch_n = x.shape[0]
        x = x.view(batch_n, 1, self.num_chan,self.t_length).cuda()
        x = self.b0(x.cuda())
        x = torch.squeeze(x.cuda())
        if batch_n > 1:
            x = torch.squeeze(x.cuda())
        else:
            x = x.view(batch_n,self.num_chan,self.t_length)
        x0 = self.sinc_cnn2d(x)
        h = x0.register_hook(self.activations_filterhook)
        #
        # # start attention
        # b, c, _, _ = x0.size()
        # y1 = self.gap0(x0).view(b, c)
        # y1 = self.mlp0(y1).view(b, c, 1, 1)
        # x1 = x0 * y1.expand_as(x0)
        # # end attention

        score0 = self.b(x0)
        score0 = self.separable_conv(score0) # output is [n, 64,1,1870)
        score0 = self.separable_conv_point(score0) # output is [n, 64,1,1870)

        #
        # start attention
        b, c, _, _ = score0.size()
        y0 = self.gap0(score0).view(b, c)
        y0 = self.mlp0(y0).view(b, c, 1, 1)
        score_new = score0 * y0.expand_as(score0)
        # end attention


        score = self.b2(score_new)
        score = F.relu(score)
        # score = F.elu(score,alpha=1)
        score = self.pool1(score)
        score = self.dropout1(score)  # output is [n, 64,1,,37)
        h2 = score.register_hook(self.activations_temporalhook)
        score2 = score.view(-1,32*3*self.output_size)  # output [batch size, 64*17)
        score3= self.fc3(score2)# output is [batch size, 1]


        # score4 = self.fc4(score2)
        # score4 = F.sigmoid(self.fc4(score2)) *3    # right now uses sigmoid
        score4 = F.softplus(self.fc4(score2))

        return score3,score4,(score_new,y0)

    def get_activations_gradient_filter(self):
        return self.gradients_filter
    def get_activations_gradient_temp(self):
        return self.gradients_temp




class Sinc_Conv2d_attention_prenew(nn.Module):
    # class global variables
    # filter_length = 251
    filter_length = 131
    t_length = 500
    pool_window = int(np.rint(((t_length- filter_length +1) * 250)/ (t_length*2)))
    stride_window =  int(np.rint(((t_length- filter_length +1) * 100)/ (t_length*2)))
    if pool_window % 2 == 0 :
        pool_window -= 1
    if stride_window % 2 == 0 :
        stride_window -= 1
    output_size = int(np.floor(((t_length- filter_length +1) - pool_window)/stride_window +1))
    num_chan = 98

    def __init__(self, dropout):
        super(Sinc_Conv2d_attention_prenew, self).__init__()
        self.b0 = nn.BatchNorm2d(1, momentum=0.99)
        self.sinc_cnn2d = sinc_conv(32, self.filter_length, 500)
        # self.fc1 = torch.nn.Linear(495,495)
        self.b =  nn.BatchNorm2d(32, momentum=0.99)
        self.separable_conv = SeparableConv2d(32, 32, depth = 3, kernel_size= (self.num_chan,1))
        self.separable_conv_point = SeparableConv2d_pointwise(32*3, 32*3, depth = 1, kernel_size= (1,8))

        # self.fc2 = torch.nn.Linear(495,495)
        self.b2 = nn.BatchNorm2d(32*3, momentum=0.99)
        # self.pool1 = nn.MaxPool2d((1, 151), stride=(1, 45))  # spatial activation of 100ms and with a stride of 13ms
        # self.pool1 = nn.AvgPool2d((1, 107), stride=(1, 45))  # spatial activation of 100ms and with a stride of 13ms
        self.pool1 = nn.AvgPool2d((1, self.pool_window), stride=(1, self.stride_window))  # spatial activation of 100ms and with a stride of 13ms

        self.dropout1 = torch.nn.Dropout(p=dropout)
        self.fc3 = torch.nn.Linear(32*3*self.output_size,1)
        self.fc4 =  torch.nn.Linear(32*3*self.output_size,1)
        # self.fc5 = torch.nn.Linear(20, 1)
        # self.LeakyRelu = nn.LeakyReLU(0.2)
        self.gradients = None

    # #
    # #
    # GLobal Average Pooling
    ### Global Average Pooling
        # self.gap = torch.nn.AdaptiveAvgPool2d((98,1))
        #
        # ### Fully Connected Multi-Layer Perceptron (FC-MLP)
        # self.mlp = torch.nn.Sequential(
        #     torch.nn.Linear(96, 96 // 8, bias=False),
        #     torch.nn.ReLU(inplace=True),
        #     torch.nn.Linear(96 // 8, 96, bias=False),
        #     torch.nn.Sigmoid()
        # )
        # GLobal Average Pooling
        ## Global Average Pooling
        self.gap0 = torch.nn.AdaptiveAvgPool2d(1)

        ### Fully Connected Multi-Layer Perceptron (FC-MLP)
        self.mlp0 = torch.nn.Sequential(
            torch.nn.Linear(32, 32 // 6, bias=False),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(32 // 6, 32, bias=False),
            torch.nn.Sigmoid()
        )


    # hook for the gradients of the activations
    def activations_filterhook(self, grad):
        self.gradients_filter = grad
    def activations_temporalhook(self, grad):
        self.gradients_temp = grad
    def forward(self, x):
        batch_n = x.shape[0]
        x = x.view(batch_n, 1, self.num_chan,self.t_length).cuda()
        x = self.b0(x.cuda())
        x = torch.squeeze(x.cuda())
        if batch_n > 1:
            x = torch.squeeze(x.cuda())
        else:
            x = x.view(batch_n,self.num_chan,self.t_length)
        x0 = self.sinc_cnn2d(x)
        # h = x0.register_hook(self.activations_filterhook)
        #
        # # start attention
        # b, c, _, _ = x0.size()
        # y1 = self.gap0(x0).view(b, c)
        # y1 = self.mlp0(y1).view(b, c, 1, 1)
        # x1 = x0 * y1.expand_as(x0)
        # # end attention
        # start attention
        b, c, _, _ = x0.size()
        y0 = self.gap0(x0).view(b, c)
        y0 = self.mlp0(y0).view(b, c, 1, 1)
        score_new = x0 * y0.expand_as(x0)

        # end attention

        score0_ = self.b(score_new)
        score0 = self.separable_conv(score0_) # output is [n, 64,1,1870)
        score0 = self.separable_conv_point(score0) # output is [n, 64,1,1870)

        #



        score = self.b2(score0)
        score = F.relu(score)
        # score = F.elu(score,alpha=1)
        score = self.pool1(score)
        score = self.dropout1(score)  # output is [n, 64,1,,37)
        # h2 = score.register_hook(self.activations_temporalhook)
        score2 = score.view(-1,32*3*self.output_size)  # output [batch size, 64*17)
        score3= self.fc3(score2)# output is [batch size, 1]


        # score4 = self.fc4(score2)
        # score4 = F.sigmoid(self.fc4(score2)) *3    # right now uses sigmoid
        score4 = F.softplus(self.fc4(score2))

        return score3,score4,(score2,y0,score0_)

    def get_activations_gradient_filter(self):
        return self.gradients_filter
    def get_activations_gradient_temp(self):
        return self.gradients_temp




class Sinc_Conv2d_ddm_2param_reduced(nn.Module):
    # class global variables
    # filter_length = 251
    filter_length = 131
    t_length = 500
    pool_window = int(np.rint(((t_length- filter_length +1) * 250)/ (t_length*2)))
    stride_window =  int(np.rint(((t_length- filter_length +1) * 100)/ (t_length*2)))
    if pool_window % 2 == 0 :
        pool_window -= 1
    if stride_window % 2 == 0 :
        stride_window -= 1
    output_size = int(np.floor(((t_length- filter_length +1) - pool_window)/stride_window +1))
    num_chan = 98

    def __init__(self, dropout):
        super(Sinc_Conv2d_ddm_2param_reduced, self).__init__()
        self.b0 = nn.BatchNorm2d(1, momentum=0.99)
        self.sinc_cnn2d = sinc_conv(32, self.filter_length, 500)
        self.b =  nn.BatchNorm2d(32, momentum=0.99)
        self.fc1_spatial_0 = torch.nn.Linear(98,1)
        self.fc1_spatial_1 = torch.nn.Linear(98,1)


        # self.separable_conv = SeparableConv2d(32, 32, depth = 2, kernel_size= (self.num_chan,1))
        # self.separable_conv_point = SeparableConv2d_pointwise(64, 64, depth = 1, kernel_size= (1,8))
    #     self.fc1 = torch.nn.Linear((32,))
    #
        # self.fc2 = torch.nn.Linear(495,495)
        self.b2_0 = nn.BatchNorm2d(32, momentum=0.99)
        self.b2_1 = nn.BatchNorm2d(32, momentum=0.99)

        # self.pool1 = nn.MaxPool2d((1, 151), stride=(1, 45))  # spatial activation of 100ms and with a stride of 13ms
        # self.pool1 = nn.AvgPool2d((1, 107), stride=(1, 45))  # spatial activation of 100ms and with a stride of 13ms
        # self.conv_0 = nn.Conv1d(32,32,kernel_size=(1,10))
        # self.conv_1 = nn.Conv1d(32,32,kernel_size=(1,10))

        self.pool1_0 = nn.AvgPool2d((1, self.pool_window), stride=(1, self.stride_window))  # spatial activation of 100ms and with a stride of 13ms
        self.pool1_1 = nn.AvgPool2d((1, self.pool_window), stride=(1, self.stride_window))  # spatial activation of 100ms and with a stride of 13ms

        self.dropout1_0 = torch.nn.Dropout(p=dropout)
        self.dropout1_1 = torch.nn.Dropout(p=dropout)

        self.fc3 = torch.nn.Linear(32*self.output_size,1)
        self.fc4 =  torch.nn.Linear(32*self.output_size,1)
        # self.fc5 = torch.nn.Linear(20, 1)
        # self.LeakyRelu = nn.LeakyReLU(0.2)
        self.gradients = None

    # #
    # #
    # GLobal Average Pooling
    ### Global Average Pooling
        # self.gap = torch.nn.AdaptiveAvgPool2d(1)
        #
        # ### Fully Connected Multi-Layer Perceptron (FC-MLP)
        # self.mlp = torch.nn.Sequential(
        #     torch.nn.Linear(32, 32 // 2, bias=False),
        #     torch.nn.ReLU(inplace=True),
        #     torch.nn.Linear(32 // 2, 32, bias=False),
        #     torch.nn.Sigmoid()
        # )
    #
    #
    #
    # hook for the gradients of the activations
    def activations_filterhook(self, grad):
        self.gradients_filter = grad
    def activations_temporalhook(self, grad):
        self.gradients_temp = grad

    def activations_spatialhook(self, grad):
        self.gradients_spatial = grad
    def forward(self, x):
        batch_n = x.shape[0]
        x = x.view(batch_n, 1, self.num_chan,self.t_length).cuda()
        x = self.b0(x.cuda())
        x = torch.squeeze(x.cuda())
        if batch_n > 1:
            x = torch.squeeze(x.cuda())
        else:
            x = x.view(batch_n,self.num_chan,self.t_length)
        x = self.sinc_cnn2d(x)

        h = x.register_hook(self.activations_filterhook)
        score = self.b(x)
        score = score.transpose(2,3)
        score0 = self.fc1_spatial_0(score)
        score1 = self.fc1_spatial_1(score)
        h_spatial_0 = score0.register_hook(self.activations_spatialhook)
        h_spatial_1 = score1.register_hook(self.activations_spatialhook)

        score0 = score0.transpose(2,3)
        score1 = score1.transpose(2,3)

    #     # score = self.separable_conv(score) # output is [n, 64,1,1870)
    #     # score = self.separable_conv_point(score) # output is [n, 64,1,1870)
    #
        # # start attention
        # b, c, _, _ = score.size()
        # y = self.gap(score).view(b, c)
        # y = self.mlp(y).view(b, c, 1, 1)
        # score = score * y.expand_as(score)
        # # end attention
    # #
    #
        score0 = self.b2_0(score0)
        score1 = self.b2_1(score1)
        score0= F.relu(score0)
        score1 = F.relu(score1)

        # score0 = self.conv_0(score0)
        # score1 = self.conv_1(score1)

        # score = F.elu(score,alpha=1)
        # score0 = F.relu(score0)
        # score1 = F.relu(score1)

        score0 = self.pool1_0(score0)
        score1 = self.pool1_1(score1)

        score0 = self.dropout1_0(score0)  # output is [n, 64,1,,37)
        score1 = self.dropout1_1(score1)  # output is [n, 64,1,,37)

        h2_0 = score0.register_hook(self.activations_temporalhook)
        h2_1 = score1.register_hook(self.activations_temporalhook)

        score2_0 = score0.view(-1,32*self.output_size)  # output [batch size, 64*17)
        score2_1 = score1.view(-1,32*self.output_size)  # output [batch size, 64*17)

        score3= self.fc3(score2_0)# output is [batch size, 1]


        # score4 = self.fc4(score2)
        score4 = F.relu6(self.fc4(score2_1),3)    # right now uses sigmoid
        # score4 = F(score4)
        # score4 = torch.clamp(score4, 0.37, 2.65)
        # score4 = F.relu(self.fc5(score4))
        # score5 = torch.exp(score4)
        # score4 = torch.max(torch.zeros_like(self.fc4(score2))+0.2, score4)
        # score4 = torch.clamp(score4, min = 0.2, max = 3)
        # score5 = torch.mean(score4,axis=0)
        return score3, score4

    def get_activations_gradient_filter(self):
        return self.gradients_filter
    def get_activations_gradient_temp(self):
        return self.gradients_temp






class Sinc_Conv2d_ddm_2param_reduced(nn.Module):
    # class global variables
    # filter_length = 251
    filter_length = 131
    t_length = 500
    pool_window = int(np.rint(((t_length- filter_length +1) * 250)/ (t_length*2)))
    stride_window =  int(np.rint(((t_length- filter_length +1) * 100)/ (t_length*2)))
    if pool_window % 2 == 0 :
        pool_window -= 1
    if stride_window % 2 == 0 :
        stride_window -= 1
    output_size = int(np.floor(((t_length- filter_length +1) - pool_window)/stride_window +1))
    num_chan = 98

    def __init__(self, dropout):
        super(Sinc_Conv2d_ddm_2param_reduced, self).__init__()
        self.b0 = nn.BatchNorm2d(1, momentum=0.99)
        self.sinc_cnn2d = sinc_conv(32, self.filter_length, 500)
        self.b =  nn.BatchNorm2d(32, momentum=0.99)
        # self.fc1_spatial_0 = torch.nn.Linear(98,1)
        # self.fc1_spatial_1 = torch.nn.Linear(98,1)
        # self.conv_spatial_0 = nn.Conv1d(32,32,kernel_size=1,)
        # self.conv_spatial_1


        # self.separable_conv = SeparableConv2d(32, 32, depth = 2, kernel_size= (self.num_chan,1))
        # self.separable_conv_point = SeparableConv2d_pointwise(64, 64, depth = 1, kernel_size= (1,8))
    #     self.fc1 = torch.nn.Linear((32,))
    #
        # self.fc2 = torch.nn.Linear(495,495)
        self.b2_0 = nn.BatchNorm2d(32, momentum=0.99)
        self.b2_1 = nn.BatchNorm2d(32, momentum=0.99)

        # self.pool1 = nn.MaxPool2d((1, 151), stride=(1, 45))  # spatial activation of 100ms and with a stride of 13ms
        # self.pool1 = nn.AvgPool2d((1, 107), stride=(1, 45))  # spatial activation of 100ms and with a stride of 13ms
        # self.conv_0 = nn.Conv1d(32,32,kernel_size=(1,10))
        # self.conv_1 = nn.Conv1d(32,32,kernel_size=(1,10))

        self.pool1_0 = nn.AvgPool2d((1, self.pool_window), stride=(1, self.stride_window))  # spatial activation of 100ms and with a stride of 13ms
        self.pool1_1 = nn.AvgPool2d((1, self.pool_window), stride=(1, self.stride_window))  # spatial activation of 100ms and with a stride of 13ms

        self.dropout1_0 = torch.nn.Dropout(p=dropout)
        self.dropout1_1 = torch.nn.Dropout(p=dropout)

        self.fc3 = torch.nn.Linear(32*self.output_size,1)
        self.fc4 =  torch.nn.Linear(32*self.output_size,1)
        # self.fc5 = torch.nn.Linear(20, 1)
        # self.LeakyRelu = nn.LeakyReLU(0.2)
        self.gradients = None

    # #
    # #
    # GLobal Average Pooling
    ### Global Average Pooling
        # self.gap = torch.nn.AdaptiveAvgPool2d(1)
        #
        # ### Fully Connected Multi-Layer Perceptron (FC-MLP)
        # self.mlp = torch.nn.Sequential(
        #     torch.nn.Linear(32, 32 // 2, bias=False),
        #     torch.nn.ReLU(inplace=True),
        #     torch.nn.Linear(32 // 2, 32, bias=False),
        #     torch.nn.Sigmoid()
        # )
    #
    #
    #
    # hook for the gradients of the activations
    def activations_filterhook(self, grad):
        self.gradients_filter = grad
    def activations_temporalhook(self, grad):
        self.gradients_temp = grad

    def activations_spatialhook(self, grad):
        self.gradients_spatial = grad
    def forward(self, x):
        batch_n = x.shape[0]
        x = x.view(batch_n, 1, self.num_chan,self.t_length).cuda()
        x = self.b0(x.cuda())
        x = torch.squeeze(x.cuda())
        if batch_n > 1:
            x = torch.squeeze(x.cuda())
        else:
            x = x.view(batch_n,self.num_chan,self.t_length)
        x = self.sinc_cnn2d(x)

        h = x.register_hook(self.activations_filterhook)
        score = self.b(x)
        score = score.transpose(2,3)
        score0 = self.fc1_spatial_0(score)
        score1 = self.fc1_spatial_1(score)
        h_spatial_0 = score0.register_hook(self.activations_spatialhook)
        h_spatial_1 = score1.register_hook(self.activations_spatialhook)

        score0 = score0.transpose(2,3)
        score1 = score1.transpose(2,3)

    #     # score = self.separable_conv(score) # output is [n, 64,1,1870)
    #     # score = self.separable_conv_point(score) # output is [n, 64,1,1870)
    #
        # # start attention
        # b, c, _, _ = score.size()
        # y = self.gap(score).view(b, c)
        # y = self.mlp(y).view(b, c, 1, 1)
        # score = score * y.expand_as(score)
        # # end attention
    # #
    #
        score0 = self.b2_0(score0)
        score1 = self.b2_1(score1)
        score0= F.relu(score0)
        score1 = F.relu(score1)

        # score0 = self.conv_0(score0)
        # score1 = self.conv_1(score1)

        # score = F.elu(score,alpha=1)
        # score0 = F.relu(score0)
        # score1 = F.relu(score1)

        score0 = self.pool1_0(score0)
        score1 = self.pool1_1(score1)

        score0 = self.dropout1_0(score0)  # output is [n, 64,1,,37)
        score1 = self.dropout1_1(score1)  # output is [n, 64,1,,37)

        h2_0 = score0.register_hook(self.activations_temporalhook)
        h2_1 = score1.register_hook(self.activations_temporalhook)

        score2_0 = score0.view(-1,32*self.output_size)  # output [batch size, 64*17)
        score2_1 = score1.view(-1,32*self.output_size)  # output [batch size, 64*17)

        score3= self.fc3(score2_0)# output is [batch size, 1]


        # score4 = self.fc4(score2)
        score4 = F.sigmoid(self.fc4(score2_1))    # right now uses sigmoid
        score4 = torch.exp(score4)
        # score4 = torch.clamp(score4, 0.37, 2.65)
        # score4 = F.relu(self.fc5(score4))
        # score5 = torch.exp(score4)
        # score4 = torch.max(torch.zeros_like(self.fc4(score2))+0.2, score4)
        # score4 = torch.clamp(score4, min = 0.2, max = 3)
        # score5 = torch.mean(score4,axis=0)
        return score3, score4

    def get_activations_gradient_filter(self):
        return self.gradients_filter
    def get_activations_gradient_temp(self):
        return self.gradients_temp




