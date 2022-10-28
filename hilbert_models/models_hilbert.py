# Created on 8/15/22 at 3:14 PM 

# Author: Jenny Sun

'''this scrip contains models that fit drift and boundary, single boundary

the variats include:
 - split from FC
 - split from Spatial
 - split from Temporal
 - one bound with classification
 - plus attention '''

# Created on 10/12/21 at 10:50 PM

# Author: Jenny Sun
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import sys
from torch.autograd import Variable
import math
from torch.autograd import Function
from torch import fft
from typing import List, Sequence, Union
import complexPyTorch
from complexPyTorch.complexLayers import ComplexBatchNorm2d, ComplexConv2d, ComplexLinear
from complexPyTorch.complexFunctions import complex_relu, complex_max_pool2d


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




class HilbertTransform(nn.Module):
    """
    Determine the analytical signal of a Tensor along a particular axis.
    Args:
        axis: Axis along which to apply Hilbert transform. Default 2 (first spatial dimension).
        n: Number of Fourier components (i.e. FFT size). Default: ``x.shape[axis]``.
    """

    def __init__(self, axis: int = 2, n: Union[int, None] = None) -> None:

        super().__init__()
        self.axis = axis
        self.n = n

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor or array-like to transform. Must be real and in shape ``[Batch, chns, spatial1, spatial2, ...]``.
        Returns:
            torch.Tensor: Analytical signal of ``x``, transformed along axis specified in ``self.axis`` using
            FFT of size ``self.N``. The absolute value of ``x_ht`` relates to the envelope of ``x`` along axis ``self.axis``.
        """

        # Make input a real tensor
        x = torch.as_tensor(x, device=x.device if isinstance(x, torch.Tensor) else None)
        if torch.is_complex(x):
            raise ValueError("x must be real.")
        x = x.to(dtype=torch.float)

        if (self.axis < 0) or (self.axis > len(x.shape) - 1):
            raise ValueError(f"Invalid axis for shape of x, got axis {self.axis} and shape {x.shape}.")

        n = x.shape[self.axis] if self.n is None else self.n
        if n <= 0:
            raise ValueError("N must be positive.")
        x = torch.as_tensor(x, dtype=torch.complex64)
        # Create frequency axis
        f = torch.cat(
            [
                torch.true_divide(torch.arange(0, (n - 1) // 2 + 1, device=x.device), float(n)),
                torch.true_divide(torch.arange(-(n // 2), 0, device=x.device), float(n)),
            ]
        )
        xf = fft.fft(x, n=n, dim=self.axis)
        # Create step function
        u = torch.heaviside(f, torch.tensor([0.5], device=f.device))
        u = torch.as_tensor(u, dtype=x.dtype, device=u.device)
        new_dims_before = self.axis
        new_dims_after = len(xf.shape) - self.axis - 1
        for _ in range(new_dims_before):
            u.unsqueeze_(0)
        for _ in range(new_dims_after):
            u.unsqueeze_(-1)

        ht = fft.ifft(xf * 2 * u, dim=self.axis)

        # Apply transform
        return torch.as_tensor(ht, device=ht.device, dtype=ht.dtype)



class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, depth, kernel_size, bias=False):
        super(SeparableConv2d, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, out_channels*depth, kernel_size=kernel_size, groups=in_channels, bias=bias)
        # self.pointwise = nn.Conv2d(out_channels*depth, out_channels*depth, kernel_size=[1,1], bias=bias)

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










class SincDriftBoundAttChoice_hilbert(nn.Module):
    '''predicts drift and boundary
    attention layer after sinc and after conv'''

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
    # output_size = 186
    num_chan = 98

    def __init__(self, dropout):
        super(SincDriftBoundAttChoice_hilbert, self).__init__()
        self.b0 = nn.BatchNorm2d(1, momentum=0.99)
        self.sinc_cnn2d = sinc_conv(32, self.filter_length, 500)
        self.sinc_cnn2d_choice = sinc_conv(32, self.filter_length, 500)

        self.b =  nn.BatchNorm2d(32, momentum=0.99)
        self.b_choice = nn.BatchNorm2d(32, momentum=0.99)

        self.separable_conv = SeparableConv2d(32, 32, depth = 3, kernel_size= (self.num_chan,1))
        self.separable_conv_choice = SeparableConv2d(32, 32, depth = 3, kernel_size= (self.num_chan,1))

        # self.separable_conv_point = SeparableConv2d_pointwise(32*3, 32*3, depth = 1, kernel_size= (1,8))

        # self.fc2 = torch.nn.Linear(495,495)
        self.b2 = nn.BatchNorm2d(32*3, momentum=0.99)
        self.b2_choice = nn.BatchNorm2d(32*3, momentum=0.99)

        # self.pool1 = nn.MaxPool2d((1, 151), stride=(1, 45))  # spatial activation of 100ms and with a stride of 13ms
        # self.pool1 = nn.AvgPool2d((1, 107), stride=(1, 45))  # spatial activation of 100ms and with a stride of 13ms
        self.hilbert = HilbertTransform(axis=3)
        self.hilbert_choice =HilbertTransform(axis = 3)
        self.pool1 = nn.AvgPool2d((1, self.pool_window), stride=(1, self.stride_window))  # spatial activation of 100ms and with a stride of 13ms
        self.pool1_choice = nn.AvgPool2d((1, self.pool_window), stride=(1, self.stride_window))  # spatial activation of 100ms and with a stride of 13ms

        self.dropout1 = torch.nn.Dropout(p=dropout)
        self.dropout1_choice = torch.nn.Dropout(p=dropout)

        self.fc3 = torch.nn.Linear(32*3*self.output_size,1)
        self.fc4 =  torch.nn.Linear(32*3*self.output_size,1)
        self.fc_choice = torch.nn.Linear(32*3*self.output_size,1)

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

        self.gap0_choice = torch.nn.AdaptiveAvgPool2d(1)

        ### Fully Connected Multi-Layer Perceptron (FC-MLP)
        self.mlp0_choice = torch.nn.Sequential(
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
        x0_choice = self.sinc_cnn2d_choice(x)
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

        # start attention for choice
        b_choice, c_choice, _, _ = x0_choice.size()
        y0_choice = self.gap0_choice(x0_choice).view(b_choice, c_choice)
        y0_choice = self.mlp0_choice(y0_choice).view(b_choice, c_choice, 1, 1)
        score_new_choice = x0_choice * y0_choice.expand_as(x0_choice)

        # end attention



        # spatial convulation layer
        score0_ = self.b(score_new)
        score0 = self.separable_conv(score0_) # output is [n, 64,1,1870)
        # score0 = self.separable_conv_point(score0) # output is [n, 64,1,1870)

        # spatial convulation layer for choice
        score0_choice_ = self.b_choice(score_new_choice)
        score0_choice = self.separable_conv_choice(score0_choice_) # output is [n, 64,1,1870)

        score = self.b2(score0)
        score = torch.abs(self.hilbert(score))

        score = self.pool1(score)
        score = self.dropout1(score)  # output is [n, 64,1,,37)


        score_choice = self.b2_choice(score0_choice)
        score_choice = torch.abs(self.hilbert_choice(score_choice))

        score_choice = self.pool1_choice(score_choice)
        score_choice = self.dropout1_choice(score_choice)  # output is [n, 64,1,,37)


        # fully connected layer
        score2 = score.view(-1,32*3*self.output_size)  # output [batch size, 64*17)
        score3= self.fc3(score2)# output is [batch size, 1]

        score2_choice = score_choice.view(-1,32*3*self.output_size)
        score3_choice = self.fc_choice(score2_choice)   # choice
        score3_choice = F.sigmoid(score3_choice)
        # score4 = self.fc4(score2)
        # score4 = F.sigmoid(self.fc4(score2)) *3    # right now uses sigmoid
        score4 = F.softplus(self.fc4(score2))

        return score3,score4,(score2,y0,score0_), score3_choice,(score2_choice,y0_choice,score0_choice)

    def get_activations_gradient_filter(self):
        return self.gradients_filter
    def get_activations_gradient_temp(self):
        return self.gradients_temp





class SincHilbert_classify_hilbert(nn.Module):
    '''predicts drift and boundary
    attention layer after sinc and after conv'''

    # filter_length = 251
    filter_length = 131
    t_length = 500
    pool_window = int(np.rint(((t_length- filter_length +1) * 150)/ (t_length*2)))
    stride_window =  int(np.rint(((t_length- filter_length +1) * 50)/ (t_length*2)))
    if pool_window % 2 == 0 :
        pool_window -= 1
    if stride_window % 2 == 0 :
        stride_window -= 1
    output_size = int(np.floor(((t_length- filter_length +1) - pool_window)/stride_window +1))
    # output_size = 186
    num_chan = 98

    def __init__(self, dropout):
        super(SincHilbert_classify_hilbert, self).__init__()
        self.b0 = nn.BatchNorm2d(1, momentum=0.99)
        # self.sinc_cnn2d = sinc_conv(32, self.filter_length, 500)
        self.sinc_cnn2d_choice = sinc_conv(32*2, self.filter_length, 500)

        # self.b =  nn.BatchNorm2d(32, momentum=0.99)
        self.b_choice = nn.BatchNorm2d(32*2, momentum=0.99)

        # self.separable_conv = SeparableConv2d(32, 32, depth = 3, kernel_size= (self.num_chan,1))
        self.separable_conv_choice = SeparableConv2d(32*2, 32*2, depth = 1, kernel_size= (self.num_chan,1))

        # self.separable_conv_point = SeparableConv2d_pointwise(32*3, 32*3, depth = 1, kernel_size= (1,8))

        # self.fc2 = torch.nn.Linear(495,495)
        # self.b2 = nn.BatchNorm2d(32*3, momentum=0.99)
        self.b2_choice = nn.BatchNorm2d(32*2*1, momentum=0.99)

        # self.pool1 = nn.MaxPool2d((1, 151), stride=(1, 45))  # spatial activation of 100ms and with a stride of 13ms
        # self.pool1 = nn.AvgPool2d((1, 107), stride=(1, 45))  # spatial activation of 100ms and with a stride of 13ms
        # self.hilbert = HilbertTransform(axis=3)
        self.hilbert_choice =HilbertTransform(axis = 3)
        # self.pool1 = nn.AvgPool2d((1, self.pool_window), stride=(1, self.stride_window))  # spatial activation of 100ms and with a stride of 13ms
        self.pool1_choice = nn.AvgPool2d((1, self.pool_window), stride=(1, self.stride_window))  # spatial activation of 100ms and with a stride of 13ms

        # self.dropout1 = torch.nn.Dropout(p=dropout)
        self.dropout1_choice = torch.nn.Dropout(p=dropout)

        # self.fc3 = torch.nn.Linear(32*3*self.output_size,1)
        # self.fc4 =  torch.nn.Linear(32*3*self.output_size,1)
        self.fc_choice = torch.nn.Linear(32*2*1*self.output_size,1)

        self.gradients = None



        self.gap0_choice = torch.nn.AdaptiveAvgPool2d(1)

        ### Fully Connected Multi-Layer Perceptron (FC-MLP)
        self.mlp0_choice = torch.nn.Sequential(
            torch.nn.Linear(32*2, (32*2) // 6, bias=False),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear((32*2) // 6, 32*2, bias=False),
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
        # x0 = self.sinc_cnn2d(x)
        x0_choice = self.sinc_cnn2d_choice(x)


        # start attention for choice
        b_choice, c_choice, _, _ = x0_choice.size()
        y0_choice = self.gap0_choice(x0_choice).view(b_choice, c_choice)
        y0_choice = self.mlp0_choice(y0_choice).view(b_choice, c_choice, 1, 1)
        score_new_choice = x0_choice * y0_choice.expand_as(x0_choice)

        # end attention



        score0_choice_ = self.b_choice(score_new_choice)
        score0_choice = self.separable_conv_choice(score0_choice_) # output is [n, 64,1,1870)


        score_choice = self.b2_choice(score0_choice)
        score_choice = torch.abs(self.hilbert_choice(score_choice))

        score_choice = self.pool1_choice(score_choice)
        score_choice = self.dropout1_choice(score_choice)  # output is [n, 64,1,,37)


        score2_choice = score_choice.view(-1,32*2*1*self.output_size)
        score3_choice = self.fc_choice(score2_choice)   # choice
        score3_choice = F.sigmoid(score3_choice)


        return score3_choice

    def get_activations_gradient_filter(self):
        return self.gradients_filter
    def get_activations_gradient_temp(self):
        return self.gradients_temp









class SincHilbert_classify_relu(nn.Module):
    '''predicts drift and boundary
    attention layer after sinc and after conv'''

    # filter_length = 251
    filter_length = 131
    t_length = 500
    pool_window = int(np.rint(((t_length- filter_length +1) * 150)/ (t_length*2)))
    stride_window =  int(np.rint(((t_length- filter_length +1) * 50)/ (t_length*2)))
    if pool_window % 2 == 0 :
        pool_window -= 1
    if stride_window % 2 == 0 :
        stride_window -= 1
    output_size = int(np.floor(((t_length- filter_length +1) - pool_window)/stride_window +1))
    # output_size = 186
    num_chan = 98

    def __init__(self, dropout):
        super(SincHilbert_classify_relu, self).__init__()
        self.b0 = nn.BatchNorm2d(1, momentum=0.99)
        # self.sinc_cnn2d = sinc_conv(32, self.filter_length, 500)
        self.sinc_cnn2d_choice = sinc_conv(32*1, self.filter_length, 500)

        # self.b =  nn.BatchNorm2d(32, momentum=0.99)
        self.b_choice = nn.BatchNorm2d(32*1, momentum=0.99)

        # self.separable_conv = SeparableConv2d(32, 32, depth = 3, kernel_size= (self.num_chan,1))
        self.separable_conv_choice = SeparableConv2d(32*1, 32*1, depth = 1, kernel_size= (self.num_chan,1))

        # self.separable_conv_point = SeparableConv2d_pointwise(32*3, 32*3, depth = 1, kernel_size= (1,8))

        # self.fc2 = torch.nn.Linear(495,495)
        # self.b2 = nn.BatchNorm2d(32*3, momentum=0.99)
        self.b2_choice = nn.BatchNorm2d(32*1*1, momentum=0.99)

        # self.pool1 = nn.MaxPool2d((1, 151), stride=(1, 45))  # spatial activation of 100ms and with a stride of 13ms
        # self.pool1 = nn.AvgPool2d((1, 107), stride=(1, 45))  # spatial activation of 100ms and with a stride of 13ms
        # self.hilbert = HilbertTransform(axis=3)
        self.hilbert_choice =HilbertTransform(axis = 3)
        # self.pool1 = nn.AvgPool2d((1, self.pool_window), stride=(1, self.stride_window))  # spatial activation of 100ms and with a stride of 13ms
        self.pool1_choice = nn.AvgPool2d((1, self.pool_window), stride=(1, self.stride_window))  # spatial activation of 100ms and with a stride of 13ms

        # self.dropout1 = torch.nn.Dropout(p=dropout)
        self.dropout1_choice = torch.nn.Dropout(p=dropout)

        # self.fc3 = torch.nn.Linear(32*3*self.output_size,1)
        # self.fc4 =  torch.nn.Linear(32*3*self.output_size,1)
        self.fc_choice = torch.nn.Linear(32*1*1*self.output_size,1)

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
        # self.gap0 = torch.nn.AdaptiveAvgPool2d(1)
        #
        # ### Fully Connected Multi-Layer Perceptron (FC-MLP)
        # self.mlp0 = torch.nn.Sequential(
        #     torch.nn.Linear(32, 32 // 6, bias=False),
        #     torch.nn.ReLU(inplace=True),
        #     torch.nn.Linear(32 // 6, 32, bias=False),
        #     torch.nn.Sigmoid()
        # )

        self.gap0_choice = torch.nn.AdaptiveAvgPool2d(1)

        ### Fully Connected Multi-Layer Perceptron (FC-MLP)
        self.mlp0_choice = torch.nn.Sequential(
            torch.nn.Linear(32*1, (32*1) // 6, bias=False),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear((32*1) // 6, 32*1, bias=False),
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
        # x0 = self.sinc_cnn2d(x)
        x0_choice = self.sinc_cnn2d_choice(x)
        # h = x0.register_hook(self.activations_filterhook)
        #
        # # start attention
        # b, c, _, _ = x0.size()
        # y1 = self.gap0(x0).view(b, c)
        # y1 = self.mlp0(y1).view(b, c, 1, 1)
        # x1 = x0 * y1.expand_as(x0)
        # # end attention

        # # start attention
        # b, c, _, _ = x0.size()
        # y0 = self.gap0(x0).view(b, c)
        # y0 = self.mlp0(y0).view(b, c, 1, 1)
        # score_new = x0 * y0.expand_as(x0)
        #
        # # end attention

        # start attention for choice
        b_choice, c_choice, _, _ = x0_choice.size()
        y0_choice = self.gap0_choice(x0_choice).view(b_choice, c_choice)
        y0_choice = self.mlp0_choice(y0_choice).view(b_choice, c_choice, 1, 1)
        score_new_choice = x0_choice * y0_choice.expand_as(x0_choice)

        # end attention



        # spatial convulation layer
        # score0_ = self.b(score_new)
        # score0 = self.separable_conv(score0_) # output is [n, 64,1,1870)
        # score0 = self.separable_conv_point(score0) # output is [n, 64,1,1870)

        # spatial convulation layer for choice
        score0_choice_ = self.b_choice(score_new_choice)
        score0_choice = self.separable_conv_choice(score0_choice_) # output is [n, 64,1,1870)

        # score = self.b2(score0)
        # score = torch.abs(self.hilbert(score))
        #
        # score = self.pool1(score)
        # score = self.dropout1(score)  # output is [n, 64,1,,37)
        #

        score_choice = self.b2_choice(score0_choice)
        score_choice = F.relu(score_choice)

        score_choice = self.pool1_choice(score_choice)
        score_choice = self.dropout1_choice(score_choice)  # output is [n, 64,1,,37)


        # # fully connected layer
        # score2 = score.view(-1,32*3*self.output_size)  # output [batch size, 64*17)
        # score3= self.fc3(score2)# output is [batch size, 1]

        score2_choice = score_choice.view(-1,32*1*1*self.output_size)
        score3_choice = self.fc_choice(score2_choice)   # choice
        score3_choice = F.sigmoid(score3_choice)
        # score4 = self.fc4(score2)
        # score4 = F.sigmoid(self.fc4(score2)) *3    # right now uses sigmoid
        # score4 = F.softplus(self.fc4(score2))

        return score3_choice

    def get_activations_gradient_filter(self):
        return self.gradients_filter
    def get_activations_gradient_temp(self):
        return self.gradients_temp





class SincHilbert_classify_hilbertComplex(nn.Module):
    '''predicts drift and boundary
    attention layer after sinc and after conv'''

    # filter_length = 251
    filter_length = 131
    t_length = 500
    pool_window = int(np.rint(((t_length- filter_length +1) * 150)/ (t_length*2)))
    stride_window =  int(np.rint(((t_length- filter_length +1) * 50)/ (t_length*2)))
    if pool_window % 2 == 0 :
        pool_window -= 1
    if stride_window % 2 == 0 :
        stride_window -= 1
    output_size = int(np.floor(((t_length- filter_length +1) - pool_window)/stride_window +1))
    # output_size = 186
    num_chan = 98

    def __init__(self, dropout):
        super(SincHilbert_classify_hilbertComplex, self).__init__()
        self.b0 = nn.BatchNorm2d(1, momentum=0.99)
        # self.sinc_cnn2d = sinc_conv(32, self.filter_length, 500)
        self.sinc_cnn2d_choice = sinc_conv(32*2, self.filter_length, 500)

        # self.b =  nn.BatchNorm2d(32, momentum=0.99)
        self.b_choice = nn.BatchNorm2d(32*2, momentum=0.99)

        # self.separable_conv = SeparableConv2d(32, 32, depth = 3, kernel_size= (self.num_chan,1))
        self.separable_conv_choice = SeparableConv2d(32*2, 32*2, depth = 1, kernel_size= (self.num_chan,1))

        # self.separable_conv_point = SeparableConv2d_pointwise(32*3, 32*3, depth = 1, kernel_size= (1,8))

        # self.fc2 = torch.nn.Linear(495,495)
        # self.b2 = nn.BatchNorm2d(32*3, momentum=0.99)
        self.b2_choice = nn.BatchNorm2d(32*2*1, momentum=0.99)

        # self.pool1 = nn.MaxPool2d((1, 151), stride=(1, 45))  # spatial activation of 100ms and with a stride of 13ms
        # self.pool1 = nn.AvgPool2d((1, 107), stride=(1, 45))  # spatial activation of 100ms and with a stride of 13ms
        # self.hilbert = HilbertTransform(axis=3)
        self.hilbert_choice =HilbertTransform(axis = 3)
        # self.pool1 = nn.AvgPool2d((1, self.pool_window), stride=(1, self.stride_window))  # spatial activation of 100ms and with a stride of 13ms
        self.pool1_choice = nn.AvgPool2d((1, self.pool_window), stride=(1, self.stride_window))  # spatial activation of 100ms and with a stride of 13ms

        # self.dropout1 = torch.nn.Dropout(p=dropout)
        self.dropout1_choice = torch.nn.Dropout(p=dropout)

        # self.fc3 = torch.nn.Linear(32*3*self.output_size,1)
        # self.fc4 =  torch.nn.Linear(32*3*self.output_size,1)
        self.fc_choice = torch.nn.Linear(32*2*1*self.output_size,1)

        self.gradients = None



        self.gap0_choice = torch.nn.AdaptiveAvgPool2d(1)

        ### Fully Connected Multi-Layer Perceptron (FC-MLP)
        self.mlp0_choice = torch.nn.Sequential(
            torch.nn.Linear(32*2, (32*2) // 6, bias=False),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear((32*2) // 6, 32*2, bias=False),
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
        # x0 = self.sinc_cnn2d(x)
        x0_choice = self.sinc_cnn2d_choice(x)


        # start attention for choice
        b_choice, c_choice, _, _ = x0_choice.size()
        y0_choice = self.gap0_choice(x0_choice).view(b_choice, c_choice)
        y0_choice = self.mlp0_choice(y0_choice).view(b_choice, c_choice, 1, 1)
        score_new_choice = x0_choice * y0_choice.expand_as(x0_choice)

        # end attention



        score0_choice_ = self.b_choice(score_new_choice)
        score0_choice = self.separable_conv_choice(score0_choice_) # output is [n, 64,1,1870)


        score_choice = self.b2_choice(score0_choice)
        score_choice = torch.abs(self.hilbert_choice(score_choice))

        score_choice = self.pool1_choice(score_choice)
        score_choice = self.dropout1_choice(score_choice)  # output is [n, 64,1,,37)


        score2_choice = score_choice.view(-1,32*2*1*self.output_size)
        score3_choice = self.fc_choice(score2_choice)   # choice
        score3_choice = F.sigmoid(score3_choice)


        return score3_choice

    def get_activations_gradient_filter(self):
        return self.gradients_filter
    def get_activations_gradient_temp(self):
        return self.gradients_temp
