# Created on 10/23/22 at 10:39 PM 

# Author: Jenny Sun
import numpy as np
import math
import matplotlib.pyplot as plt
from numpy.fft import fft, ifft


def makeSincFilters(filt_begin, filt_end, sr = 1000, N = 1001, windowFunc =True):
    '''filt_begin takes frequency, not fraction of sampling rate
    therefore, sr and N doesn't really matter
    N must be odd number'''

    filt_begin = filt_begin /sr
    filt_end = filt_end / sr
    n = np.arange(N)
    numFilters = len(filt_begin)
    myfilters = np.zeros((numFilters,N))
    for numf in range(numFilters):
        fc0 = filt_begin[numf]
        h0 = 2 * fc0 * np.sinc((2 * fc0 * (n - (N - 1) / 2)))

        fc1 = filt_end[numf]
        h1 = 2 * fc1 * np.sinc((2 * fc1 * (n - (N - 1) / 2)))
        band = h1 - h0
        band = band / np.max(band)
        if windowFunc:
            band = band * (0.54 - 0.46 * np.cos(2 * math.pi * n / N))
        myfilters[numf, :] = band
    return myfilters

def plotFFT(x,dur=1, sr=1000, xlim = 50):
    dur = 1
    L = sr * dur
    T = 1.0 / sr  # sampling period
    # fft the data
    Y = fft(x)
    # compute the two-sided spectrum P2
    P2 = np.abs(Y / L)  # power spectrum
    # then compute the single-sided spectrum P1 based on P2a and the real-valued signal length L
    P1 = P2[0:int(L / 2)]  # 0 Hz included
    P1[1:] = 2 * P1[1:]
    f = sr * np.arange(0, L / 2) / L
    # ax.plot(f, P1)
    # ax.set_xlim(0, xlim)
    return f, P1

# # plot sinc filters
#
# def flip(x, dim):
#     xsize = x.size()
#     dim = x.dim() + dim if dim < 0 else dim
#     x = x.contiguous()
#     x = x.view(-1, *xsize[dim:])
#     x = x.view(x.size(0), x.size(1), -1)[:, getattr(torch.arange(x.size(1) - 1,
#                                                                  -1, -1), ('cpu', 'cuda')[x.is_cuda])().long(),
#         :]  # flip left and right
#     return x.view(xsize)
#
#
# def sinc(band, t_right):
#     # print('band', band.is_cuda)
#     # print('t_right', t_right.is_cuda)
#
#     y_right = torch.sin(2 * math.pi * band * t_right) / (2 * math.pi * band * t_right).cuda()
#     y_left = flip(y_right, 0)
#     y = torch.cat([y_left.cuda(), Variable(torch.ones(1)).cuda(), y_right.cuda()])
#
#     return y
#
#
# filters = torch.zeros((32, 131))
# N = 131
# t_right = torch.linspace(1, (N - 1) / 2, int((N - 1) / 2)) / sr
# filt_begin_drift_sinc = torch.tensor(filt_begin_drift / sr).cuda()
# filt_end_drift_sinc = torch.tensor(filt_end_drift / sr).cuda()
# n = torch.linspace(0, N, N)
# window = 0.54 - 0.46 * torch.cos(2 * math.pi * n / N);
# for i in range(32):
#     low_pass1 = 2 * filt_begin_drift_sinc[i] * sinc(filt_begin_drift_sinc[i] * sr, t_right.cuda())
#     low_pass2 = 2 * filt_end_drift_sinc[i] * sinc(filt_end_drift_sinc[i] * sr, t_right.cuda())
#     band_pass = (low_pass2 - low_pass1)
#     band_pass = band_pass / torch.max(band_pass)
#     filters[i, :] = band_pass * window.cuda()
#
# filters = filters.cpu().numpy()
# filters_sum = filters.T @ attentionTs_drift_mean
# plt.plot(filters_sum)

