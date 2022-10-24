# Created on 10/23/22 at 3:25 PM 

# Author: Jenny Sun

import torch
import numpy as np

def getFilt(model_dic: dict, branchName: str, sr: int, min_freq=1, min_band=2, cutoff=50):
    '''branchName: 'bound', 'drift', or 'choice' '''
    keys = [f for f in model_dic.keys() if 'sinc_cnn2d' in f and branchName in f]
    print('found keys: ', keys)
    key_b1, key_band = [k for k in keys if 'filt_b1' in k], [k for k in keys if 'filt_band' in k]
    p_low = model_dic[key_b1[0]]
    p_band = model_dic[key_band[0]]
    filt_beg_freq = (torch.abs(p_low) + min_freq / sr)
    filt_end_freq = torch.clamp(filt_beg_freq + torch.abs(p_band) + min_band / sr, \
                                int(min_freq + min_band) / sr, cutoff / sr)
    filt_beg_freq = filt_beg_freq.cpu().numpy() * sr
    filt_end_freq = filt_end_freq.cpu().numpy() * sr

    return p_low, p_band, filt_beg_freq, filt_end_freq