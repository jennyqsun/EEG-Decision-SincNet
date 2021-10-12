# Created on 10/5/21 at 4:47 PM 

# Author: Jenny Sun
# speaker_id.py
# Mirco Ravanelli
# Mila - University of Montreal

# Description:
# This code performs SincNet on pdmattention

# How to run it:
# python speaker_id.py --cfg=cfg/SincNet_TIMIT.cfg

import os
# import scipy.io.wavfile
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import sys
import numpy as np
from dnn_models import MLP, flip
from dnn_models import SincNet as CNN
from data_io import ReadList, read_conf, str_to_bool
# from pymatreader import read_mat
import numpy as np
import scipy.signal as signal
from scipy import linalg
from scipy.io import savemat
from matplotlib import pyplot as plt
from scipy.fftpack import fft2
from scipy.fftpack import fft
from scipy import signal
import os
# from pymatreader import read_mat
# from hdf5storage import savemat
from hdf5storage import loadmat
#%%
# import lab modules
import timeop
from ssvep_subject import *
from readchans import *

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
num_epochs = 5
batch_size = 4
learning_rate = 0.001

def getIDs():
    path = '/home/jenny/pdmattention/'
    subj = loadmat('/home/jenny/Downloads/behavior2_task3')['uniquepart'][0]
    allDataFiles = os.listdir(path + 'task3/final_interp')
    sublist = []
    for sub in subj:
        # newsub= [x[0:10] for ind, x in enumerate(allDataFiles) if int(x[1:4]) == sub]
        newsub = [x[0:10] for ind, x in enumerate(allDataFiles) if int(x[1:4]) == sub]
        sublist += newsub
    return sublist


def loadsubjdict(subID):
    path = '/home/jenny/pdmattention/task3/final_interp/'
    datadict = loadmat(path + subID + 'final_interp.mat')
    return datadict

def getdata(datadict):
    data = np.array(datadict['data'])
    sr = np.array(datadict['sr'])
    condition = np.array(datadict['condition'])[0]
    goodtrials = np.array(datadict['trials'])[0]
    goodchan = np.array(datadict['goodchans'])[0]
    data = data[:,:, goodtrials]
    data = data[:,goodchan,:]
    return data, condition

subIDs = getIDs()
datadict = loadsubjdict(subIDs[0])
data, condition = getdata(datadict)

# produce the dataset
class PdmSubject(Dataset):
