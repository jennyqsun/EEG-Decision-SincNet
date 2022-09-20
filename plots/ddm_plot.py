# Created on 2/13/22 at 6:00 PM 

# Created on 12/13/21 at 11:16 AM

# Author: Jenny Sun
# Created on 11/9/21 at 11:02 AM

# Author: Jenny Sun


import os
import scipy.stats
# import scipy.io.wavfile
# import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.utils
import torchvision
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
# python speaker_id.py --cfg=cfg/SincNet_TIMIT.cfg
from torch.utils.data import Dataset
from torchvision import datasets, transforms
import sys
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from matplotlib import interactive
interactive(True)
from dnn_models_pdm import *
from data_io import read_conf_inp
# from pymatreader import read_mat
import numpy as np
import os
# from hdf5storage import savemat
from hdf5storage import loadmat
from sklearn.model_selection import train_test_split
from pytorchtools import EarlyStopping, RMSLELoss
from torch.utils.tensorboard import SummaryWriter
from sklearn.linear_model import LinearRegression
import readchans
import random
from sklearn.metrics import r2_score
import pickle
import matplotlib
from matplotlib.gridspec import GridSpec
import pandas as pd
seednum = 2021

############################ define model parameters ######################

timestart = 625-150
timeend = 625+900
trialdur = timeend * 2 - timestart * 2
correctModel = False
notrainMode = True

# timeend = 800 # when 300ms after stim

# Hyper-parameters
num_epochs = 300
batch_size = 64
learning_rate = 0.001
num_chan = 98
dropout_rate = 0.5

cross_val_dir = 'crossval_metric_30_625_1625/'

############################# define random seeds ###########################
torch.manual_seed(seednum)
np.random.seed(seednum)
random.seed(seednum)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


g = torch.Generator()
g.manual_seed(seednum)

######################## tensorbaord initilization ###########################
tb = SummaryWriter('runs/regression_new')

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


####################### some functions for getting the EEG data ##############
def remove_ticks(fig):
    for i, ax in enumerate(fig, axes):
        ax.tick_params(labelbottom=False, labelleft=False)


def viz_histograms(model, epoch):
    for name, weight in model.named_parameters():
        try:
            tb.add_histogram(name, weight, epoch)
            tb.add_histogram(f'{name}.grad', weight.grad, epoch)
        except NotImplementedError:
            continue


def getIDs():
    path = '/home/jenny/pdmattention/'
    subj = loadmat('behavior2_task3')['uniquepart'][0]
    allDataFiles = os.listdir(path + 'task3/final_interp')
    sublist = []
    for sub in subj:
        # newsub= [x[0:10] for ind, x in enumerate(allDataFiles) if int(x[1:4]) == sub]
        newsub = [x[0:10] for ind, x in enumerate(allDataFiles) if int(x[1:4]) == sub]
        sublist += newsub
    finalsub = []
    for i in sublist:
        finalsub = [x[0:10] for ind, x in enumerate(allDataFiles) if
                    x not in sublist and x[1:4] != '236' and x[1:4] != '193']
    finalsub.sort()
    return sublist, finalsub


def chansets_new():
    chans = np.arange(0, 128)
    chans_del = np.array(
        [56, 63, 68, 73, 81, 88, 94, 100, 108, 114, 49, 43, 48, 38, 32, 44, 128, 127, 119, 125, 120, 121, 126,
         113, 117, 1, 8, 14, 21, 25]) - 1
    chans = np.delete(chans, chans_del)
    return chans


def plot_grad_flow(named_parameters):
    '''Plots the gradients flowing through different layers in the net during training.
        Can be used for checking for possible gradient vanishing / exploding problems.'''
    gradss = []
    ave_grads = []
    max_grads = []
    layers = []
    for n, p in named_parameters:
        if (p.requires_grad) and ("bias" not in n):
            layers.append(n)
            gradss.append(p.grad.detach())
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
    # plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    # plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    # plt.hlines(0, 0, len(ave_grads) + 1, lw=2, color="k")
    # plt.xticks(range(0, len(ave_grads), 1), layers, rotation="75")
    # plt.xlim(left=0, right=len(ave_grads))
    # plt.ylim(bottom=-0.001, top=0.02)  # zoom in on the lower gradient regions
    # plt.xlabel("Layers")
    # plt.ylabel("average gradient")
    # plt.title("Gradient flow")
    # plt.grid(True)
    # plt.legend(['max-gradient', 'mean-gradient', 'zero-gradient'])
    # print('grads ', grads[1][0].detach())
    # grads_all.append(grads[1][0].detach())
    return max_grads, ave_grads, gradss, layers


def loadsubjdict(subID):
    path = '/home/jenny/pdmattention/task3/final_interp/'
    datadict = loadmat(path + subID + 'final_interp.mat')
    return datadict


def loadinfo(subID):
    path = '/home/jenny/pdmattention/task3/expinfo/'
    infodict = loadmat(path + subID + 'task3_expinfo.mat')
    spfs = np.squeeze(infodict['spfs'])

    correctchoice = np.zeros(infodict['rt'].shape[1])
    easycond = np.squeeze((infodict['condition'] == 1) | (infodict['condition'] == 4))
    medcond = np.squeeze((infodict['condition'] == 2) | (infodict['condition'] == 5))
    hardcond = np.squeeze((infodict['condition'] == 3) | (infodict['condition'] == 6))

    correctchoice[((easycond == True) & (spfs > 2.5))] = 1
    correctchoice[((medcond == True) & (spfs > 2.5))] = 1
    correctchoice[((hardcond == True) & (spfs > 2.5))] = 1
    # 1 would be high freq 0 would be low, 1 would be right hand, 0 would be left hand
    datadict = loadsubjdict(subID)
    correctchoice = np.squeeze(correctchoice[datadict['trials']])
    acc = np.squeeze(datadict['correct'])
    responsemat = np.zeros(acc.shape)
    responsemat[(acc == 1) & (correctchoice == 1)] = 1
    responsemat[(acc == 0) & (correctchoice == 1)] = 0
    responsemat[(acc == 1) & (correctchoice == 0)] = 0
    responsemat[(acc == 0) & (correctchoice == 0)] = 1
    return responsemat, acc


def goodchans():
    datadict = loadsubjdict('s182_ses1_')
    goodchan = datadict['goodchans'][0]
    return goodchan


def getdata(datadict, Tstart=250, Tend=1250):
    data = np.array(datadict['data'])
    data = data[::2, :, :]
    sr = np.array(datadict['sr']) / 2
    condition = np.array(datadict['condition'])[0]
    goodtrials = np.array(datadict['trials'])[0]
    correct = np.array(datadict['correct'])[0]
    goodchan = goodchans()
    data = data[:, :, goodtrials]
    data = data[:, :, correct == 1]
    condition = condition[correct == 1]
    data = data[:, goodchan, :]
    return data[Tstart:Tend, :, :], condition


def getrtdata(datadict, Tstart=250, Tend=1250):
    data = np.array(datadict['data'])
    data = data[::2, :, :]
    sr = np.array(datadict['sr']) / 2
    condition = np.array(datadict['condition'])[0]
    goodtrials = np.array(datadict['trials'])[0]
    correct = np.array(datadict['correct'])[0]
    rt = np.array(datadict['rt'])[0]
    rt_label = np.hstack((np.zeros(len(rt) // 3), np.ones(len(rt) // 3)))
    slowest = np.ones(len(rt) - len(rt_label)) + 1
    rt_label = np.hstack((rt_label, slowest))
    rt_label += 1
    # goodchan = goodchans()
    # goodchan = chanmotor()
    goodchan = chansets_new()
    data = data[:, :, goodtrials]
    # data = data[:, :, correct==1]
    # condition = condition[correct==1]
    data = data[:, goodchan, :]
    return data[Tstart:Tend, :, :], condition, rt_label, rt,sr


def reshapedata(data):
    timestep, nchan, ntrial = data.shape
    newdata = np.zeros((ntrial, nchan, timestep))
    for i in range(0, ntrial):
        newdata[i, :, :] = data[:, :, i].T
    return newdata


############################# class for dataloaders ########################
# produce the dataset
class SubTrDataset(Dataset):
    def __init__(self, transform=None):
        self.n_samples = X_train_sub.shape[0]
        self.x_data = np.asarray(X_train_sub, dtype=np.float32)
        Xmean = np.mean(self.x_data, axis=2)
        Xmean_mat = Xmean[:, :, np.newaxis].repeat(X_train_sub.shape[-1], axis=2)
        self.x_data = self.x_data - Xmean_mat

        self.y_data = np.asarray(y_train_sub, dtype=np.float32)

        self.transform = transform

    def __getitem__(self, index):
        sample = self.x_data[index], self.y_data[[index]]

        if self.transform:  # if transform is not none
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return self.n_samples


# produce the dataset
class ValDataset(Dataset):
    def __init__(self, transform=None):
        self.n_samples = X_val.shape[0]
        self.x_data = np.asarray(X_val, dtype=np.float32)
        Xmean = np.mean(self.x_data, axis=2)
        Xmean_mat = Xmean[:, :, np.newaxis].repeat(X_val.shape[-1], axis=2)
        self.x_data = self.x_data - Xmean_mat
        self.y_data = np.asarray(y_val, dtype=np.float32)

        self.transform = transform

    def __getitem__(self, index):
        sample = self.x_data[index], self.y_data[[index]]

        if self.transform:  # if transform is not none
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return self.n_samples


# produce the dataset
class TrDataset(Dataset):
    def __init__(self, transform=None):
        self.n_samples = X_train0.shape[0]
        self.x_data = np.asarray(X_train0, dtype=np.float32)
        Xmean = np.mean(self.x_data, axis=2)
        Xmean_mat = Xmean[:, :, np.newaxis].repeat(X_train0.shape[-1], axis=2)
        self.x_data = self.x_data - Xmean_mat

        self.y_data = np.asarray(y_train0, dtype=np.float32)
        self.y_data_resp = np.asarray(y_train0_resp, dtype=np.float32)

        self.transform = transform

    def __getitem__(self, index):
        sample = self.x_data[index], self.y_data[[index]], self.y_data_resp[[index]]

        if self.transform:  # if transform is not none
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return self.n_samples


# produce the dataset
class TestDataset(Dataset):
    def __init__(self, transform=None):
        self.n_samples = X_test.shape[0]
        self.x_data = np.asarray(X_test, dtype=np.float32)
        Xmean = np.mean(self.x_data, axis=2)
        Xmean_mat = Xmean[:, :, np.newaxis].repeat(X_test.shape[-1], axis=2)
        self.x_data = self.x_data - Xmean_mat
        self.y_data = np.asarray(y_test, dtype=np.float32)
        self.y_data_resp = np.asarray(y_test_resp, dtype=np.float32)

        self.transform = transform

    def __getitem__(self, index):
        sample = self.x_data[index], self.y_data[[index]],self.y_data_resp[[index]]

        if self.transform:  # if transform is not none
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return self.n_samples


class ToTensor:
    # Convert ndarrays to Tensors
    def __call__(self, sample):  # not it became a callable object
        inputs, targets, targets_resp = sample
        return torch.from_numpy(inputs), torch.from_numpy(targets), torch.from_numpy(targets_resp).long()


def reset_weights(m):
    '''
      Try resetting model weights to avoid
      weight leakage.
    '''
    for layer in m.children():
        if hasattr(layer, 'reset_parameters'):
            print(f'Reset trainable parameters of layer = {layer}')
            layer.reset_parameters()


def initialize_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_uniform_(m.weight.data)
        print('init xavier uniform %s' % m)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight.data, 1)
        nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        print('init xavier uniform %s' % m)
        nn.init.kaiming_uniform_(m.weight.data)
        nn.init.constant_(m.bias.data, 0)


# %%
############################################################################
################################# starts here ###############################
############################################################################
results = dict()  # a results dictionary for storing all the data
subIDs, finalsubIDs = getIDs()
mylist = np.arange(0, len(finalsubIDs))

############################################
############### set subject ######################
############################################
goodl = [4,14,19,29,31,33,41]
for s in range(14, 15):
# for s in goodl:
    print(s)
    # a results dictionary for storing all the data
    subIDs, finalsubIDs = getIDs()
    # for i in range(0,1):
    torch.manual_seed(seednum)
    np.random.seed(seednum)
    random.seed(seednum)


    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2 ** 32
        np.random.seed(worker_seed)
        random.seed(worker_seed)


    g = torch.Generator()
    g.manual_seed(seednum)
    subjectstart = mylist[s]
    subjectend = subjectstart + 1

    ####################### define sub #########################################
    datadict = loadsubjdict(finalsubIDs[subjectstart])
    ddmparams = loadmat('/home/jenny/pdmattention/sincnet/single_nocond_' + finalsubIDs[s] + '.mat')

    print(str(subjectstart) + '/' + 'subjectID: ' + finalsubIDs[subjectstart])
    data, _, _, condition,sr = getrtdata(datadict, timestart, timeend)
    response, acc = loadinfo(finalsubIDs[subjectstart])
    rtall = condition.copy()
    # condition = np.log(condition)
    targetlabel = (condition, response)
    newdata = reshapedata(data).astype('float32')
    # X_train0, X_test, y_train0, y_test = train_test_split(newdata, condition, test_size=0.2, random_state=42)
    # X_train0, X_test, y_train0_resp, y_test_resp = train_test_split(newdata,  response, test_size=0.2, random_state=42)
    # _, _, y_train0_resp, y_test_resp = train_test_split(newdata,  response, test_size=0.2, random_state=42)

    alpha, ndt_mcmc, delta = ddmparams['alpha'][0][0][2][0][0],ddmparams['ndt'][0][0][2][0][0],ddmparams['delta'][0][0][2][0][0]
    tau = ndt_mcmc
    # tau_motor = ndt_mcmc/2
    beta= 0.5
    varsigma = 1

    maxtime = 1.8
    ndom = 5e2
    from scipy import signal
    try:
        from makefilter import makefiltersos
        import baselinecorrect
        from baselinecorrect import baselinecorrect
    except ModuleNotFoundError:
        from timeop import makefiltersos
        from timeop import baselinecorrect
    import matplotlib.lines as lines
    from scipy import linalg
    from scipy.stats import gaussian_kde

    rt = 500
    sr = np.squeeze(int(sr))
    eeg_n200 = np.mean(data, axis=2)
    sos, w, h = makefiltersos(sr, 10, 20) #lowpass filter
    erpfilt = signal.sosfiltfilt(sos, eeg_n200, axis=0, padtype='odd')
    erpfiltbase = baselinecorrect(erpfilt, np.arange(0, 150, 1))
    u, _, vh = linalg.svd(erpfiltbase[225:225+135])

    weights = np.zeros((98, 1))

    # This is an optimal set of weights to estimate a single erp peak.

    weights[:, 0] = np.matrix.transpose(vh[0, :])

    # Lets test it on the average time series.
    erpfiltproject = np.matmul(erpfiltbase, weights)
    erpmin = np.amin(erpfiltproject[225:225+135])
    erpmax = np.amax(erpfiltproject[225:225+135])
    if abs(erpmin) < abs(erpmax):
        weights = -weights
        erpfiltproject = -erpfiltproject
#
#     # erp_peaktiming = np.argmin(erpfiltproject[150:]) + 150
#     # indices = np.arange(erp_peaktiming - 10, erp_peaktiming + 10, 1)
#     # erp_peakvalue = np.mean(erpfiltproject[indices])
#     # now we need to apply it to every sample in the data set.
#
    numtrial = data.shape[-1]
    trialestimate = np.zeros((data.shape[0], data.shape[-1]))
    for trial in range(0,data.shape[-1]):
        trialdata = np.squeeze(data[:, :, trial])
        trialproject = np.matmul(trialdata, weights)
        trialestimate[:, trial] = trialproject[:, 0]
    trialestimatefilt = signal.sosfiltfilt(sos, trialestimate, axis=0, padtype='odd')
    trialestimatefiltbase = baselinecorrect(trialestimatefilt, np.arange(0, 150, 1))

    trialfinal = scipy.signal.detrend(trialestimatefiltbase[150:,:], axis=0)



    peakvalue = np.zeros(numtrial)
    peaktiming = np.zeros(numtrial)

    for j in range(numtrial):
        peaktiming[j] = 2 * (np.argmin(trialestimatefiltbase[225:225+135,j])) + 225
        indices = np.arange(peaktiming[j] - 5, peaktiming[j] + 5, 1)
        peakvalue[j] = (np.mean(trialestimatefiltbase[indices.astype(int), j], axis=0))

######################################
#%%
    fig, ax = plt.subplots(figsize = (10,8))

    upperbound = ax.axhline(alpha, xmin = 0, xmax = maxtime, color= 'black')
    lowerbound = ax.axhline(0, xmin=0, xmax=maxtime, color='black')
    neuterline = ax.axhline(alpha/2, xmin=0, xmax=maxtime, color='black')
    tartline = ax.axvline(0, ymin=0, ymax=alpha+1, color='black')
    upperupper = ax.axhline(alpha+2, xmin=0,xmax = maxtime, color = 'black')
    uppermiddle = ax.axhline(alpha+1, xmin=0,xmax = maxtime, color = 'black')



    #EEG plot
    #
    # ax.text(-.061, alpha + 1, r'0 $\mu$ V')
    # ax.text(-.06, alpha + 2, r'600 $\mu$ V');
    tlength = int(timeend - 625)
    trialmean = np.tile(np.mean(trialfinal, axis=0), (tlength,1))
    trialfinal = trialfinal -trialmean
    from cycler import cycler
    ax.prop_cycle: cycler(color='cmyk')

    ax.plot(np.arange(0,tlength*2,2), trialfinal/600/2 + alpha + 1, alpha=0.6)

    rt_left = rtall[(response == 0) ]
    rt_right = rtall[(response == 1) ]

    # PDF
    rt_left_correct = rtall[(response==0) & (acc==1) ]
    rt_left_incorrect = rtall[(response == 0) & (acc == 0)]
    rt_right_correct = rtall[(response==1) & (acc==1)]
    rt_right_incorrect = rtall[(response == 1) & (acc == 0)]
    ndom = 5000
    pp=[]
    for kd in [rt_left_correct, rt_left_incorrect,rt_right_correct,rt_right_incorrect]:
        kde = scipy.stats.gaussian_kde(kd)
        x = np.linspace(0, maxtime,ndom)
        p = kde(x)
        pp.append(p)

    pp_1= []
    for kd in [rt_left, rt_right]:
        kde = scipy.stats.gaussian_kde(kd)
        x = np.linspace(0, maxtime, ndom)
        p = kde(x)
        pp_1.append(p)

    scale = 0.4
    pdf01 = scale * pp[0] + alpha
    pdf00 = -scale*0.5 * pp[1]

    pdf11 = -scale * pp[2]
    pdf10 = scale*0.5 * pp[3] + alpha

    # scale = 0.4
    #
    # pdf011 = scale * pp_1[0]+alpha
    # pdf001 = -scale * pp_1[1]

    domain = np.linspace(0, maxtime, ndom);

    xvec = np.linspace(0,maxtime,ndom)*1000
    ax.plot(xvec, pdf01,color = 'tab:blue')
    ax.fill_between(xvec, np.ones_like(xvec)*alpha,pdf01, color = 'tab:blue')
    ax.plot(xvec, pdf00,ls='dashed',color = 'tab:blue')
    # ax.fill_between(xvec, np.zeros_like(xvec), pdf00,color = 'tab:blue',alpha=0.3)
    ax.plot(xvec, pdf10,ls='dashed',color = 'tab:red')
    # ax.fill_between(xvec, np.ones_like(xvec)*alpha,pdf10,color = 'tab:red',alpha=0.3)
    ax.plot(xvec, pdf11,color = 'tab:red')
    ax.fill_between(xvec, np.zeros_like(xvec), pdf11,color = 'tab:red')
    # fig.suptitle(str(s))

#   randomwealk

# %To simulate a Wiener process use:
# %X_t = mu*t + sigma*W_t
# %which implies X_t/dt = mu + sigma*(W_t/dt)
# %Approximate by X_t(t+dt) = X_t(t) + mu*dt + sigma*sqrt(dt)*randn

    ############## upper bound and drift arrow
    walk_t = np.linspace(tau / 2, maxtime, ndom)
    dt = maxtime / ndom
    path = np.zeros_like(walk_t)
    def drawone():
        for w in range(1, len(walk_t)):
            path[w] = path[w - 1] + dt * delta + 1 * np.sqrt(dt) * np.random.standard_normal(1);
        return path
    sim=True
    while sim:
        path = drawone()
        try:
            cross = np.where((path>=1) | (path<=-1))[0][0]
            sim=False
        except IndexError:
            sim=True
            ('redraw')
    ax.plot(walk_t[1:cross]*1000,path[1:cross]*alpha*beta + alpha/2,'tab:blue')
    # path += beta
    rt  = walk_t[cross] + tau/2
    xmarker = np.linspace(tau/2*1000,rt*1000/2,1000)
    ymarker = (delta*(np.linspace(0,rt/2,1000) *alpha*beta))+ alpha*beta

    ax.plot(xmarker,ymarker,'black',linewidth= 5)
    # ax.patches.Polygon()
    a =ax.plot(xmarker[-20:],ymarker[-20:],'>',color = 'black',markersize = 8)

    #ms marker
    ax.plot([500,500], [beta*alpha-0.1,beta*alpha+0.1],linewidth  = 2, color = 'grey')
    ax.plot([250,250], [beta*alpha-0.1,beta*alpha+0.1],linewidth  = 2, color = 'grey')
    ax.plot([750, 750], [beta * alpha - 0.1, beta * alpha + 0.1], linewidth=2, color='grey')
    ax.plot([1000, 1000], [beta * alpha - 0.1, beta * alpha + 0.1], linewidth=2, color='grey')
    ax.plot([1250, 1250], [beta * alpha - 0.1, beta * alpha + 0.1], linewidth=2, color='grey')
    ax.plot([1500, 1500], [beta * alpha - 0.1, beta * alpha + 0.1], linewidth=2, color='grey')
    ylim = ax.get_ylim()
    ax.plot([tau/2 *1000, tau/2*1000], (0,alpha*1.5) ,linewidth=2, color='tab:green')
    ax.plot([rt *1000, rt*1000], (0,alpha),color = 'tab:green')
    ax.fill_between
    tauvec = np.linspace(0,tau/2) *1000
    ax.fill_between(tauvec, np.zeros_like(tauvec) , np.zeros_like(tauvec) + alpha, color='tab:green', alpha=0.5)

    ax.fill_between(tauvec  +rt *1000 - tau/2*1000, np.zeros_like(tauvec), np.zeros_like(tauvec) + alpha, color='tab:green',alpha=0.5)

    ax.text(tau/2*1000-100, 1, r'$\tau_e$', fontsize=20)
    ax.text(rt *1000-100, 1, r'$\tau_m$', fontsize=20)


# transform=fig.transFigure, horizontalalignment='center', fontsize=20, weight='bold')
    ax.margins(x = 0)
    ax.margins(y=0)
    ax.plot(np.linspace(0,1800), 0 * np.linspace(0,1800) + 2.5, 'black')
    fig.text(0.15,0.55,'Choice 1', fontsize = 15)
    fig.text(0.15, 0.22, 'Choice 2', fontsize=15)
    fig.text(0.05, 0.28, 'Evidence \nAccumulation', fontsize=15, rotation=90)
    fig.text(0.05, 0.60, 'Single Trial\n EEG Amplitude', fontsize=15, rotation=90)
    fig.text(0.1, 0.50, r'$\alpha$', fontsize=18)
    fig.text(0.1, 0.25, r'$0$', fontsize=18)
    fig.text(0.1, 0.37, r'$\beta$', fontsize=18)




    ax.set_xlabel('Time (ms)', fontsize =15)
    # ax.set_axis_off()
    ax.get_yaxis().set_visible(False)
    fig.savefig('/home/jenny/sincnet_eeg/result_figures/ddm_plot.png')

    fig.show()



# ax.arrow(xmarker, ymarker)
    # fig.show()
    # # ax.annotate(xy = (xmarker[-1],ymarker[-1]),(tau/2*1000, beta*alpha), arrowprops=dict(facecolor='black', shrink=0.05),horizontalalignment='left',
    # #         verticalalignment='bottom')
    # import matplotlib.patches as mpatches

    # ax.arrow(x_tail,y_tail, dx,dy, head_width=0.5, head_length=0.3, fc='lightblue', ec='black')
#     arrow = mpatches.FancyArrowPatch((x_tail, y_tail), (dx, dy),
#                                  mutation_scale=50)

#     ax.add_patch(arrow)
#
# # ax.patches.arrow(tau/2*1000, beta*alpha,xmarker[-1],ymarker[-1])
# #
# #     ,fc='red', ec='red',shape='full', width=0.01,length_includes_head=True,head_width=0.3, )

    #
    #


# , alpha=0.5, width=1,
#             head_width=0.2,
    # sim = True
    # while sim:
    #     path = drawone()
    #     try:
    #         cross = np.where((path >= 1) | (path <= -1))[0][0]
    #         sim = False
    #     except IndexError:
    #         sim = True
    #         ('redraw')
    # ax.plot(walk_t[1:cross]*1000,-path[1:cross]*alpha*beta + alpha/2,'tab:red')
    #
    # fig.show()
#
#