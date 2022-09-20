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
# interactive(True)
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
from scipy.io import savemat
seednum = 2021
font = 17
############################ define model parameters ######################
timestart = 625
timeend = 625+500
trialdur = timeend * 2 - timestart * 2
correctModel = False
notrainMode = True
# sr = 1000
# timeend = 800 # when 300ms after stim

# Hyper-parameters
num_epochs = 300
batch_size = 64
learning_rate = 0.001
num_chan = 98
dropout_rate = 0.5
compute_likelihood = False

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

######################## creating directory and file nmae ############for s########
# postname = '_prestim500_1000_0123_ddm_2param'
postname = '_1000_ddm_2param_attention_bound'

modelpath = 'trained_model' + postname
resultpath = 'results' + postname
figurepath = 'figures' + postname

isExist = os.path.exists(modelpath)
isExist = os.path.exists(modelpath)
if not isExist:
    os.makedirs(modelpath)
    print(modelpath + ' created')

isExist = os.path.exists(figurepath)
if not isExist:
    os.makedirs(figurepath)
    print(figurepath + ' created')

isExist = os.path.exists(resultpath)
if not isExist:
    os.makedirs(resultpath)
    print(resultpath + ' created')

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
                    x[1:4] != '236' and x[1:4] != '193']
    finalsub.sort()
    return sublist, finalsub

def getddmparams(subj):
    path = '/home/jenny/pdmattention/alphaDC/estimates/'
    paramdic = loadmat(path + 'behavior2_task3_HDDM_AlphaJan_20_21_14_04_estimates.mat')
    uniquepart= loadmat('behavior2_task3')['uniquepart'][0]
    ind = np.where(uniquepart == int(subj[1:4]))[0]
    print('ind:',ind)
    if len(ind) == 0:
        print('!!!Warning: No DDM parameters extracted')
        sys.exit()
    else:
        print('subject DDM Parameters Deteted')

    alpha = paramdic['alpha'][0,0][2][ind,:]   # take the median
    ndt = paramdic['ndt'][0,0][2][ind,:]
    delta = paramdic['delta'][0,0][2][ind,:]
    return (np.mean(alpha), np.mean(ndt), np.mean(delta))


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
    return responsemat

    return datadict


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
    return data[Tstart:Tend, :, :], condition, rt_label, rt, correct


def reshapedata(data):
    timestep, nchan, ntrial = data.shape
    newdata = np.zeros((ntrial, nchan, timestep))
    for i in range(0, ntrial):
        newdata[i, :, :] = data[:, :, i].T
    return newdata

#
#
# def my_loss(t, v, t0, a):
#     # t is target RT
#     # v is output
#     # t0 is non decision time
#
#
#     w = torch.tensor(0.5).cuda()     # convert to relative start point
#     kk = torch.arange(-4,6)             # we set K to be 10
#     try:
#         k = torch.tile(kk,(t.shape[0],1)).cuda()
#     except IndexError:
#         k = kk.cuda()
#
#     err = torch.tensor(0.01).cuda()
#     tt = torch.max(torch.tensor(t.cuda() - torch.tensor(t0).cuda()),err) / torch.max(err,a.cuda()) ** 2  # normalized time
#     tt_vec = torch.tile(tt, (1, 10))
#     pp = torch.cumsum((w+2*k)*torch.exp(-(((w+2*k)**2)/2)/tt_vec),axis=1)
#     pp = pp[:,-1]/torch.sqrt(2*torch.tensor(np.pi)*torch.squeeze(tt)**3)
#     pp = pp[:, None]
#
#     p = torch.log(pp * torch.exp(-v*torch.max(err, a)*w - (v**2)*torch.tensor(t).cuda()/2) /(torch.max(err,a)**2))
#     return -(p.sum())


def my_loss(t, v, t0, a):
    # t is target RT
    # v is output
    # t0 is non decision time


    w = torch.tensor(0.5).cuda()     # convert to relative start point
    kk = torch.arange(-4,6)             # we set K to be 10
    try:
        k = torch.tile(kk,(t.shape[0],1)).cuda()
    except IndexError:
        k = kk.cuda()

    err = torch.tensor(0.02).cuda()
    tt = torch.max(torch.tensor(torch.abs(t.cuda()) - torch.tensor(t0).cuda()),err) / torch.max(err,a.cuda()) ** 2  # normalized time
    tt_vec = torch.tile(tt, (1, 10))
    pp = torch.cumsum((w+2*k)*torch.exp(-(((w+2*k)**2)/2)/tt_vec),axis=1)
    pp = pp[:,-1]/torch.sqrt(2*torch.tensor(np.pi)*torch.squeeze(tt)**3)
    pp = pp[:, None]
    # v = torch.where(torch.tensor(t).cuda()>0, v, -v)   # if time is negative, flip the sign of v
    v = torch.clamp(v, -6,6)
    # t = torch.where(torch.tensor(t).cuda() > 0, torch.tensor(t).cuda(), torch.tensor(-t).cuda())
    p = torch.log(pp * torch.exp(-v*torch.max(err, a)*w - (v**2)*torch.tensor(t).cuda()/2) /(torch.max(err,a)**2))
    # p = torch.where(torch.tensor(t).cuda()>0, p, -p)
    # print(t,a,v)
    # print('probability is ', p)
    return -(p.sum())
# def my_loss(t, v, t0, a,z,err=1e-29):
#     # t is target RT
#     # v is output
#     # t0 is non decision time
#
#     tt = torch.tensor(t.cuda()-torch.tensor(t0).cuda())/(torch.tensor(a).cuda()**2)   # normalized time
#     tt[tt<0] = 0.01
#     w = torch.tensor(z).cuda()/torch.tensor(a).cuda()        # convert to relative start point
#     ks = 2 + torch.sqrt(-2 * tt * torch.log(2 * torch.sqrt(2 * torch.tensor(np.pi) * tt) * err))   #bound
#     ks = torch.max(ks,torch.square(tt)+1)  # ensure bouhndary conditions are met
#     kk = torch.arange(-4,6)             # we set K to be 10
#     try:
#         k = torch.tile(kk,(t.shape[0],1)).cuda()
#     except IndexError:
#         k = kk.cuda()
#     tt_vec = torch.tile(tt, (1,10))
#     pp = torch.cumsum((w+2*k)*torch.exp(-(((w+2*k)**2)/2)/tt_vec),axis=1)
#     pp = pp[:,-1]/torch.sqrt(2*torch.tensor(np.pi)*torch.squeeze(tt)**3)
#     pp = pp[:, None]
#
#     p = torch.log(pp * torch.exp(-v*a*w - (v**2)*torch.tensor(t).cuda()/2) /(a**2))
#     return -(p.sum())
#
# def my_loss:
#     p = (t-t0)/a**2
#     p = 1/(2*np.pi*(tt**3))
#
#
# def my_loss(t, v, t0, a, err=1e-29):
#     # t is target RT
#     # v is output
#     # t0 is non decision time
#
#     tt = torch.tensor(t.cuda() - torch.tensor(t0).cuda()) / (torch.tensor(a).cuda() ** 2)  # normalized time
#     tt[tt < 0] = 0.01
#     w = 0.5
#     ks = 2 + torch.sqrt(-2 * tt * torch.log(2 * torch.sqrt(2 * torch.tensor(np.pi) * tt) * err))  # bound
#     ks = torch.max(ks, torch.square(tt) + 1)  # ensure bouhndary conditions are met
#     kk = torch.arange(-4, 6)  # we set K to be 10
#     try:
#         k = torch.tile(kk, (t.shape[0], 1)).cuda()
#     except IndexError:
#         k = kk.cuda()
#     tt_vec = torch.tile(tt, (1, 10))
#     pp = torch.cumsum(20.5 * torch.exp(-((20.5 ** 2) / 2) / tt_vec), axis=1)
#     pp = pp[:, -1] / torch.sqrt(2 * torch.tensor(np.pi) * torch.squeeze(tt) ** 3)
#     pp = pp[:, None]
#
#     p = torch.log(pp * torch.exp(-v * a * w - (v ** 2) * torch.tensor(t).cuda() / 2) / (a ** 2))
#     return -(p.sum())

    # # loss = torch.zeros(len(target),requires_grad=True).cuda()
    # # #
    # # for i in range(0,len(target)):
    # #     # loss[i] = - torch.tensor((wfpt_logp1(target[i], 1, bias[i], torch.abs(ndt[i]), drift[i], 1, eps = 1e-10))).cuda()
    # #     loss[i] = - torch.tensor((wfpt_logp1(target[i], 1, torch.abs(torch.tensor(-0.6)), torch.abs(torch.tensor(0.3)), drift[i], 1, eps = 1e-10))).cuda()
    # #     if torch.isinf(loss[i]):
    # #         loss[i] = - torch.log(torch.tensor(8.423e-40).cuda()) #to avoid having inf
    # loss = -1 * (((-1/2) * torch.log(2*torch.tensor(pi))) - ((1/2) * torch.log(torch.tensor(1)**2)) -(1/(2*torch.tensor(1)**2))*(target - ndt)**2)
    # # print('loss--------------': , loss )
    # return torch.mean(loss)

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

        self.transform = transform

    def __getitem__(self, index):
        sample = self.x_data[index], self.y_data[[index]]

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

        self.transform = transform

    def __getitem__(self, index):
        sample = self.x_data[index], self.y_data[[index]]

        if self.transform:  # if transform is not none
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return self.n_samples


class ToTensor:
    # Convert ndarrays to Tensors
    def __call__(self, sample):  # not it became a callable object
        inputs, targets = sample
        return torch.from_numpy(inputs), torch.from_numpy(targets)


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
subj = loadmat('behavior2_task3')['uniquepart'][0].tolist()
############################################
############### set subject ######################
############################################
for s in range(24, 25):
    # a results dictionary for storing all the data
    subIDs, finalsubIDs = getIDs()
    # for i in range(0,1):
    torch.manual_seed(seednum)
    np.random.seed(seednum)
    random.seed(seednum)


    # if int(finalsubIDs[s][1:4]) in subj:
    #     print('in-sample subject')
    # else:
    #     print('no in-sample subject, skipping to the next one>>>')
    #     continue



    # ddmparams = getddmparams(finalsubIDs[s])


    ddmparams = loadmat('/home/jenny/pdmattention/sincnet/single_nocond_' + finalsubIDs[s] + '.mat')
    alpha, ndt_mcmc, drift = ddmparams['alpha'][0][0][2][0][0],ddmparams['ndt'][0][0][2][0][0],ddmparams['delta'][0][0][2][0][0]

    # alpha, ndt, drift = ddmparams
    # alpha =  1.39681064
    # ndt = 0.39675787
    # drift = 0.89709653
    # alpha = alpha *2

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
    print(str(subjectstart) + '/' + 'subjectID: ' + finalsubIDs[subjectstart])
    data, cond, _, condition, correct = getrtdata(datadict, timestart, timeend)
    # response = loadinfo(finalsubIDs[subjectstart])
    rtall = condition.copy()
    correct = correct.astype('int')
    if correctModel is True:
        condition = (correct * 2 - 1) * condition
    # condition = np.log(condition)
    newdata = reshapedata(data).astype('float32')

    # # # get rid of the rts that are lower than ndt
    # newdata = newdata[rtall>ndt,:,:]
    # cond = cond[rtall>ndt]
    # correct = correct[rtall>ndt]
    # rtall = rtall[rtall>ndt]
    #
    # condition = condition[condition>ndt]

    # # get correct only trials
    # newdata=newdata[correct==1,:,:]
    # cond = cond[correct==1]
    # rtall = rtall[correct==1]
    # condition = condition[correct==1]
    # X_train000, X_test000, y_train000, y_test000 = train_test_split(newdata, condition, test_size=0.2, random_state=42)
    # ndt = np.percentile(y_train000,1)




    X_train0, X_test, y_train0, y_test = train_test_split(newdata, condition, test_size=0.2, random_state=42)
    ndt = np.min(np.abs(y_train0)) * 0.93
    print('MCMC ndt: ', ndt_mcmc)
    print('ndt: ', ndt)

    X_train00, X_test0, y_train0_cond, y_test_cond = train_test_split(newdata, cond, test_size=0.2, random_state=42)

    # ndtint_train = y_train0>ndt
    # ndtint_test = y_test> ndt
    # X_train0, X_test, y_train0, y_test = X_train0[ndtint_train,:,:], X_test[ndtint_test,:,:], y_train0[ndtint_train], y_test[ndtint_test]
    # X_train00, X_test0, y_train0_cond, y_test_cond = X_train00[ndtint_train,:,:], X_test0[ndtint_test,:,:], y_train0_cond[ndtint_train], y_test_cond[ndtint_test]
    #

    # y_train0 = np.ones_like(y_train0) * drift
    # print(X_train0[200, 50, 150])
    # print(X_test[24, 50, 150])

    train_set = TrDataset(transform=ToTensor())
    train_loader = DataLoader(dataset=train_set,
                              batch_size=batch_size,
                              shuffle=True,  # shuffle the data
                              num_workers=0, worker_init_fn=seed_worker,
                              generator=g)
    test_set = TestDataset(transform=ToTensor())
    test_loader = DataLoader(dataset=test_set,
                             batch_size=batch_size,
                             shuffle=False,  # shuffle the data
                             num_workers=0, worker_init_fn=seed_worker,
                             generator=g)

    # sample the data
    data, target = next(iter(train_loader))

    # plt.plot(data[10,:,:].T)
    # plt.show()
    data, target = next(iter(test_loader))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #################################################################################
    ######################## creating pre training visulization #####################
    #################################################################################
    targetlist = []
    predictedlist = []
    plt.rcParams.update({'font.size': 8})
    plt.rcParams["font.serif"] = "Times New Roman"
    plt.rcParams['lines.markersize'] = 3.0

    fig = plt.figure(figsize=(5, 5))
    gs = GridSpec(2, 2, figure=fig)
    # ax6 = fig.add_subplot(gs[0, 0])
    ax7 = fig.add_subplot(gs[0, 0:2])
    ax8 = fig.add_subplot(gs[1, 0:2])
    # ax9 = fig.add_subplot(gs[1, 1])
    ax4 = ax7
    # ax3 = ax6
    ax5 = ax8
    gradlist = []
    model_0 = Sinc_Conv2d_attention_pre(dropout=dropout_rate).cuda()
    model_0.eval()
    criterion = nn.MSELoss()
    n_total_steps = len(test_loader)

    fig2 = plt.figure(figsize=(5, 5))
    gs2 = GridSpec(1, 1, figure=fig)
    ax9 = fig2.add_subplot(gs2[0, 0])
    # ax10 = fig2.add_subplot(gs2[0,1])

    for i, (test_data, test_target) in enumerate(test_loader):
        cond_target = y_test_cond[i*batch_size+test_target.shape[0]-test_target.shape[0]:i*batch_size+test_target.shape[0]]
        #     # test_data, test_target = next(iter(test_loader))
        pred, pred_1,_ = model_0(test_data.cuda())

        pred_copy = pred.detach().cpu()
        pred.mean().backward()
        gradients = model_0.get_activations_gradient_filter()
        gradlist.append(gradients)
        test_target = torch.squeeze((test_target))
        if cond_target.shape[0]==1:
            test_target= test_target.view(1, 1)
        else:
            test_target = test_target.view(test_target.shape[0], 1)
        # test_loss = my_loss(test_target.cuda(), pred_copy.cuda(), ndt, alpha,alpha/2, err = 1e-29)
        test_loss = my_loss(test_target.cuda(), pred_copy.cuda(), ndt, torch.mean(pred_1.detach().cuda(), axis=0).cuda())

        r2 = r2_score(test_target.cpu().detach().numpy(), pred_copy.cpu().detach().numpy())
        # print("validation accuracy: ", val_acc)
        # print("validation loss: ", val_loss)
        # valacc_batch.append(val_acc.cpu())
        try:
            targetlist += torch.squeeze(test_target).tolist()
            predictedlist += torch.squeeze(-pred_copy).cpu().tolist()
        except TypeError:
            targetlist += [torch.squeeze(test_target).tolist()]
            predictedlist += [torch.squeeze(-pred_copy).cpu().tolist()]

        print(f'Testing Batch: {i}, Step [{i + 1}/{n_total_steps}], Loss: {test_loss.item():.4f}, R^2 : {r2}')
        # if i % 1 == 0:
            # plt.plot(test_target, label='target')
            # plt.plot(test_output.cpu().detach().numpy(), label='predicted')

            # ax0.scatter(test_target, pred_copy.cpu().detach().numpy(), color ='b')
    targetlist = np.array(targetlist)
    predictedlist = np.array(predictedlist)
    # ax0.scatter(targetlist[y_test_cond==1], predictedlist[y_test_cond==1], color='green', marker = 'o', label = 'easy')
    # ax0.scatter(targetlist[y_test_cond==2], predictedlist[y_test_cond==2], color='blue', marker = '*', label = 'median')
    # ax0.scatter(targetlist[y_test_cond==3], predictedlist[y_test_cond==3], color='red', marker = '^', label = 'hard')
    # ax0.legend()
    # ax0.set_xlabel('actual RT')
    # ax0.set_ylabel('predicted Drift')
    # ax1.hist(rtall * 1000, bins=12, color='green')
    # if timestart < 625:
    #     ax1.axvspan(0, (timeend-625)*2, color='cornflowerblue', alpha=0.5)
    # else:
    #     ax1.axvspan(0, trialdur, color='cornflowerblue', alpha=0.5)
    #     # xt = ax0.get_xticks()
    #     # xt= np.append(xt, trialdur)
    #     # xtl = xt.tolist()
    #     #
    #     # xtl[-1] = [format(trialdur)]
    # ax1.set_xticks([trialdur])
    # ax1.set_xticklabels(['window length' + format(trialdur) + 'ms\n' + 'post-stimulus:' + format(2*(timeend-625)) + 'ms'])
    # if timestart < 625:
    #     fractionrt = sum(rtall * 1000 < (timeend-625)*2) / len(rtall) * 100
    # else:
    #     fractionrt = sum(rtall * 1000 < trialdur) / len(rtall) * 100
    # ax1.text(0, ax1.get_ylim()[1] / 3, '%.2f' % fractionrt + '% \nof all\n RTs')
    # ax1.set_title('Fraction of RT')

        # fig.show()
    try:
        G = torch.abs(torch.cat((gradlist[0], gradlist[1]), axis=0))
    except IndexError:
        G = torch.abs((gradlist[0]))
    g_ij = np.mean(G.cpu().numpy(), axis=(-2, -1))
    g_j = np.mean(g_ij, axis=0)
    g_scaled = g_j / np.max(g_j)
    order = np.argsort(g_scaled)
    # r2all = r2_score(targetlist, predictedlist)
    # print('r2all', r2all)
    # corr_log = scipy.stats.pearsonr(targetlist, predictedlist)
    # print('model0 corr log ----: ', corr_log)
    # corr_rho = scipy.stats.spearmanr(targetlist, predictedlist)
    # targetlist = [np.exp(i) for i in targetlist]
    # predictedlist = [np.exp(i) for i in predictedlist]

    print('correlation: ', scipy.stats.pearsonr(targetlist, predictedlist))

    #
    corr = scipy.stats.pearsonr(targetlist, predictedlist)
    corr_rho = scipy.stats.spearmanr(targetlist, predictedlist)
    # ax[0].set_title('corr = %.2f'% corr[0] + ' r2 = %.2backwardf' % r2all)
    # ax0.set_title('Untrained Model: corr = %.2f' % corr[0] + '\n    (corr_'r'$\rho = %.2f$)'% corr_rho[0])

    # #
    p = model_0.state_dict()
    p_low = p['sinc_cnn2d.filt_b1']
    p_band = p['sinc_cnn2d.filt_band']
    #
    filt_beg_freq = (torch.abs(p_low) + 1 / 500)
    filt_end_freq = (filt_beg_freq + torch.abs(p_band) + 2 / 500)

    filt_beg_freq = filt_beg_freq.cpu().numpy() * 500
    filt_end_freq = filt_end_freq.cpu().numpy() * 500
    for i in range(0, 32):
        lines0, = ax9.plot([filt_beg_freq[i], filt_end_freq[i]], [i] * 2, ls = 'dashed',color = 'Blue')
        # if i == order[-1]:
        #     ax9.axvspan(filt_beg_freq[i], filt_end_freq[i], ymin =0.2, ymax = 0.5, color='darkred', alpha=0.5,
        #                      label='1st \n(filter %s)' % order[-1])
        #     print('1st: %s' % [filt_beg_freq[i], filt_end_freq[i]])
        # if i == order[-2]:
        #     ax9.axvspan(filt_beg_freq[i], filt_end_freq[i], ymin = i -3, ymax = i +3, color='red', alpha=0.5,
        #                      label='2nd \n(filter %s)' % order[-2])
        #     print('2nd: %s' % [filt_beg_freq[i], filt_end_freq[i]])
        # if i == order[-3]:
        #     ax9.axvspan(filt_beg_freq[i], filt_end_freq[i], ymin = i -3, ymax = i +3, color='plum', alpha=0.5,
        #                      label='3rd \n(filter %s)' % order[-3])
        #     print('3rd: %s' % [filt_beg_freq[i], filt_end_freq[i]])

    # ax9.set_title('subject %s' % finalsubIDs[s])
    ax9.set_title('Filters')
    # ax9.legend()
    ax9.set_xlabel('Frequency')
    # plt.show()

    p = []

    torch.manual_seed(seednum)
    np.random.seed(seednum)
    random.seed(seednum)


    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2 ** 32
        np.random.seed(worker_seed)
        random.seed(worker_seed)




    # %%

    g = torch.Generator()
    g.manual_seed(seednum)

    # model.apply(initialize_weights)

    criterion = nn.MSELoss()
    # criterion = nn.L1Loss()
    print('criterion: ', criterion)
    model = Sinc_Conv2d_attention_pre(dropout=dropout_rate).cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    classes = ('easy', 'medium', 'hard')
    # find the best patience and train
    ##########################################################################
    ###################### stage 2 training ##################################
    ##########################################################################
    # pick the best target loss
    # myfile = open(cross_val_dir + f'{finalsubIDs[subjectstart]}.pkl', 'rb')
    myfile = open(cross_val_dir + f'{finalsubIDs[subjectstart]}.pkl', 'rb')
    outfile = pickle.load(myfile)
    val_lossmin = -100
    for k in outfile.keys():
        newloss = outfile[k][2]
        if newloss > val_lossmin:
            val_lossmin = newloss
            train_lossmin = outfile[k][0]
            optimal_p = int(k)
    print('optimal patience for early stop was %s' % optimal_p, '\ntarget training loss is %s' % train_lossmin)

    n_total_steps = len(train_loader)
    # criterion = nn.CrossEntropyLoss().cuda()
    # criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    early_stopping = EarlyStopping(patience=20, verbose=True)


    ###########################################################################################
    ########################## load trained model ##############################################
    if notrainMode is True:
        model.load_state_dict(torch.load('/home/jenny/sincnet_eeg/' +  modelpath  +
      '/mymodel_%s' % finalsubIDs[s] + postname + '.pth'))
        model.eval()
    else:
        # grads_all = dict()
        # ave_grads_all = []
        # max_grads_all = []
        train_loss = []
        flag = False
        for epoch in range(num_epochs):
            epoch_acc = []
            epoch_loss = []
            # if flag is True:
            #     break
            for i, (data, target) in enumerate(train_loader):

                gradss = []
                # origin shape: [4, 3, 32, 32] = 4, 3, 1024
                # input_layer: 3 input channels, 6 output channels, 5 kernel size
                target = torch.squeeze((target))
                # target = target.long()

                try:
                    target = target.view(target.shape[0], 1)
                except IndexError:
                    test_target = test_target.view(test_target.shape[0], 1)

                batch_n = data.shape[0]
                # if (epoch==4) & (i ==1):
                #     print('break')
                #     flag = True
                #     break
                # Forward pass
                outputs, outputs_alpha,_ = model(data)



                # print('d',outputs,'a',outputs_alpha)
                # print('target', target)
                # print(outputs_alpha)
                # loss = criterion(outputs, target.cuda())
                # loss = my_loss(target,outputs,ndt, torch.mean(outputs_alpha),torch.mean(outputs_alpha)/2,1e-29)
                loss = my_loss(target.cuda(), outputs.cuda(), ndt, torch.mean(outputs_alpha,axis=0).cuda())
                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                max_grads, ave_grads, gradss, layers = plot_grad_flow(model.named_parameters())
                # print('average grads',ave_grads)
                optimizer.step()
                # model.sinc_cnn2d.filt_b1 = nn.Parameter(torch.clamp(model.sinc_cnn2d.filt_b1, 0, 0.025))
                # model.sinc_cnn2d.filt_band = nn.Parameter(torch.clamp(model.sinc_cnn2d.filt_band, 0, 0.016))

                epoch_loss.append(loss.detach().cpu().numpy())
                # _, predicted = torch.max(outputs.detach(), 1)
                # acc = (predicted == target.cuda()).sum() / predicted.shape[0]
                # epoch_acc.append(acc.cpu().numpy())

                print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{n_total_steps}], Loss: {loss.item():.4f}')
            tb.add_scalar('training loss', epoch_loss[-1], epoch)
            viz_histograms(model, epoch)

                ##################### save the gradient ###################
                # print('grads', gradss[1][1].detach().cpu())
                # if epoch == 0 and i == 0:
                #     print('create dic')
                #     for l in layers:
                #         grads_all[l] = []

                # for l, j in enumerate(layers):
                #     grads_all[j].append(gradss[l].detach().cpu())
                # ave_grads_all.append(ave_grads)
                # max_grads_all.append(max_grads)

            train_loss.append(loss.detach().cpu().numpy())
            lossmean = np.mean(epoch_loss)

            #
            # early_stopping(lossmean, model)
            # if early_stopping.early_stop:
            #     print("Early stopping")
            #     break
            #
            # print('pretrain target loss', pretrain_loss[-1])
            # print(lossmean)
            # if lossmean < train_lossmin:
            #     print('reached minimum')
            #     break
            early_stopping(lossmean, model)
            if early_stopping.early_stop:
                print("Early stopping")
                break

    # ### save the model
    # p = model.state_dict()
    # torch.save(p, modelpath + '/' + 'mymodel_%s' % finalsubIDs[s] + postname + '.pth')

    # %%
    ##########################################################################
    ########################## final testing #################################
    ##########################################################################

    # read my model
    # model = Sinc_Conv2d_new().cuda()
    # model.load_state_dict(torch.load(modelpath + '/' + 'mymodel_%s'%finalsubIDs[s] + postname + '.pth'))
    # torch.save(p, modelpath + '/' + 'mymodel_%s'%finalsubIDs[s] + postname + '.pth')

    targetlist = []
    predictedlist = []
    predictedlist_alpha =[]
    plt.rcParams.update({'font.size': 17})
    #
    # fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    gradlist = []
    gradtemplist = []
    pred_copy = []
    # model = Sinc_Conv2d_new().cuda()
    model.eval()
    criterion = nn.MSELoss()
    n_total_steps = len(test_loader)
    trial_grad_list = []
    for i, (test_data, test_target) in enumerate(test_loader):
        cond_target = y_test_cond[i*batch_size+test_target.shape[0]-test_target.shape[0]:i*batch_size+test_target.shape[0]]

        #     # test_data, test_target = next(iter(test_loader))
        pred, pred_1,_ = model(test_data.cuda())
        pred_1_copy = pred_1.detach().cpu()
        pred_copy = pred.detach().cpu()
        # pred.backward(gradient=torch.ones(64, 1).cuda())
        pred.mean().backward()
        gradients = model.get_activations_gradient_filter()
        gradients_temp = model.get_activations_gradient_temp()
        gradlist.append(gradients)
        gradtemplist.append(gradients_temp)
        test_target = torch.squeeze((test_target))
        if cond_target.shape[0] == 1:
            test_target = test_target.view(1, 1)
        else:
            test_target = test_target.view(test_target.shape[0], 1)

        # test_loss = my_loss(test_target.cuda(), pred_copy.cuda(),ndt, alpha,alpha/2, err = 1e-29)
        test_loss = my_loss(test_target, pred_copy.cuda(), ndt, torch.mean(pred_1, axis=0).cuda())

        r2 = r2_score(test_target.cpu().detach().numpy(), pred_copy.cpu().detach().numpy())
        # print("validation accuracy: ", val_acc)
        # print("validation loss: ", val_loss)
        # valacc_batch.append(val_acc.cpu())
        try:
            targetlist += torch.squeeze(test_target).tolist()
            predictedlist += torch.squeeze(-pred_copy).cpu().tolist()
            predictedlist_alpha += torch.squeeze(pred_1_copy).cpu().tolist()
        except TypeError:
            targetlist += [torch.squeeze(test_target).tolist()]
            predictedlist += [torch.squeeze(-pred_copy).cpu().tolist()]
            predictedlist_alpha +=[torch.squeeze(pred_1_copy).cpu().tolist()]

        print(f'Testing Batch: {i}, Step [{i + 1}/{n_total_steps}], Loss: {test_loss.item():.4f}, R^2 : {r2}')
        # if i % 1 == 0:
        #     # plt.plot(test_target, label='target')
        #     # plt.plot(test_output.cpu().detach().numpy(), label='predicted')
        #     ax2.scatter(test_target, -pred_copy.cpu().detach().numpy(), color='b')
        #     ax2.set_xlabel('actual RT')
        #     ax2.set_ylabel('predicted Drift')
            # ax[0].scatter(test_target, test_output.cpu().detach().numpy(), color ='b')
    # corr_log1 = scipy.stats.pearsonr(targetlist, predictedlist)
    targetlist = np.array(targetlist)
    predictedlist = np.array(predictedlist)
    predictedlist_alpha = np.array(predictedlist_alpha)

    # ax2.scatter(targetlist[y_test_cond==1], predictedlist[y_test_cond==1], color='green', marker = 'o',)
    # ax2.scatter(targetlist[y_test_cond==2], predictedlist[y_test_cond==2], color='blue', marker = '*')
    # ax2.scatter(targetlist[y_test_cond==3], predictedlist[y_test_cond==3], color='red', marker = '^')
    # ax2.scatter(1/targetlist[targetlist>0], predictedlist[targetlist>0], marker = 'o', color = 'blue')
    # ax2.scatter(-1/targetlist[targetlist<0], predictedlist[targetlist<0], marker = 'o',color = 'green')
    #
    # # ax2.axhline(np.median(predictedlist_alpha))
    # ax2.set_xlabel('1/RT')
    # ax2.set_ylabel('predicted Drift')
    # print('corr log1: ', corr_log1)
    # targetlist = [np.exp(i) for i in targetlist]
    # predictedlist = [np.exp(i) for i in predictedlist]
    corr1 = scipy.stats.pearsonr(1/targetlist[targetlist>0], predictedlist[targetlist>0])
    corr_rho1 = scipy.stats.spearmanr(1/targetlist[targetlist>0], predictedlist[targetlist>0])

    corr1 = scipy.stats.pearsonr(np.abs(1/targetlist), predictedlist)
    corr_rho1 = scipy.stats.spearmanr(np.abs(1/targetlist), predictedlist)

    r2all = r2_score(targetlist, predictedlist)
    print('r2all', r2all)
    print('correlation exp: ', corr1)

    # ax2.set_title('Trained Model: corr = %.2f' % corr1[0] + '\n    (corr_'r'$\rho = %.2f$)' % corr_rho1[0] + '\n Boundary:%.3f'% np.median(predictedlist_alpha))

    #
    # for j in range(test_data.shape[0]):
    #     test_data_trial = test_data[[j],:,:]
    #     test_target_trial = test_target[[j],:]
    #     model.eval()
    #     pred_trial = model(test_data_trial.cuda())
    #     pred_trial.backward()
    #     trial_grad = model.get_activations_gradient()
    #     trial_grad_mean = torch.mean(torch.abs(trial_grad.detach().cpu()), axis = (-2,-1))
    #     trial_grad_list.append(trial_grad_mean)

    err = np.sqrt(np.subtract(targetlist, predictedlist) ** 2)
    threshold = np.percentile(err, 90)
    errind = [i for i, j in enumerate(err) if j < threshold]

    try:
        G = torch.abs(torch.cat((gradlist[0], gradlist[1]), axis=0))
    except IndexError:
        G = torch.abs((gradlist[0]))
    g_ij = np.mean(G.cpu().numpy(), axis=(-2, -1))
    g_j = np.mean(g_ij[errind, :], axis=0)
    g_scaled = g_j / np.max(g_j)
    order = np.argsort(g_scaled)
    try:
        Gt = torch.abs((torch.cat((gradtemplist[0], gradtemplist[1]), axis=0)))
        Gtemp = torch.squeeze(abs((torch.cat((gradtemplist[0], gradtemplist[1]), axis=0))))
    except IndexError:
        Gt = torch.abs((gradtemplist[0]))
        Gtemp = torch.squeeze(abs(((gradtemplist[0]))))
    Gt = Gt[errind, :, :, :].mean(axis=0)
    Gt1 = Gt[order[-1] * 2:order[-1] * 2 + 2, :, :]
    Gt1max = torch.argmax(Gt1, axis=2).detach().cpu().numpy()
    Gt2 = Gt[order[-2] * 2:order[-2] * 2 + 2, :, :]
    Gt2max = torch.argmax(Gt2, axis=2).detach().cpu().numpy()
    Gt3 = Gt[order[-3] * 2:order[-3] * 2 + 2, :, :]
    Gt3max = torch.argmax(Gt3, axis=2).detach().cpu().numpy()

    Gt = torch.squeeze(Gt.cpu())
    Gtmean = torch.mean(Gt, axis=1)
    Gtresult = torch.matmul(Gtmean, Gt)
    Gorder = torch.argsort(Gtresult)

    # draw the estimated training from

    result = []
    out = []
    train_target= []
    out_alpha = []
    for i, (data, target) in enumerate(train_loader):
        gradss = []
        # origin shape: [4, 3, 32, 32] = 4, 3, 1024
        # input_layer: 3 input channels, 6 output channels, 5 kernel size
        target = torch.squeeze((target))
        # target = target.long()

        try:
            target = target.view(target.shape[0], 1)
        except IndexError:
            test_target = test_target.view(test_target.shape[0], 1)

        batch_n = data.shape[0]

        # Forward pass
        outputs, outputs_1,_ = model(data)
        # print(outputs)
        train_drift = torch.squeeze(-outputs).cpu().tolist()
        out += train_drift
        train_rt= torch.squeeze(target).cpu().tolist()
        train_target += train_rt
        train_alpha = torch.squeeze(outputs_1).cpu().tolist()
        out_alpha += train_alpha
        #
        # result.append(torch.mean(-outputs[:, 0]).detach().cpu().tolist())
        # result.append(torch.mean(-outputs[:, 0]).detach().cpu().tolist())

    print('mean drift from training:---------------- ', np.mean(out))
    out = np.asarray(out)
    train_target = np.asarray(train_target)
    out_alpha = np.asarray(out_alpha)

    # ax0.scatter(1/np.array(train_target[train_target>0]), out[train_target>0], label = 'correct',color = 'blue')
    # ax0.scatter(-1/np.array(train_target[train_target<0]), out[train_target<0], color = 'green', label = 'incorrect')
    # # ax0.axhline(np.median(out[train_target>0]), label = 'median drift',linewidth = 4)
    # ax0.axhline(np.median(out), label = 'median drift',linewidth = 4)
    #
    # ax0.axhline(drift, label = 'MCMC drift',color='red',linewidth = 4)

    # corr_train = scipy.stats.pearsonr(1/np.array(train_target[train_target>0]), out[train_target>0])
    # corr_rho_train = scipy.stats.spearmanr(1/np.array(train_target[train_target>0]), out[train_target>0])


    corr_train = scipy.stats.pearsonr(np.abs(1/np.array(train_target)), out)
    corr_rho_train = scipy.stats.spearmanr(np.abs(1/np.array(train_target)), out)



    # %%
    p = model.state_dict()
    p_low = p['sinc_cnn2d.filt_b1']
    p_band = p['sinc_cnn2d.filt_band']
    #
    filt_beg_freq = (torch.abs(p_low) + 1 / 500)
    filt_end_freq = (filt_beg_freq + torch.abs(p_band) + 2 / 500)

    filt_beg_freq = filt_beg_freq.cpu().numpy() * 500
    filt_end_freq = filt_end_freq.cpu().numpy() * 500
    for i in range(0, 32):
    #     # if i == order[-1]:
    #     #     ax3.axvspan(filt_beg_freq[i], filt_end_freq[i], color='red', alpha=0.8,
    #     #                 label='1st' % order[-1],zorder= 3)
    #     #     print('1st: %s' % [filt_beg_freq[i], filt_end_freq[i]])
    #     # if i == order[-2]:
    #     #     ax3.axvspan(filt_beg_freq[i], filt_end_freq[i], color='darkorange', alpha=0.8,
    #     #                 label='2nd' % order[-2], zorder= 2)
    #     #     print('2nd: %s' % [filt_beg_freq[i], filt_end_freq[i]])
    #     # if i == order[-3]:
    #     #     ax3.axvspan(filt_beg_freq[i], filt_end_freq[i], color='turquoise', alpha=0.8,
    #     #                 label='3rd' % order[-3], zorder= 1)
    #     #     print('3rd: %s' % [filt_beg_freq[i], filt_end_freq[i]])
    #
        lines1, =ax9.plot([filt_beg_freq[i], filt_end_freq[i]], [i] * 2,linewidth = 5,color ='Red')
    lines0.set_label('Untrained')
    lines1.set_label('Trained')
    ax9.legend(loc=1, bbox_to_anchor=(1, 1),fontsize = 'small',handlelength=1)
    # # ax10.set_title('subject %s' % finalsubIDs[s])
    # # ax10.set_title('Trained Filters')
    # #
    # # # ax10.legend(loc='lower left')
    # # ax10.set_xlabel('Frequency')
    # # ax10.get_xlim()
    # # ax9.set_xlim(ax10.get_xlim())
    #
    #
    # # filt_sort = np.argsort(filt_beg_freq)
    # # count = 0
    # # for i in filt_sort:
    # #     ax[1].plot([filt_beg_freq[i] , filt_end_freq[i]], [count]*2)
    # #     count +=1
    # # ax[1].set_title('subject %s' % finalsubIDs[s])
    #
    # # results[finalsubIDs[s]] = dict()
    # # results[finalsubIDs[s]] = {'filt_beg_freq': filt_beg_freq, 'filt_end_freq': filt_end_freq,
    # #                            'corr': corr1, 'corr_rho': corr_rho1, 'filter_grads': G, 'temporal_grads': Gt,
    # #                            'chan_weights': torch.squeeze(p['separable_conv.depthwise.weight']).cpu()}
    #
    # sub_dict= dict()
    # sub_dict = {'filt_beg_freq': filt_beg_freq, 'filt_end_freq': filt_end_freq,
    #                            'corr': corr1, 'corr_rho': corr_rho1, 'filter_grads': G.cpu().numpy(), 'temporal_grads': Gtemp.cpu().numpy(),
    #                            'chan_weights': torch.squeeze(p['separable_conv.depthwise.weight']).cpu().numpy(),
    #             'target_rt_test': targetlist, 'delta_test': predictedlist, 'alpha_test': predictedlist_alpha,
    #             'target_rt_train': np.array(train_target),  'delta_train': np.array(out) , 'alpha_train':np.array(out_alpha)
    #             }
    # # savemat(resultpath + '/%s' % finalsubIDs[s][0:-1]+ '_results' + postname + '.mat', sub_dict)
    # my_file = open(resultpath + f'/%s' % finalsubIDs[s][0:-1]+ '_results' + postname + '.pkl', 'wb')
    # pickle.dump(sub_dict, my_file)
    # my_file.close()
    #
    # from topo import *
    # from mpl_toolkits.axes_grid.inset_locator import inset_axes
    #
    # # plt.show()
    # # filter analysis
    # goodchan = chansets_new()
    #
    #
    # def getweights(i):
    #     filt_ind = order[i]
    #     weight1 = p['separable_conv.depthwise.weight'][filt_ind * 2:(filt_ind * 2 + 2), :, :, :]
    #     weight1 = torch.squeeze(weight1.cpu())
    #     maxweight1 = np.argmax(weight1, axis=1)
    #     chan0, chan1 = goodchan[maxweight1[0]], goodchan[maxweight1[1]]
    #     print(chan0, chan1)
    #     return weight1
    #
    #
    # weight1 = getweights(-1)
    # w = weight1.argsort(axis=1)
    # axin11 = ax4.inset_axes([0.01, 0.2, 0.3, 0.6])
    # axin12 = ax4.inset_axes([0.35, 0.2, 0.3, 0.6])
    # axin13 = ax4.inset_axes([0.7, 0.2, 0.3, 0.6])
    #
    # plottopo(weight1[0, :].reshape(num_chan, 1), axin11, w[:, -1], 'red', 12)
    # plottopo(weight1[0, :].reshape(num_chan, 1), axin11, w[:, -2], 'red', 10)
    # plottopo(weight1[0, :].reshape(num_chan, 1), axin11, w[:, -3], 'red', 8)
    #
    #
    # # axin11.set_title('1st filter weights')
    #
    #
    # weight2 = getweights(-2)
    # w2 = weight2.argsort(axis=1)
    #
    # plottopo(weight2[0, :].reshape(num_chan, 1), axin12, w2[:, -1], 'darkorange', 12)
    # plottopo(weight2[0, :].reshape(num_chan, 1), axin12, w2[:, -2], 'darkorange', 10)
    # plottopo(weight2[0, :].reshape(num_chan, 1), axin12, w2[:, -3], 'darkorange', 8)
    # # axin12.set_title('2nd filter weights')
    #
    # weight3 = getweights(-3)
    # w3 = weight3.argsort(axis=1)
    #
    # plottopo(weight2[0, :].reshape(num_chan, 1), axin13, w3[:, -1], 'turquoise', 12)
    # plottopo(weight2[0, :].reshape(num_chan, 1), axin13, w3[:, -2], 'turquoise', 10)
    # plottopo(weight2[0, :].reshape(num_chan, 1), axin13, w3[:, -3], 'turquoise', 8)
    # # axin13.set_title('3rd filter weights')
    #
    # def crit_timewindow(time_index, stride, window_length, conv_length, tend, tstart):
    #     '''this retursn the crtiical time window, where 0 is the beginning of the window used
    #     e.g., 0 for 625 to 1625 would be 625 (1250ms)
    #     but if tstart is not 625, but smaller, for example 375, t1 would be -500'''
    #     t1, t2 = (time_index * stride) / conv_length * (tend - tstart) * 2, \
    #              (time_index * stride + window_length) / conv_length * (tend - tstart) * 2
    #     return t1 + (-2*(625-tstart)) , t2+(-2*(625-tstart))
    #
    # # ax5.clear()
    # if timestart < 625:
    #     ax5.plot(np.arange(2*(timestart - 625),(timeend-625)*2, 2),data[0,:,:].T, alpha=0.8,color = 'silver')
    #     ax5.set_xticks(ticks = np.arange(2*(timestart - 625),(timeend-625)*2+1, 250))
    # else:
    #     ax5.plot(np.arange(0,trialdur, 2),data[0,:,:].T, alpha=0.8,color = 'silver')
    #     ax5.set_xticks(ticks = np.arange(0,trialdur+1, 250))
    # # ax5.set_xticklabels(labels = np.arange(timestart*2, timeend*2+1, 250))
    # ax5.set_xlabel('Stimulus Locked time (ms)')
    # yminn = ax5.get_ylim()
    # # ax5.add_
    # ax51 = ax5.twinx()
    # ax51.set_ylim(0,500)
    # ax51.hist(rtall*1000, bins=12, color = 'green')
    # ax51.get_yaxis().set_visible(False)
    # f1 = int(0.5* (filt_beg_freq[order[-1]]+filt_end_freq[order[-1]]))
    # f2 =int(0.5* (filt_beg_freq[order[-2]]+filt_end_freq[order[-2]]))
    # f3= int(0.5* (filt_beg_freq[order[-3]]+filt_end_freq[order[-3]]))
    #
    # convlength = data.shape[-1] - model.filter_length + 1
    # time11, time11end = crit_timewindow(Gt1max[0], model.pool1.stride[1], model.pool1.kernel_size[1], convlength,
    #                                     timeend, timestart)
    # time12, time12end = crit_timewindow(Gt1max[1], model.pool1.stride[1], model.pool1.kernel_size[1], convlength,
    #                                     timeend,
    #                                     timestart)
    #
    # ax5.axvspan(time11, time11end, ymin = 0.9, ymax = 1, alpha=0.8, color='red', label='1st filter (%.0f Hz)'% f1, zorder=2)
    # ax5.axvspan(time12, time12end, ymin = 0.8, ymax = 0.9,alpha=0.8, color='red',zorder=2)
    #
    # time21, time21end = crit_timewindow(Gt2max[0], model.pool1.stride[1], model.pool1.kernel_size[1], convlength,
    #                                     timeend, timestart)
    # time22, time22end = crit_timewindow(Gt2max[1], model.pool1.stride[1], model.pool1.kernel_size[1], convlength,
    #                                     timeend,
    #                                     timestart)
    #
    # ax5.axvspan(time21, time21end, ymin = 0.7, ymax = 0.8, alpha=0.8, color='darkorange', label='2nd filter (%.0f Hz)'% f2, zorder=3)
    # ax5.axvspan(time22, time22end, ymin = 0.6, ymax = 0.7,alpha=0.8, color='darkorange',zorder=3)
    #
    # time31, time31end = crit_timewindow(Gt3max[0], model.pool1.stride[1], model.pool1.kernel_size[1], convlength,
    #                                     timeend,
    #                                     timestart)
    # time32, time32end = crit_timewindow(Gt3max[1], model.pool1.stride[1], model.pool1.kernel_size[1], convlength,
    #                                     timeend,
    #                                     timestart)
    #
    # ax5.axvspan(time31, time31end, ymin = 0.5, ymax = 0.6, alpha=0.8, color='turquoise', label='3rd filter (%.0f Hz)'% f3,zorder=4)
    # ax5.axvspan(time32, time32end, ymin = 0.4, ymax = 0.5, alpha=0.8, color='turquoise', zorder=4)
    # ax7.set_title('Channel Weights')
    # ax5.set_title('Critical Time Period')
    # ax5.set_xlabel('Stimulus Locked time (ms)')
    # ax5.legend()
    #
    # if compute_likelihood is True:
    #     ax4.clear()
    #     axin11 = ax4.inset_axes([0.01, 0.3, 0.9, 0.5])
    #     axin12 = ax4.inset_axes([0.01, 0.1, 0.9, 0.5])
    #     axin11.set_axis_off()
    #     axin12.set_axis_off()
    #
    #
    #     from my_wfpt import neg_wfpt
    #     rt_train = sub_dict['target_rt_train']
    #     rt_test = sub_dict['target_rt_test']
    #     drift_train = sub_dict['delta_train']
    #     alpha_train = sub_dict['alpha_train']
    #     drift_test = sub_dict['delta_test']
    #     alpha_test = sub_dict['alpha_test']
    #     ndt = np.min(rt_train) * 0.93
    #
    #     ll_trial_trainRT = []
    #     for i, j in enumerate(rt_train):
    #         l = neg_wfpt(j, drift_train[i], ndt, alpha_train[i])
    #         # print(l)
    #         ll_trial_trainRT += [l]
    #     ll_trial_trainRT = np.sum(ll_trial_trainRT)
    #
    #     ll_median_trainRT = []
    #     for i, j in enumerate(rt_train):
    #         l = neg_wfpt(j, np.median(drift_train), ndt, np.median(alpha_train))
    #         # print(l)
    #         ll_median_trainRT += [l]
    #     ll_median_trainRT = np.sum(ll_median_trainRT)
    #
    #     ll_trial_drift_trainRT = []
    #     for i, j in enumerate(rt_train):
    #         l = neg_wfpt(j, drift_train[i], ndt, np.median(alpha_train))
    #         # print(l)
    #         ll_trial_drift_trainRT += [l]
    #     ll_trial_drift_trainRT = np.sum(ll_trial_drift_trainRT)
    #
    #     ll_trial_alpha_trainRT = []
    #     for i, j in enumerate(rt_train):
    #         l = neg_wfpt(j, np.median(drift_train), ndt, alpha_train[i])
    #         # print(l)
    #         ll_trial_alpha_trainRT += [l]
    #     ll_trial_alpha_trainRT = np.sum(ll_trial_alpha_trainRT)
    #
    #     ############## test data ##################################################
    #     ll_median_train_on_testRT = []  # median drift and rt on test
    #     for i, j in enumerate(rt_test):
    #         l = neg_wfpt(j, np.median(drift_train), ndt, np.median(alpha_train))
    #         # print(l)
    #         ll_median_train_on_testRT += [l]
    #     ll_median_train_on_testRT = np.sum(ll_median_train_on_testRT)
    #
    #     ll_trial_train_on_testRT = []  # trial drift and alpha on test
    #     for i, j in enumerate(rt_test):
    #         l = neg_wfpt(j, drift_test[i], ndt, alpha_test[i])
    #         # print(l)
    #         ll_trial_train_on_testRT += [l]
    #     ll_trial_train_on_testRT = np.sum(ll_trial_train_on_testRT)
    #
    #     ll_trial_drift_testRT = []  # trial drift and alpha on test
    #     for i, j in enumerate(rt_test):
    #         l = neg_wfpt(j, drift_test[i], ndt, np.median(alpha_test))
    #         # print(l)
    #         ll_trial_drift_testRT += [l]
    #     ll_trial_drift_testRT = np.sum(ll_trial_drift_testRT)
    #
    #     ll_trial_alpha_testRT = []  # trial drift and alpha on test
    #     for i, j in enumerate(rt_test):
    #         l = neg_wfpt(j, np.median(drift_test), ndt, alpha_test[i])
    #         # print(l)
    #         ll_trial_alpha_testRT += [l]
    #     ll_trial_alpha_testRT = np.sum(ll_trial_alpha_testRT)
    #
    #     ll_median_testRT = []  # trial drift and alpha on test
    #     for i, j in enumerate(rt_test):
    #         l = neg_wfpt(j, np.median(drift_test), ndt, np.median(alpha_test))
    #         # print(l)
    #         ll_median_testRT += [l]
    #     ll_median_testRT = np.sum(ll_median_testRT)
    #
    #     ax4.set_axis_off()
    #
    #     dataTrain = [ ll_median_trainRT, ll_trial_trainRT,ll_trial_drift_trainRT, ll_trial_alpha_trainRT]
    #     dataTrain = [[round(i,3) for i in dataTrain]]
    #
    #     rows = ['Sum of NLL']
    #     axin11.set_axis_off()
    #     columns = (r'$\overline{\delta}, \overline{\alpha}\;| RT_{train}$', r'$\delta_i, \alpha_i | RT_{train}$',
    #                r'$\delta_i, \overline{\alpha}\;| RT_{train}$', r'$\overline{\delta}, \alpha_i\;| RT_{train}$')
    #     the_table = axin11.table(cellText=dataTrain,
    #                              # rowLabels=rows,
    #                              colLabels=columns, cellLoc='center', loc='top', fontsize=16)
    #     the_table.set_fontsize(14)
    #
    #     cellDict = the_table.get_celld()
    #     for i in range(0, len(columns)):
    #         cellDict[(0, i)].set_height(.4)
    #         for j in range(1, len(dataTrain) + 1):
    #             cellDict[(j, i)].set_height(.3)
    #     # cellDict[(1, -1)].set_height(.1)
    #     cellDict[(0, np.argmin(dataTrain[0]))].set_facecolor("#56b5fd")
    #     cellDict[(1, np.argmin(dataTrain[0]))].set_facecolor("#56b5fd")
    #     the_table.set_fontsize(16)
    #     the_table.scale(1.2, 1.2)
    #
    #     dataTest = [ll_median_train_on_testRT, ll_median_testRT, ll_trial_train_on_testRT,ll_trial_drift_testRT, ll_trial_alpha_testRT]
    #     dataTest = [[round(i, 3) for i in dataTest]]
    #     columnsT = (r'$\overline{\delta}, \overline{\alpha}\;| RT_{test}$',
    #                r'$\overline{\hat{\delta}}, \overline{\hat{\alpha}}\;| RT_{test}$',
    #                r'$\hat{\delta_i}, \hat{\alpha_i} | RT_{test}$',
    #                r'$\hat{delta_i}, \overline{\hat{\alpha}}\;| RT_{test}$',
    #                r'$\overline{\hat{\delta}}, \hat{\alpha_i}\;| RT_{test}$')
    #
    #     # rows = ['Sum of NLL']
    #     axin12.set_axis_off()
    #     test_table = axin12.table(cellText=dataTest,
    #                              # rowLabels=rows,
    #                              colLabels=columnsT, cellLoc='center', loc='center', fontsize=16)
    #     test_table.set_fontsize(16)
    #
    #     cellDict = test_table.get_celld()
    #     for i in range(0, len(columnsT)):
    #         cellDict[(0, i)].set_height(.4)
    #         for j in range(1, len(dataTest) + 1):
    #             cellDict[(j, i)].set_height(.3)
    #     # cellDict[(1, -1)].set_height(.1)
    #     cellDict[(0, 0)].set_facecolor("lightgrey")
    #     cellDict[(1, 0)].set_facecolor("lightgrey")
    #     cellDict[(0, 1+np.argmin(dataTest[0][1:]))].set_facecolor("#56b5fd")
    #     cellDict[(1, 1+np.argmin(dataTest[0][1:]))].set_facecolor("#56b5fd")
    #     test_table.set_fontsize(16)
    #     test_table.scale(1.2, 1.2)
    #
    #     # axin11.set_title('Sum of Negative Log Likelihood')
    #     axin12.text(0,-1, r'$\overline{\delta}, \overline{\alpha}$ are median of trial estimates from training data,'
    #                                +'\n'+r'$\delta_i, \alpha_i$ are trial estimates from training data,'
    #                           +'\n'+        r'$\overline{\hat{\delta}}, \overline{\hat{\alpha_i}}$ are median of trial estimates from testing data,'
    #                            +'\n'+      r'$\hat{\delta_i}, \hat{\alpha_i}$ are trial estimates from testing data'
    #                 +'\nblue indicates the best model', fontsize = 12)
    #     axin11.text(0,2,'Sum of Negative Log Likelihood')
    # ax7.set_axis_off()
    # fig.text(0.04, 0.93, 'A', transform=fig.transFigure, horizontalalignment='center',fontsize=font, weight= 'bold')
    # fig.text(0.04, 0.5, 'B', transform=fig.transFigure, horizontalalignment='center',fontsize=font, weight= 'bold')
    #
    # fig.tight_layout()
    # fig2.tight_layout()
    #
    # # fig.savefig(figurepath + '/%s' % finalsubIDs[s][0:-1] + postname + '.png')
    #
    # plt.show()
    #
    #
    #
# savemat(modelpath + '/results_motorsets' + postname + '.mat', results)