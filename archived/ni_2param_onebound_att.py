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
from delta_bound_models import *
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

import os
os.environ['CUDA_VISIBLE_DEVICES']='0,1'
# Device configuration
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
print(device)


############################# define random seeds ###########################

seednum = 2022
torch.manual_seed(seednum)
np.random.seed(seednum)
random.seed(seednum)
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(seednum)

############################ define model parameters ######################
timestart = 0
timeend = 500
trialdur = timeend * 2 - timestart * 2
correctModel = True   # whether target is signed or not
notrainMode = True    # if true, just load the model
sr = 500
# timeend = 800 # when 300ms after stim

# Hyper-parameters
num_epochs = 300
batch_size = 64
learning_rate = 0.001
num_chan = 98
dropout_rate = 0.5
compute_likelihood = False





######################## tensorbaord initilization ###########################
tb = SummaryWriter('runs/regression_new')
model_0 = SincDriftBoundAtt(dropout=dropout_rate).cuda()
model = SincDriftBoundAtt(dropout=dropout_rate).cuda()


######################## creating directory and file nmae ############for s########
# postname = '_prestim500_1000_0123_ddm_2param'
postname = '_ni_2param_onebound1'
# postname = '_1000_0123_ddm_2param_final'

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
    path = '/home/jenny/sincnet_eeg/ni_data/exp5data/'
    allDataFiles = os.listdir(path)
    finalsub = [i[:-4] for i in allDataFiles]
    finalsub.sort()
    return finalsub

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
    path = '/home/jenny/sincnet_eeg/ni_data/exp5data/'
    datadict = pickle.load(open(path + subID + '.pkl','rb'))
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
    data = np.array(datadict['eegdata'])
    data = data[::2, :, :]
    sr = np.array(datadict['sr']) / 2
    condition = np.array(datadict['snr'])
    # goodtrials = np.array(datadict['trials'])[0]
    correct = np.array(datadict['acc'])
    rt = np.array(datadict['rt'])

    # goodchan = goodchans()
    # goodchan = chanmotor()
    goodchan = chansets_new()
    # data = data[:, :, correct==1]
    # condition = condition[correct==1]
    data = data[:, goodchan, :]
    return data[Tstart:Tend, :, :], condition, rt, correct


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

def correlation_loss(y_pred, y_true):
    x = y_pred.clone()
    y = y_true.clone()
    vx = x - torch.mean(x)
    vy = y - torch.mean(y)
    cov = torch.sum(vx * vy)
    corr = cov / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)) + 1e-12)
    return corr

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
    tt = torch.max(torch.tensor(torch.abs(t.cuda()) - t0.cuda()),err) / torch.max(err,a.cuda()) ** 2  # normalized time
    tt_vec = torch.tile(tt, (1, 10))
    pp = torch.cumsum((w+2*k)*torch.exp(-(((w+2*k)**2)/2)/tt_vec),axis=1)
    pp = pp[:,-1]/torch.sqrt(2*torch.tensor(np.pi)*torch.squeeze(tt)**3)
    pp = pp[:, None]
    # v = torch.where(torch.tensor(t).cuda()>0, v, -v)   # if time is negative, flip the sign of v
    v = torch.clamp(v, -6,6)



    # t = torch.where(torch.tensor(t).cuda() > 0, torch.tensor(t).cuda(), torch.tensor(-t).cuda())
    p = (pp * (torch.exp(-v*torch.max(err, a)*w - (v**2)*torch.tensor(t).cuda()/2) /(torch.max(err,a)**2)))
    # p = torch.where(torch.tensor(v).cuda()>0, 1*p, 6*p)
    p = torch.log(p)
    # p = torch.where(torch.tensor(v).cuda()>0, p, -p)
    # print(t,a,v)
    # print('probability is ', p)
    return -(p.sum())


def my_loss_choice(t, v, t0, a):
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
    tt = torch.max(torch.tensor(torch.abs(t.cuda()) - t0.cuda()),err) / torch.max(err,a.cuda()) ** 2  # normalized time
    tt_vec = torch.tile(tt, (1, 10))
    pp = torch.cumsum((w+2*k)*torch.exp(-(((w+2*k)**2)/2)/tt_vec),axis=1)
    pp = pp[:,-1]/torch.sqrt(2*torch.tensor(np.pi)*torch.squeeze(tt)**3)
    pp = pp[:, None]
    v = torch.where(torch.tensor(t).cuda()<0, -v, v)   # if time is negative, flip the sign of v
    v = torch.clamp(v, -6,6)
    t = torch.where(torch.tensor(t).cuda() > 0, torch.tensor(t).cuda(), torch.tensor(-t).cuda())
    p = (pp * (torch.exp(-v*torch.max(err, a)*w - (v**2)*torch.tensor(t).cuda()/2) /(torch.max(err,a)**2)))
    # p = torch.where(torch.tensor(v).cuda()>0, 1*p, 6*p)
    p = torch.log(p)
    # p = torch.where(torch.tensor(v).cuda()>0, p, -p)
    print(v)
    # print('probability is ', p)
    return -(p.sum())

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
finalsubIDs = getIDs()
mylist = np.arange(0, len(finalsubIDs))
############################################
############### set subject ######################
############################################
for s in range(1, 2):
    # a results dictionary for storing all the data
    finalsubIDs = getIDs()
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
    print(str(subjectstart) + '/' + 'subjectID: ' + finalsubIDs[subjectstart])
    data, cond, rt,correct = getrtdata(datadict, timestart, timeend)
    # response = loadinfo(finalsubIDs[subjectstart])
    rtall = rt.copy()
    correct = correct.astype('int')
    if correctModel is True:
        rt = (correct * 2 - 1) * rt
    # correctind = condition>0
    #
    newdata = reshapedata(data).astype('float32')
    rt = rt*0.001
    #
    # condition = condition[correctind]
    # newdata = newdata[correctind,:,:]
    # cond = cond[correctind]

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




    X_train0, X_test, y_train0, y_test = train_test_split(newdata, rt, test_size=0.2, random_state=42)
    ndt = np.min(np.abs(y_train0)) * 0.93
    ndt = torch.tensor(ndt)
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


    # %% optimization
    g = torch.Generator()
    g.manual_seed(seednum)

    # model.apply(initialize_weights)

    # criterion = nn.MSELoss()
    # criterion = nn.L1Loss()
    # print('criterion: ', criterion)

    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    classes = ('easy', 'medium', 'hard')

    n_total_steps = len(train_loader)
    # criterion = nn.CrossEntropyLoss().cuda()
    # criterion = nn.MSELoss()
    weight_decay = 1e-2

    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # optimize different parameters
    alpha_param = []
    for n, p in model.named_parameters():
        if "fc4" in n:
            alpha_param.append(p)

    drift_param = []
    for n, p in model.named_parameters():
        if "fc4" not in n:
            drift_param.append(p)
    #
    optimizer_drift = torch.optim.Adam(drift_param, lr=learning_rate)
    optimizer_alpha = torch.optim.Adam(alpha_param, lr=learning_rate,weight_decay= weight_decay)

    early_stopping = EarlyStopping(patience=8, verbose=True)

    ###########################################################################################
    ########################## load trained model ##############################################
    if notrainMode is True:
        model.load_state_dict(torch.load('/home/jenny/sincnet_eeg/' + modelpath +
                                         '/mymodel_%s' % finalsubIDs[s] + postname + '.pth'))
        model.eval()
    else:
        train_loss = []
        flag = False
        for epoch in range(num_epochs):
            epoch_acc = []
            epoch_loss = []
            # if flag is True:
            #     break
            for i, (data, target) in enumerate(train_loader):

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
                outputs, outputs_alpha, y_train = model(data)
                target_ = torch.abs(target)

                # print('d',outputs,'a',outputs_alpha)
                # print('target', target)
                # print(outputs_alpha)
                # loss = criterion(outputs, target.cuda())
                # loss = my_loss(target,outputs,ndt, torch.mean(outputs_alpha),torch.mean(outputs_alpha)/2,1e-29)
                # loss = my_loss(target.cuda(), outputs.cuda(), ndt, torch.mean(outputs_alpha,axis=0).cuda())
                # loss = my_loss(torch.abs(target.cuda()), outputs.cuda(), ndt,outputs_alpha.cuda()) -correlation_loss(outputs.cuda(),torch.abs(target.cuda()))
                loss = my_loss(target_.cuda(), outputs.cuda(), ndt, outputs_alpha.cuda()) - correlation_loss(
                    outputs.cuda(), (target_.cuda()))

                # Backward and optimize
                optimizer_drift.zero_grad()
                optimizer_alpha.zero_grad()
                # optimizer.zero_grad()
                loss.backward()
                # max_grads, ave_grads, gradss, layers = plot_grad_flow(model.named_parameters())
                # print('average grads',ave_grads)
                optimizer_drift.step()
                optimizer_alpha.step()
                # optimizer.step()

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

    ### save the model
    p = model.state_dict()
    torch.save(p, modelpath + '/' + 'mymodel_%s' % finalsubIDs[s] + postname + '.pth')




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
    y_att_test_list =[]
    for i, (test_data, test_target) in enumerate(test_loader):
        cond_target = y_test_cond[i*batch_size+test_target.shape[0]-test_target.shape[0]:i*batch_size+test_target.shape[0]]

        #     # test_data, test_target = next(iter(test_loader))
        with torch.no_grad():
            pred, pred_1,y_att_test = model(test_data.cuda())
        pred_1_copy = pred_1.detach().cpu()
        pred_copy = pred.detach().cpu()
        y_att_test_list.append(y_att_test)

        test_target = torch.squeeze((test_target))
        if cond_target.shape[0] == 1:
            test_target = test_target.view(1, 1)
        else:
            test_target = test_target.view(test_target.shape[0], 1)

        # test_loss = my_loss(test_target.cuda(), pred_copy.cuda(),ndt, alpha,alpha/2, err = 1e-29)
        # test_loss = my_loss(test_target, pred_copy.cuda(), ndt, torch.mean([2pred_1, axis=0).cuda())
        ndt= torch.tensor(ndt)
        test_loss = my_loss(test_target, pred_copy.cuda(), ndt, pred_1.cuda()) -correlation_loss(pred_copy.cuda(), torch.abs((test_target)).cuda())

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


    target_test = np.array(targetlist)
    drift_test = np.array(predictedlist)
    alpha_test = np.array(predictedlist_alpha)

    # test performance
    corr_drift_test = scipy.stats.pearsonr(np.abs(1/target_test), drift_test)
    corr_drift_test_rho = scipy.stats.spearmanr(np.abs(1/target_test), drift_test)

    corr_alpha_test = scipy.stats.pearsonr(np.abs(target_test), alpha_test)
    corr_alpha_test_rho = scipy.stats.spearmanr(np.abs(target_test), alpha_test)

    ##########################################################################
    ########################## get training data #############################
    ##########################################################################


    result = []
    out = []
    train_target= []
    out_alpha = []
    y_att_train_list =[]
    for i, (data, target) in enumerate(train_loader):
        print('train batch ', i)
        # origin shape: [4, 3, 32, 32] = 4, 3, 1024
        # input_layer: 3 input channels, 6 output channels, 5 kernel size
        target = torch.squeeze((target))
        # target = target.long()

        try:
            target = target.view(target.shape[0], 1)
        except IndexError:
            test_target = test_target.view(test_target.shape[0], 1)

        batch_n = data.shape[0]
        # device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
        # Forward pass
        with torch.no_grad():
            outputs, outputs_1,y_att_train = model(data)
        y_att_train_list.append(y_att_train[0:2])
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
    drift_train = np.asarray(out)
    target_train = np.asarray(train_target)
    alpha_train = np.asarray(out_alpha)

    # train performance
    corr_drift_train = scipy.stats.pearsonr(np.abs(1/target_train), drift_train)
    corr_drift_train_rho = scipy.stats.spearmanr(np.abs(1/target_train), drift_train)

    corr_alpha_train  = scipy.stats.pearsonr(np.abs(target_train), alpha_train)
    corr_alpha_train_rho  = scipy.stats.spearmanr(np.abs(target_train), alpha_train)

    # print out performance
    print('train drift: ', corr_drift_train)
    print('train drift_rho: ', corr_drift_train_rho)

    print('train alpha: ', corr_alpha_train)
    print('train alpha_rho: ', corr_alpha_train_rho)


    print('test drift: ', corr_drift_test)
    print('test drift_rho: ', corr_drift_test_rho)

    print('test alpha: ', corr_alpha_test)
    print('test alpha_rho: ', corr_alpha_test_rho)













    # making plots
    fig = plt.figure(figsize=(18, 9))
    gs = GridSpec(2, 4, figure=fig)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])
    ax4 = fig.add_subplot(gs[0, 2:])
    ax5 = fig.add_subplot(gs[1, 2:])

    set_alpha = 0.5
    ax0.scatter(1/target_train[target_train>0], drift_train[target_train>0], marker = 'o', color = 'blue',alpha=set_alpha)
    ax0.scatter(-1/target_train[target_train<0], drift_train[target_train<0], marker = 'o',color = 'red',alpha=set_alpha)
    ax1.scatter(np.abs(target_train[target_train>0]), alpha_train[target_train>0],color = 'blue',alpha=set_alpha)
    ax1.scatter(np.abs(target_train[target_train<0]), alpha_train[target_train<0], marker = 'o',color = 'red',alpha=set_alpha)

    ax2.scatter(1/target_test[target_test>0], drift_test[target_test>0], marker = 'o', color = 'blue',alpha=set_alpha)
    ax2.scatter(-1/target_test[target_test<0], drift_test[target_test<0], marker = 'o',color = 'red',alpha=set_alpha)
    ax3.scatter(np.abs(target_test[target_test>0]), alpha_test[target_test>0],color = 'blue',alpha=set_alpha)
    ax3.scatter(np.abs(target_test[target_test<0]), alpha_test[target_test<0], marker = 'o',color = 'red',alpha=set_alpha)


    ax0.set_xlabel('1/RT')
    ax0.set_ylabel('trained Drift')
    ax1.set_xlabel('RT')
    ax1.set_ylabel('trained Alpha')
    ax2.set_xlabel('1/RT')
    ax2.set_ylabel('predicted Drift')
    ax3.set_xlabel('RT')
    ax3.set_ylabel('predicted Alpha')


    ax0.set_title('Corr_'r'$\rho = %.2f$' % corr_drift_train_rho[0])
    ax1.set_title('Corr_'r'$\rho = %.2f$' % corr_alpha_train_rho[0])
    ax2.set_title('Corr_'r'$\rho = %.2f$' % corr_drift_test_rho[0])
    ax3.set_title('Corr_'r'$\rho = %.2f$' % corr_alpha_test_rho[0])

    fig.tight_layout()
