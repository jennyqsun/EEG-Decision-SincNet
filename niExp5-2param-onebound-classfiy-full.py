# Created on 11/9/21 at 11:02 AM

# Author: Jenny Sun


import os
import scipy.stats
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.utils
import torchvision
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import Dataset
from torchvision import datasets, transforms
import sys
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from matplotlib import interactive
interactive(True)
from models_full import *
import numpy as np
import os
import hdf5storage
from sklearn.model_selection import train_test_split
from pytorchtools import EarlyStopping, RMSLELoss
from torch.utils.tensorboard import SummaryWriter
from sklearn.linear_model import LinearRegression
import random
from sklearn.metrics import r2_score
import pickle
import matplotlib
from matplotlib.gridspec import GridSpec
from scipy.io import savemat
from bipolar import hotcold
import shutil
from zscore_training import *
import os
from configparser import ConfigParser
from topo import *
from my_wfpt import *
import matplotlib.ticker as ticker
import matplotlib.patches as patches
from mpl_toolkits.axes_grid.inset_locator import inset_axes

# set up cuda

torch.cuda.device_count()
gpu0  = torch.device(0)
gpu1 = torch.device(1)
torch.cuda.set_device(gpu1)
device = torch.device(gpu1)
print(gpu0,gpu1)

import time
t1 = time.time()
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
timestart = 0   # 0 means starts from stimlus
timeend = 500
trialdur = timeend * 2 - timestart * 2


correctModel = False  # whether the signed rt is coded as correct and incorrect
choiceModel = True   # whether target is choice 1 and choice 2
notrainMode = True     # if true, just load the model
                        # if false, train model
if notrainMode:
    keepTrainMode = False
    createConfig = False
else:
    createConfig = True    # when training, create config files.
    keepTrainMode = False  # set this to be True if wants to keep training from previous model
    zScoreData = False



datapath = '/home/jenny/sincnet_eeg/ni_data/exp5data/'
sr = 500
# timeend = 800 # when 300ms after stim

# Hyper-parameters
num_epochs = 300
batch_size = 64
learning_rate = 0.001
num_chan = 98
dropout_rate = 0.7
compute_likelihood = True
EarlyStopPatience = 8

######################## tensorbaord initilization ###########################

model_0 = SincDriftBoundAttChoice_full(dropout=dropout_rate).to(device)
model = SincDriftBoundAttChoice_full(dropout=dropout_rate).to(device)

model_0 = torch.nn.DataParallel(model_0, device_ids = [1])
model = torch.nn.DataParallel(model, device_ids = [1])

######################## creating directory and file nmae ############for s########
# postname = '_prestim500_1000_0123_ddm_2param'
# postname = '_ni_2param_onebound_classify_full_cfg' # clamp at forward
postname = '_ni_2param_onebound_classify_full'
# postname = '_ni_2param_onebound_choice_model0'
# postname = '_ni_2param_onebound_choice_model0'


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
def viz_histograms(model, epoch):
    for name, weight in model.named_parameters():
        try:
            tb.add_histogram(name, weight, epoch)
            tb.add_histogram(f'{name}.grad', weight.grad, epoch)
        except NotImplementedError:
            continue


def getIDs(path):
    allDataFiles = os.listdir(path)
    finalsub = [i[:-4] for i in allDataFiles if "high" not in i]
    finalsub.sort()
    return np.unique(finalsub)

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
    '''
    :return:
    '''
    chans = np.arange(0, 128)
    chans_del = np.array(
        [56, 63, 68, 73, 81, 88, 94, 100, 108, 114, 49, 43, 48, 38, 32, 44, 128, 127, 119, 125, 120, 121, 126,
         113, 117, 1, 8, 14, 21, 25]) - 1
    chans = np.delete(chans, chans_del)
    return chans

def loadsubjdict(path, subID, filetype = '.mat'):
    path = '/home/jenny/sincnet_eeg/ni_data/exp5data/'
    if filetype == '.mat':
        datadict = hdf5storage.loadmat(path + subID+ '_high' + '.mat')
    else:
        datadict = pickle.load(open(path + subID  +'.pkl','rb'))
    return datadict


def getrtdata(datadict, Tstart=250, Tend=1250):
    data = np.array(datadict['eegdata'])
    data = data[::2, :, :]
    sr = np.array(datadict['sr']) / 2
    condition = np.array(datadict['snr'])
    # goodtrials = np.array(datadict['trials'])[0]
    correct = np.array(datadict['acc'])
    rt = np.array(datadict['rt'])

    rtInclude = np.abs(rt) >=300
    # goodchan = goodchans()
    # goodchan = chanmotor()
    goodchan = chansets_new()
    # data = data[:, :, correct==1]
    # condition = condition[correct==1]
    data = data[:, goodchan, :]
    spfs =datadict['spfs']
    highlow =datadict['highlow']
    choice =np.zeros_like(correct)
    choice[(highlow == 2) &( correct ==1)] = 1 # chose high
    choice[(highlow==1) & (correct==0)] = 1 # choce high
    
    
    return data[Tstart:Tend, :, rtInclude], condition[rtInclude], rt[rtInclude], correct[rtInclude], choice[rtInclude]


def reshapedata(data):
    timestep, nchan, ntrial = data.shape
    newdata = np.zeros((ntrial, nchan, timestep))
    for i in range(0, ntrial):
        newdata[i, :, :] = data[:, :, i].T
    return newdata

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

    err = torch.tensor(0.01).cuda()
    tt = torch.max(torch.tensor(torch.abs(t.cuda()) - t0.cuda()),err) / torch.max(err,a.cuda()) ** 2  # normalized time
    tt_vec = torch.tile(tt, (1, 10))
    pp = torch.cumsum((w+2*k)*torch.exp(-(((w+2*k)**2)/2)/tt_vec),axis=1)
    pp = pp[:,-1]/torch.sqrt(2*torch.tensor(np.pi)*torch.squeeze(tt)**3)
    pp = pp[:, None]
    # v = torch.where(torch.tensor(t).cuda()>0, v, -v)   # if time is negative, flip the sign of v
    v = torch.clamp(v, -4,4)
    # a = torch.clamp(a, 0.5, 4)
    # t = torch.where(torch.tensor(t).cuda() > 0, torch.tensor(t).cuda(), torch.tensor(-t).cuda())
    p = (pp * (torch.exp(-v*torch.max(err, a)*w - (v**2)*torch.tensor(t).cuda()/2) /(torch.max(err,a)**2)))
    # p = torch.where(torch.tensor(v).cuda()>0, 1*p, 6*p)
    p = torch.log(p)
    # p = torch.where(torch.tensor(v).cuda()>0, p, -p)
    # print(t,a,v)
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




class weightConstraint(object):
    def __init__(self, model):
        self.cutoff = model.module.sinc_cnn2d_drift.cutoff
        self.min_freq = 1
        self.min_band = 2
        self.b1max = int(self.cutoff - self.min_freq -  self.min_band)
    def __call__(self, module):
        if hasattr(module, 'filt_b1'):
            b1 = module.filt_b1.data
            band = module.filt_band.data
            fs = module.freq_scale
            b1 = b1.clamp(-1 * (torch.min(torch.abs(b1)+(self.min_freq /fs)+torch.abs(band),
                                                            torch.ones_like(band) * self.b1max/fs)), torch.min(torch.abs(b1)+(self.min_freq /fs)+torch.abs(band),
                                                            torch.ones_like(band) * self.b1max/fs))
            module.filt_b1.data = b1


################################### CREATE CONFIG FILES ###################
if createConfig:
    config_object = ConfigParser()
    config_object["data"] = {
        "dataset": datapath,
        "filters": [1,45],
        "sr": sr,
        "zscore": zScoreData
    }

    config_object["hyperparameters"] = {
        "N_filters":model.module.num_filters,
        "filter_length": model.module.filter_length,
        "pool_window_ms": model.module.pool_window_ms,
        "stride_window_ms": model.module.stride_window_ms,
        "attentionLatent": model.module.attentionLatent,
        "N_chan":model.module.num_chan,
        "patience":EarlyStopPatience
    }

    config_object["optimization"] = {
        "batch_size": batch_size,
        "maxepoch": num_epochs,
        "seed": seednum,
        "weights_constrain": model.module.sinc_cnn2d_drift.cutoff
    }
    #Write the above sections to config.ini file
    with open(modelpath + '/config.ini', 'w') as conf:
        config_object.write(conf)
else:
    #Write the above sections to config.ini file
    config_object = ConfigParser()
    config_object.read(modelpath + "/config.ini")
    zScoreData = config_object["data"]["zscore"] == 'True'
    datapath = config_object["data"]["dataset"]
    EarlyStopPatience = int(config_object["hyperparameters"]["patience"])
    seednum = int(config_object["optimization"]["seed"])


# %%
############################################################################
################################# starts here ###############################
############################################################################
results = dict()  # a results dictionary for storing all the data
finalsubIDs = getIDs(datapath)
mylist = np.arange(0, len(finalsubIDs))
############################################
############### set subject ######################
############################################
for s in range(1,2):
    # a results dictionary for storing all the data
    finalsubIDs = getIDs(datapath)
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

    tbdir = 'runs/' + postname + '/' + finalsubIDs[s]
    if notrainMode is False:
        try:
            len(os.listdir(tbdir)) != 0
            shutil.rmtree(tbdir)
        except:
            pass
        tb = SummaryWriter(tbdir)
    ####################### define sub #########################################
    datadict = loadsubjdict(datapath, finalsubIDs[subjectstart], '.mat')
    # datadict1 = loadsubjdict(finalsubIDs[subjectstart], '.pkl')

    print(str(subjectstart) + '/' + 'subjectID: ' + finalsubIDs[subjectstart])
    data, cond, rt,correct,choice = getrtdata(datadict, timestart, timeend)

    rtall = rt.copy()
    correct = correct.astype('int')
    if correctModel is True:
        rt_acc = (correct * 2 - 1) * np.abs(rt)
        correct = correct.astype('int')
    if choiceModel is True:
        rt = (choice * 2 - 1) * rt
    # correctind = condition>0
    #
    newdata = reshapedata(data).astype('float32')
    rt = rt*0.001


    indices = np.arange(0,len(rt))
    X_train0, X_test, y_train0, y_test, indx_train, indx_test = train_test_split(newdata, rt, indices, test_size=0.2, random_state=42)


    ndt = np.min(np.abs(y_train0)) * 0.93
    ndt = torch.tensor(ndt)
    print('ndt: ', ndt)

    _, _, y_train0_cond, y_test_cond = train_test_split(newdata, cond, test_size=0.2, random_state=42)


    # ztransform
    if zScoreData:
        X_train0, X_train0Original, zmodel = zscore(X_train0.copy())
        X_test, X_testOriginal = ztransform(X_test.copy(), zmodel)


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



    # %% optimization
    g = torch.Generator()
    g.manual_seed(seednum)

    # model.apply(initialize_weights)

    # criterion = nn.MSELoss()
    # criterion = nn.L1Loss()
    # print('criterion: ', criterion)

    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)


    n_total_steps = len(train_loader)
    # criterion = nn.CrossEntropyLoss().cuda()

    # define optimization ###############
    criterion = nn.BCELoss()
    # criterion = nn.BCEWithLogitsLoss()
    # criterion = nn.MSELoss()

    weight_decay = 1e-2

    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # optimize different parameters
    alpha_param = []
    for n, p in model.named_parameters():
        if "bound" in n:
            alpha_param.append(p)

    drift_param = []
    for n, p in model.named_parameters():
        if "bound" not in n and "choice" not in n:
            # print(n)
            drift_param.append(p)

    choice_param = []
    for n, p in model.named_parameters():
        if "choice" in n:
            choice_param.append(p)

    optimizer_drift = torch.optim.Adam(drift_param, lr=learning_rate)
    optimizer_alpha = torch.optim.Adam(alpha_param, lr=learning_rate,weight_decay= weight_decay)
    optimizer_choice = torch.optim.Adam(choice_param, lr=learning_rate)

    early_stopping = EarlyStopping(patience=EarlyStopPatience, verbose=True)

    ###########################################################################################
    ########################## load trained model ##############################################
    if notrainMode is True:
        print('no train mode on')
        model.load_state_dict(torch.load('/home/jenny/sincnet_eeg/' + modelpath +
                                         '/mymodel_%s' % finalsubIDs[s] + postname + '.pth'))
        model.eval()
        p = model.state_dict()
    else:
        if keepTrainMode is True:
            print('keep training mode on')
            model.load_state_dict(torch.load('/home/jenny/sincnet_eeg/' + modelpath +
                                         '/mymodel_%s' % finalsubIDs[s] + postname + '.pth'))
        else:
            print('start training from scratch')
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
                data = data.cuda()
                outputs, outputs_alpha, choice, = model(data)

                target_ = torch.abs(target)
                choice_target = (target > 0).long().to(torch.float32).cuda()
                # choice_target = choice_target*2 -1
                if i%10==0:
                    print('acc: ',sum(torch.round(choice.detach().cpu())==choice_target.detach().cpu()) / len(choice))
                # print('d',outputs,'a',outputs_alpha)
                # print('target', target)
                # print(outputs_alpha)
                # loss = criterion(outputs, target.cuda())
                # loss = my_loss(target,outputs,ndt, torch.mean(outputs_alpha),torch.mean(outputs_alpha)/2,1e-29)
                # loss = my_loss(target.cuda(), outputs.cuda(), ndt, torch.mean(outputs_alpha,axis=0).cuda())
                # loss = my_loss(torch.abs(target.cuda()), outputs.cuda(), ndt,outputs_alpha.cuda()) -correlation_loss(outputs.cuda(),torch.abs(target.cuda()))
                loss = my_loss(target_.cuda(), outputs.cuda(), ndt, outputs_alpha.cuda()) - correlation_loss(
                    outputs.cuda(), (target_.cuda())) +  1*criterion(torch.squeeze(choice), torch.squeeze(choice_target))\
                              + 0.5*outputs_alpha.cuda().sum()

                # Backward and optimize
                optimizer_drift.zero_grad()
                optimizer_alpha.zero_grad()
                optimizer_choice.zero_grad()
                # optimizer.zero_grad()
                loss.backward()
                # max_grads, ave_grads, gradss, layers = plot_grad_flow(model.named_parameters())

                optimizer_drift.step()
                optimizer_alpha.step()
                optimizer_choice.step()
                # optimizer.step()
                clipper = weightConstraint(model=model)
                model.module.sinc_cnn2d_drift.apply(clipper)
                model.module.sinc_cnn2d_choice.apply(clipper)
                model.module.sinc_cnn2d_bound.apply(clipper)
                epoch_loss.append(loss.detach().cpu().numpy())

                print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{n_total_steps}], Loss: {loss.item():.4f}')
            tb.add_scalar('training loss', np.mean(epoch_loss), epoch)
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
    n_total_steps = len(test_loader)
    trial_grad_list = []
    y_att_test_list =[]
    y_att_test_list_choice =[]
    choice_targetlist_test = []
    choice_predict_test=[]
    for i, (test_data, test_target) in enumerate(test_loader):
        cond_target = y_test_cond[i*batch_size+test_target.shape[0]-test_target.shape[0]:i*batch_size+test_target.shape[0]]
        choice_test_target = np.squeeze((test_target > 0).detach().numpy()).astype('int')
        # choice_test_target = choice_test_target *2 -1
        choice_targetlist_test.extend(choice_test_target.tolist())
        #     # test_data, test_target = next(iter(test_loader))
        with torch.no_grad():
            pred, pred_1,choice_test= model(test_data.cuda())
        choice_predict_test.extend(np.squeeze((choice_test).detach().cpu().round().numpy()).tolist())
        pred_1_copy = pred_1.detach().cpu()
        pred_copy = pred.detach().cpu()
        # y_att_test_list.append(y_att_test[0:3])
        # y_att_test_list_choice.append(y_att_test_choice[0:3])
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

    train_loader = DataLoader(dataset=train_set,
                              batch_size=batch_size,
                              shuffle=False,  # shuffle the data
                              num_workers=0, worker_init_fn=seed_worker,
                              generator=g)

    result = []
    out = []
    train_target= []
    out_alpha = []
    y_att_train_list =[]
    y_att_train_list_choice = []
    choice_targetlist_train = []
    choice_predict_train=[]

    for i, (data, target) in enumerate(train_loader):
        print('train batch ', i)
        # origin shape: [4, 3, 32, 32] = 4, 3, 1024
        # input_layer: 3 input channels, 6 output channels, 5 kernel size
        target = torch.squeeze((target))
        # target = target.long()
        choice_train_target = np.squeeze((target > 0).detach().numpy()).astype('int')
        # choice_train_target = choice_train_target *2 -1
        choice_targetlist_train.extend(choice_train_target.tolist())
        try:
            target = target.view(target.shape[0], 1)
        except IndexError:
            test_target = test_target.view(test_target.shape[0], 1)

        batch_n = data.shape[0]
        # device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
        # Forward pass
        with torch.no_grad():
            outputs, outputs_1,choice_train = model(data.cuda())
        choice_predict_train.extend(np.squeeze(choice_train.detach().cpu().round().numpy()).tolist())

        # y_att_train_list.append(y_att_train[0:2])
        # y_att_train_list_choice.append(y_att_train_choice[0:2])

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

    acc_train = sum(np.array(choice_targetlist_train)== np.array(choice_predict_train)) / len(choice_targetlist_train)
    acc_test = sum(np.array(choice_targetlist_test)== np.array(choice_predict_test)) / len(choice_targetlist_test)








    # making plots
    plt.rcParams.update({'font.size': 20})
    fig = plt.figure(figsize=(18, 9))
    gs = GridSpec(2, 3, figure=fig)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])
    ax4 = fig.add_subplot(gs[0, 2])
    ax5 = fig.add_subplot(gs[1, 2])

    set_alpha = 0.6
    ax0.scatter(1/target_train[target_train>0], drift_train[target_train>0], marker = 'o', color = 'tab:purple',alpha=set_alpha,  label = 'Low Frequency')
    ax0.scatter(-1/target_train[target_train<0], drift_train[target_train<0], marker = '*',color = 'tab:red',alpha=set_alpha-0.1, label = 'High Frequency')
    ax1.scatter(np.abs(target_train[target_train>0]), alpha_train[target_train>0],color = 'tab:purple',alpha=set_alpha)
    ax1.scatter(np.abs(target_train[target_train<0]), alpha_train[target_train<0], marker = '*',color = 'tab:red',alpha=set_alpha-0.1)

    ax2.scatter(1/target_test[target_test>0], drift_test[target_test>0], marker = 'o', color = 'tab:purple',alpha=set_alpha)
    ax2.scatter(-1/target_test[target_test<0], drift_test[target_test<0], marker = '*',color = 'tab:red',alpha=set_alpha-0.1)
    ax3.scatter(np.abs(target_test[target_test>0]), alpha_test[target_test>0],color = 'tab:purple',alpha=set_alpha)
    ax3.scatter(np.abs(target_test[target_test<0]), alpha_test[target_test<0], marker = '*',color = 'tab:red',alpha=set_alpha-0.1)


    ax0.set_xlabel('1/RT')
    ax0.set_ylabel('Trained Drift')
    ax1.set_xlabel('RT')
    ax1.set_ylabel('Trained Boundary')
    ax2.set_xlabel('1/RT')
    ax2.set_ylabel('Predicted Drift')
    ax3.set_xlabel('RT')
    ax3.set_ylabel('Predicted Boundary')
    ax0.legend()

    ax0.set_title('Spearman 'r'$\rho = %.2f$' % corr_drift_train_rho[0])
    ax1.set_title('Spearman 'r'$\rho = %.2f$' % corr_alpha_train_rho[0])
    ax2.set_title('Spearman 'r'$\rho = %.2f$' % corr_drift_test_rho[0])
    ax3.set_title('Spearman 'r'$\rho = %.2f$' % corr_alpha_test_rho[0])

    # confusion matrix
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    import seaborn as sns
    cm_train = confusion_matrix(np.array(choice_targetlist_train), np.array(choice_predict_train),normalize='true')
    disp_train = ConfusionMatrixDisplay(confusion_matrix=cm_train,display_labels =('Low \nFrequency', 'High \nFrequency'))
    disp_train.plot(ax=ax4,cmap = 'Greens')
    ax4.set_title('Accuracy: %.2f'%acc_train)


    cm_test= confusion_matrix(np.array(choice_targetlist_test), np.array(choice_predict_test),normalize='true')
    disp_test = ConfusionMatrixDisplay(confusion_matrix=cm_test,display_labels =['Low    \nFrequency', 'High    \nFrequency'], )
    disp_test.plot(ax=ax5, cmap = 'Greens')
    ax5.set_title('Accuracy: %.2f'%acc_test)
    ymax = max(max(ax0.get_ylim()), max(ax2.get_ylim()))
    ymin = min(min(ax0.get_ylim()), min(ax2.get_ylim()))
    xmax = max(max(ax0.get_xlim()), max(ax2.get_xlim()))
    xmin  = min(min(ax0.get_xlim()), min(ax2.get_xlim()))
    ax0.set_xlim(xmin,xmax)
    ax0.set_ylim(ymin,ymax)
    ax2.set_xlim(xmin,xmax)
    ax2.set_ylim(ymin,ymax)

    ymax1 = max(max(ax1.get_ylim()), max(ax3.get_ylim()))
    ymin1 = min(min(ax1.get_ylim()), min(ax3.get_ylim()))
    xmax1 = max(max(ax1.get_xlim()), max(ax3.get_xlim()))
    xmin1 = min(min(ax1.get_xlim()), min(ax3.get_xlim()))
    ax1.set_xlim(xmin1, xmax1)
    ax1.set_ylim(ymin1, ymax1)
    ax3.set_xlim(xmin1, xmax1)
    ax3.set_ylim(ymin1, ymax1)
    ax4.tick_params(axis='both', which='major', labelsize=18)
    ax5.tick_params(axis='both', which='major', labelsize=18)
    # fit a line


    m, b = np.polyfit(np.abs(1 / np.abs(target_train)), drift_train, 1)
    ax0.plot(np.abs(1 / target_train), m * np.abs(1 / target_train) + b, color='black')
    ax0.plot(np.linspace(ax0.get_xlim()[0], ax0.get_xlim()[1]),
             m * np.linspace(ax0.get_xlim()[0], ax0.get_xlim()[1]) + b, color='grey')

    m, b = np.polyfit(np.abs(np.abs(target_train)), alpha_train, 1)
    ax1.plot(np.abs(train_target), m * np.abs(train_target) + b, color='black')
    ax1.plot(np.linspace(ax1.get_xlim()[0], ax1.get_xlim()[1]),
             m * np.linspace(ax1.get_xlim()[0], ax1.get_xlim()[1]) + b, color='grey')

    m, b = np.polyfit(np.abs(1 / np.abs(target_test)), drift_test, 1)
    ax2.plot(np.abs(1 / target_test), m * np.abs(1 / target_test) + b, color='black')
    ax2.plot(np.linspace(ax2.get_xlim()[0], ax2.get_xlim()[1]),
             m * np.linspace(ax2.get_xlim()[0], ax2.get_xlim()[1]) + b, color='grey')

    m, b = np.polyfit(np.abs(np.abs(target_test)), alpha_test, 1)
    ax3.plot(np.abs(target_test), m * np.abs(target_test) + b, color='black')
    ax3.plot(np.linspace(ax3.get_xlim()[0], ax3.get_xlim()[1]),
             m * np.linspace(ax3.get_xlim()[0], ax3.get_xlim()[1]) + b, color='grey')

    fig.tight_layout()
    fig.show()
    fig.savefig(figurepath + '/'+ finalsubIDs[s] + 'performance' + '.png')



    #%% Attention
 # %%
    ############### plotting the conditions ############################

    # get the index of conditions, acc
    cond_train, cond_test = datadict['snr'][indx_train], datadict['snr'][indx_test]
    correct_train, correct_test = datadict['acc'][indx_train], datadict['acc'][indx_test]


    # making plots
    plt.rcParams.update({'font.size': 20})
    fig2 = plt.figure(figsize=(18, 9))
    gs = GridSpec(2, 3, figure=fig)
    ax0 = fig2.add_subplot(gs[0, 0])
    ax1 = fig2.add_subplot(gs[0, 1])
    ax2 = fig2.add_subplot(gs[0, 2])
    ax3 = fig2.add_subplot(gs[1, 0])
    ax4 = fig2.add_subplot(gs[1, 1])
    ax5 = fig2.add_subplot(gs[1, 2])


    import seaborn as sns
    x = ['High','Med','Low']   # low snr means hard, med snr means median, high snr means hard
    rt_train_hard = np.abs(target_train[cond_train == 0.5])
    rt_train_med = np.abs(target_train[cond_train == 1])
    rt_train_easy = np.abs(target_train[cond_train == 2])

    drift_train_hard = drift_train[cond_train == 0.5]
    drift_train_med = drift_train[cond_train == 1]
    drift_train_easy =drift_train[cond_train == 2]

    alpha_train_hard = alpha_train[cond_train == 0.5]
    alpha_train_med = alpha_train[cond_train == 1]
    alpha_train_easy = alpha_train[cond_train == 2]


    rt_test_hard = np.abs(target_test[cond_test == 0.5])
    rt_test_med = np.abs(target_test[cond_test == 1])
    rt_test_easy = np.abs(target_test[cond_test == 2])

    alpha_test_hard = alpha_test[cond_test == 0.5]
    alpha_test_med = alpha_test[cond_test == 1]
    alpha_test_easy = alpha_test[cond_test == 2]


    drift_test_hard = drift_test[cond_test == 0.5]
    drift_test_med = drift_test[cond_test == 1]
    drift_test_easy = drift_test[cond_test == 2]
    sns.set_palette("Set2")
    sns.boxenplot(data = (rt_train_easy, rt_train_med,rt_train_hard), ax = ax0)
    sns.boxenplot(data = (drift_train_easy, drift_train_med,drift_train_hard), ax = ax1)
    sns.boxenplot(data = (alpha_train_easy, alpha_train_med,alpha_train_hard), ax = ax2)
    #
    sns.boxenplot(data=(rt_test_easy, rt_test_med, rt_test_hard), ax=ax3)
    sns.boxenplot(data=(drift_test_easy, drift_test_med, drift_test_hard), ax=ax4)
    sns.boxenplot(data=(alpha_test_easy, alpha_test_med, alpha_test_hard), ax=ax5)

    # ax0.violinplot([rt_train_easy, rt_train_med,rt_train_hard], positions = (0,1,2), color = ('red','blue','black'))
    ax0.set_ylabel('RT (training) ')
    ax1.set_ylabel('Drift (training)')
    ax2.set_ylabel('Boundary (training)')
    ax3.set_ylabel('RT (test) ')
    ax4.set_ylabel('Drift (test)')
    ax5.set_ylabel('Boundary (test)')

    # manually set up legend
    import matplotlib.patches as mpatches

    for a in (ax0,ax1,ax2,ax3,ax4,ax5):
        a.set_xticklabels(x)
    colors = [r for r in ax1.get_children() if hasattr(r, 'get_facecolors')]
    fig2.show()
    mycolor = []
    for c in colors:
        c_ = c.get_facecolors()
        if len(c_) != 1:
            print(c_[-1])
            mycolor.append(c_[-1])


    patch0 = mpatches.Patch(color=mycolor[0], label='Low SNR')
    patch1 = mpatches.Patch(color=mycolor[1], label='Med SNR')
    patch2 = mpatches.Patch(color=mycolor[2], label='High SNR')
    fig2.legend(handles=[patch0,patch1,patch2])
    fig2.tight_layout()
    fig2.subplots_adjust(right=0.85)
    fig2.show()
    fig2.savefig(figurepath + '/' + finalsubIDs[s] + '_condition' + '.png')

#%%
#
# def LikelihoodRatio(l1,l2):
#     '''sum of negative log likelihood L (parameters | data)'''
#     if np.sign(l1) == np.sign(l2):
#         LR = l1/l2
#         if LR < 1:
#
#     return LR,
# calculate likelihood
    from my_wfpt import *
    #  L (delta_tr, alpha_tr | RT_tr, ndt)
    L_train = wfpt_vec(np.abs(train_target), -np.array(drift_train), ndt.numpy(), np.array(alpha_train))
    L_check = my_loss(torch.tensor(np.abs(train_target)).reshape(-1,1).cuda(), -torch.tensor(drift_train).reshape(-1,1).cuda(), \
                      ndt.cuda(), torch.tensor(alpha_train).reshape(-1,1).cuda()).cpu().numpy()

    #  train mean on tran rt
    train_drift_mean = np.zeros_like(train_target) + np.median(drift_train)
    train_boundary_mean = np.zeros_like(train_target) + np.median(alpha_train)
    L_tr_mean_on_tr = wfpt_vec(np.abs(train_target), -1* train_drift_mean, ndt.numpy(), np.array(train_boundary_mean))


    #  train mean on test rt
    train_drift_mean = np.zeros_like(target_test) + np.median(drift_train)
    train_boundary_mean = np.zeros_like(target_test) + np.median(drift_train)
    L_tr_on_ts = wfpt_vec(np.abs(target_test), -1* train_drift_mean, ndt.numpy(), np.array(train_boundary_mean))

    # test on test rt
    L_ts_on_ts = wfpt_vec(np.abs(target_test), -1* drift_test, ndt.numpy(), np.array(alpha_test))

    print('L(train_on_trainRT): ', L_train, L_check)  # check if they are the same
    print('L(train_median_on_trainRT): ', L_tr_mean_on_tr)  # check if they are the same
    print('L(train_median_testRT): ', L_tr_on_ts)  # check if they are the same
    print('L(test_on_testRT): ', L_ts_on_ts)  # check if they are the same

    LRT_train = L_tr_mean_on_tr/ L_train
    print('Ltrain:', LRT_train)
    if LRT_train > 1:
        print('L_train_on_trainRT is a better model!!!')
    else:
        print('L_train_median_on_trainRT is better.')

    LRT_test = L_tr_on_ts / L_ts_on_ts
    print('Ltest:', LRT_test)
    if LRT_test > 1:
        print('L_test_on_testRT is a better model!!!')
    else:
        print('L_train_median_on_testRT is better.')
    #%%
    #############3  DDM parameters #####################
    p = model.state_dict()


    # visualize the bands
    p_low = p['module.sinc_cnn2d_drift.filt_b1']
    p_band = p['module.sinc_cnn2d_drift.filt_band']

    filt_beg_freq = (torch.abs(p_low) + 1 / 500)
    filt_end_freq = torch.clamp(filt_beg_freq + torch.abs(p_band) + 2 / 500, 3/500, 50/500)

    filt_beg_freq = filt_beg_freq.cpu().numpy() * 500
    filt_end_freq = filt_end_freq.cpu().numpy() * 500
    goodchan = chansets_new()

    #
    # results[finalsubIDs[s]] = dict()
    # results[finalsubIDs[s]] = {'filt_beg_freq': filt_beg_freq, 'filt_end_freq': filt_end_freq,
    #                             # 'corr': corr1, 'corr_rho': corr_rho1, 'filter_grads': G, 'temporal_grads': Gt,
    #                             'chan_weights': torch.squeeze(p['module.separable_conv.depthwise.weight']).cpu()}
    #
    # sub_dict= dict()
    # sub_dict = {'filt_beg_freq': filt_beg_freq, 'filt_end_freq': filt_end_freq,
    #
    #                             'chan_weights': torch.squeeze(p['module.separable_conv.depthwise.weight']).cpu().numpy(),
    #             'target_rt_test': targetlist, 'delta_test': predictedlist, 'alpha_test': predictedlist_alpha,
    #             'target_rt_train': np.array(train_target),  'delta_train': np.array(out) , 'alpha_train':np.array(out_alpha)
    #             }
    # savemat(resultpath + '/%s' % finalsubIDs[s][0:-1]+ '_results' + postname + '.mat', sub_dict)
    # my_file = open(resultpath + f'/%s' % finalsubIDs[s][0:-1]+ '_results' + postname +'.pkl', 'wb')
    # pickle.dump(sub_dict, my_file)
    # my_file.close()


    # plt.show()
    # filter analysis


    #
    # # let's look at the the attnetion mechanism
    # att_train0 = np.empty((0,32,98,370))
    # for i in y_att_train_list:
    #     att_train0 = np.vstack((att_train0, np.array(torch.squeeze(i[0].cpu().detach()))))
    #
    # weights_train = np.mean(att_train0,axis=0)


    # get the attention weights
    # split correct incorrect
    rt_acc = (correct * 2 - 1) * np.abs(rt)
    correct = correct.astype('int')
    _, _, _, y_testacc = train_test_split(newdata, rt_acc, test_size=0.2, random_state=42)

    attentionTr = np.empty((0,32))
    for i in y_att_train_list:
        attentionTr = np.vstack((attentionTr, np.array(torch.squeeze(i[1].cpu().detach()))))

    attentionTr_mean =  np.mean(attentionTr,axis=0)


    attentionTs = np.empty((0,32))
    for i in y_att_test_list:
        attentionTs = np.vstack((attentionTs, np.array(torch.squeeze(i[1].cpu().detach()))))

    attentionTs_mean =  np.mean(attentionTs,axis=0)
    attentionTs_mean_corr =  np.mean(attentionTs[y_testacc>0,:],axis=0)
    attentionTs_mean_incorr =  np.mean(attentionTs[y_testacc<0,:],axis=0)

    # choice
    attentionTr_choice = np.empty((0, 32))
    for i in y_att_train_list_choice:
        attentionTr_choice = np.vstack((attentionTr_choice, np.array(torch.squeeze(i[1].cpu().detach()))))

    attentionTr_mean_choice = np.mean(attentionTr_choice, axis=0)

    attentionTs_choice = np.empty((0, 32))
    for i in y_att_test_list_choice:
        attentionTs_choice = np.vstack((attentionTs_choice, np.array(torch.squeeze(i[1].cpu().detach()))))

    attentionTs_mean_choice= np.mean(attentionTs_choice, axis=0)



    attentionTs_mean_corr_choice = np.mean(attentionTs_choice[y_testacc > 0, :], axis=0)
    attentionTs_mean_incorr_choice = np.mean(attentionTs_choice[y_testacc < 0, :], axis=0)

    # # # let's pick the 10 most important weights to visualize
    # numFreq = 4
    # depth = int(model.separable_conv.depthwise.out_channels / model.separable_conv.depthwise.in_channels)
    # fig2,ax2 = plt.subplots(numFreq,depth+1,figsize= (10,12))
    # for i in range(numFreq):
    #     weight_ind = np.argsort(attentionTs_mean)[-(i+1)]
    #     bands = filt_beg_freq[weight_ind],filt_end_freq[weight_ind]
    #     sum_weights = []
    #     for d in range(depth):
    #         plottopomap(sub_dict['chan_weights'][weight_ind*depth+d],ax2[i][d])
    #         sum_weights.append(sub_dict['chan_weights'][weight_ind*depth+d])
    #         ax2[i][d].set_xlabel('%.0fHz'% bands[0] + '-%.0fHz'% bands[1])
    #     plottopomap(np.mean(np.array(sum_weights), axis=0), ax2[i][depth])
    #     ax2[i][depth].set_xlabel('Sum ')
    #
    # fig2.suptitle('Weights of Important Frequency bands')
    # fig2.tight_layout()
    #
    # fig2.savefig('figures_final/' +  finalsubIDs[s] + 'spatial_kernel_ts' +'.png')
    # fig2.show()
    # #
    # # #

    # #
    # weights_ = np.repeat(attentionTs_mean, depth)
    # weighted_channels = weights_ @ sub_dict['chan_weights']
    # fig4,ax4 = plt.subplots(1)
    # plottopomap(weighted_channels,ax4)
    # fig4.suptitle('Weighted Sum of all Frequency Bands')
    # fig4.savefig('figures_final/' +  finalsubIDs[s] + 'spatial_kernel_allFreq_ts' +'.png')
    #
    # fig4.show()
    # #
    #
    #
    #%%
    #############################################################
    ##       visulize band pass fitlers    ###########
    #########################################################


    def norm(vec):
        f_min, f_max = np.min(vec), np.max(vec)
        newV =  2* (vec - f_min) / (f_max - f_min) -1
        return newV
    def normZeroOne(vec):
        f_min, f_max = np.min(vec), np.max(vec)
        newV =  (vec - f_min) / (f_max - f_min)
        return newV
    fig5, ax5_ = plt.subplots(1, figsize = (5,5.5))
    ax5_.axis('off')
    ax5 = fig5.add_axes([0.2, 0.12, 0.7, 0.68])
    p = model_0.state_dict()
    p_low = p['module.sinc_cnn2d.filt_b1']
    p_band = p['module.sinc_cnn2d.filt_band']
    #
    filt_beg_freq0 = (torch.abs(p_low) + 1 / 500)
    filt_end_freq0 = torch.clamp(filt_beg_freq0 + torch.abs(p_band) + 2 / 500, 3/500, 50/500)

    filt_beg_freq0 = filt_beg_freq0.cpu().numpy() * 500
    filt_end_freq0 = filt_end_freq0.cpu().numpy() * 500

    p = model.state_dict()
    p_low = p['module.sinc_cnn2d.filt_b1']
    p_band = p['module.sinc_cnn2d.filt_band']
    #
    filt_beg_freq = (torch.abs(p_low) + 1 / 500)
    filt_end_freq = torch.clamp(filt_beg_freq + torch.abs(p_band) + 2 / 500, 3/500, 50/500)

    filt_beg_freq = filt_beg_freq.cpu().numpy() * 500
    filt_end_freq = filt_end_freq.cpu().numpy() * 500
    alpha=1
    ms=5
    for i in range(0, 32):
        w = np.argsort(attentionTs_mean)[-(i+1)]
        lines0, = ax5.plot([filt_beg_freq0[w], filt_end_freq0[w]], [31-i] * 2, ls='dashed', color='Blue')
        print(filt_beg_freq[w], filt_end_freq[w])
        lines1, = ax5.plot([filt_beg_freq[w], filt_end_freq[w]], [31-i] * 2, linewidth=ms, color='tab:orange',alpha=alpha)
        alpha -= 0.02
        ms -= 0.1
        print(31-i)


    ax5.set(yticklabels=[])
    ax5.set(yticks=[])
    ax5.xaxis.set_ticks(np.arange(0, 41,10))
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    # Create a Rectangle patch
    # rect = patches.Rectangle((0.2, 0.72), 0.7, 0.10, linewidth=2, edgecolor='black', facecolor='wheat',alpha=0.4)
    # Add the patch to the Axes
    # p = ax5_.add_patch(rect)
    lines0.set_label('Initialized Filters')
    lines1.set_label('Learned Filters')

    # p.set_label('4 Most Important Filters')

    # ax5_.text(0.8, 0.8, 'Most \nImportant', horizontalalignment='center',
    #      verticalalignment='center', transform=ax5_.transAxes)

    # ax5_.text(1, 0.15, 'Least \nImportant', horizontalalignment='center',
    #
    #      verticalalignment='center', transform=ax5_.transAxes, bbox=props)
    ax5_.annotate('', xy=(0.12, 0.2), xycoords='axes fraction', xytext=(0.12,0.8),
    arrowprops=dict(arrowstyle="simple", color='black'))
    ax5_.set_zorder(1)
    fig5.legend(bbox_to_anchor=(0.8, 1.0), fontsize='small', handlelength=2)
    ax5.set_xlabel('Frequency (Hz) ')
    ax5.set_ylabel('Filters Ranked by Importance\n')
    # fig5.suptitle('Ranked Filters for Drift and Boundary Prediction')
    fig5.tight_layout()
    fig5.show()

    fig5.savefig(figurepath +   '/'+ finalsubIDs[s] + 'filters_rank_ts' +'.png')

    ##################################3
    ## visulize band pass fitlers by correct and incorrect
    #############################
    # %%


    #################################3
    ## visualize classfication filters ##3
    ######################################



    fig5_choice, ax5_choice_ = plt.subplots(1, figsize=(5, 5.5))
    ax5_choice_.axis('off')
    ax5_choice = fig5_choice.add_axes([0.2, 0.12, 0.7, 0.68])
    p = model_0.state_dict()
    p_low_choice = p['module.sinc_cnn2d_choice.filt_b1']
    p_band_choice = p['module.sinc_cnn2d_choice.filt_band']
    #
    filt_beg_freq0_choice = (torch.abs(p_low_choice) + 1 / 500)
    filt_end_freq0_choice = torch.clamp(filt_beg_freq0_choice + torch.abs(p_band_choice) + 2 / 500, 3/500, 50/500)

    filt_beg_freq0_choice = filt_beg_freq0_choice.cpu().numpy() * 500
    filt_end_freq0_choice = filt_end_freq0_choice.cpu().numpy() * 500

    p = model.state_dict()
    p_low_choice = p['module.sinc_cnn2d_choice.filt_b1']
    p_band_choice = p['module.sinc_cnn2d_choice.filt_band']
    #
    filt_beg_freq_choice = (torch.abs(p_low_choice) + 1 / 500)
    filt_end_freq_choice = torch.clamp(filt_beg_freq_choice + torch.abs(p_band_choice) + 2 / 500, 3/500, 50/500)

    filt_beg_freq_choice = filt_beg_freq_choice.cpu().numpy() * 500
    filt_end_freq_choice = filt_end_freq_choice.cpu().numpy() * 500
    alpha = 1
    ms = 5
    for i in range(0, 32):
        w = np.argsort(attentionTs_mean_choice)[-(i + 1)]
        lines0, = ax5_choice.plot([filt_beg_freq0[w], filt_end_freq0[w]], [31 - i] * 2, ls='dashed', color='Blue')
        print(filt_beg_freq_choice[w], filt_end_freq_choice[w])
        lines1, = ax5_choice.plot([filt_beg_freq_choice[w], filt_end_freq_choice[w]], [31 - i] * 2, linewidth=ms, color='tab:green', alpha=alpha)
        alpha -= 0.02
        ms -= 0.1
        print(31 - i)

    ax5_choice.set(yticklabels=[])
    ax5_choice.set(yticks=[])
    ax5_choice.xaxis.set_ticks(np.arange(0, 41, 10))
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    # Create a Rectangle patch
    # Add the patch to the Axes
    lines0.set_label('Initialized Filters')
    lines1.set_label('Learned Filters')


    # ax5_.text(0.8, 0.8, 'Most \nImportant', horizontalalignment='center',
    #      verticalalignment='center', transform=ax5_.transAxes)

    # ax5_.text(1, 0.15, 'Least \nImportant', horizontalalignment='center',
    #
    #      verticalalignment='center', transform=ax5_.transAxes, bbox=props)


    ax5_choice_.annotate('', xy=(0.12, 0.2), xycoords='axes fraction', xytext=(0.12, 0.8),
                  arrowprops=dict(arrowstyle="simple", color='black'))
    ax5_choice_.set_zorder(1)
    fig5_choice.legend(bbox_to_anchor=(0.68, 1.0), fontsize='small', handlelength=2)
    ax5_choice.set_xlabel('Frequency (Hz) ')
    ax5_choice.set_ylabel('Filters Ranked by Importance\n')
    fig5_choice.tight_layout()
    fig5_choice.show()

    fig5_choice.savefig(figurepath  + '/'+  finalsubIDs[s] + 'filters_rank_ts_choice' + '.png')

    ##################################3
    ## visulize band pass fitlers
    #############################

    import matplotlib.ticker as ticker
    import matplotlib.patches as patches


    def norm(vec):
        f_min, f_max = np.min(vec), np.max(vec)
        newV = 2 * (vec - f_min) / (f_max - f_min) - 1
        return newV


    # fig5, ax5_ = plt.subplots(1, figsize=(5, 5.5))
    # ax5_.axis('off')
    # ax5 = fig5.add_axes([0.2, 0.12, 0.7, 0.68])
    # p = model_0.state_dict()
    # p_low = p['sinc_cnn2d.filt_b1']
    # p_band = p['sinc_cnn2d.filt_band']
    # #
    # filt_beg_freq0 = (torch.abs(p_low) + 1 / 500)
    # filt_end_freq0 = (filt_beg_freq0 + torch.abs(p_band) + 2 / 500)
    #
    # filt_beg_freq0 = filt_beg_freq0.cpu().numpy() * 500
    # filt_end_freq0 = filt_end_freq0.cpu().numpy() * 500
    #
    # p = model.state_dict()
    # p_low = p['sinc_cnn2d.filt_b1']
    # p_band = p['sinc_cnn2d.filt_band']
    # #
    # filt_beg_freq = (torch.abs(p_low) + 1 / 500)
    # filt_end_freq = (filt_beg_freq + torch.abs(p_band) + 2 / 500)
    #
    # filt_beg_freq = filt_beg_freq.cpu().numpy() * 500
    # filt_end_freq = filt_end_freq.cpu().numpy() * 500
    # alpha = 1
    # ms = 5
    # for i in range(0, 32):
    #     w = np.argsort(attentionTs_mean_corr)[-(i + 1)]
    #     lines0, = ax5.plot([filt_beg_freq0[w], filt_end_freq0[w]], [31 - i] * 2, ls='dashed', color='Blue')
    #     print(filt_beg_freq[w], filt_end_freq[w])
    #     lines1, = ax5.plot([filt_beg_freq[w], filt_end_freq[w]], [31 - i] * 2, linewidth=ms, color='green', alpha=alpha)
    #     alpha -= 0.02
    #     ms -= 0.1
    #     print(31 - i)
    #
    # ax5.set(yticklabels=[])
    # ax5.set(yticks=[])
    # ax5.xaxis.set_ticks(np.arange(0, 41, 10))
    # props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    # # Create a Rectangle patch
    #
    # lines0.set_label('Initialized Filters')
    # lines1.set_label('Learned Filters')
    #
    #
    # # ax5_.text(0.8, 0.8, 'Most \nImportant', horizontalalignment='center',
    # #      verticalalignment='center', transform=ax5_.transAxes)
    #
    # # ax5_.text(1, 0.15, 'Least \nImportant', horizontalalignment='center',
    # #
    # #      verticalalignment='center', transform=ax5_.transAxes, bbox=props)
    # ax5_.annotate('', xy=(0.12, 0.2), xycoords='axes fraction', xytext=(0.12, 0.8),
    #               arrowprops=dict(arrowstyle="simple", color='black'))
    # ax5_.set_zorder(1)
    # fig5.legend(bbox_to_anchor=(0.68, 1.0), fontsize='small', handlelength=2)
    # ax5.set_xlabel('Frequency (Hz) ')
    # ax5.set_ylabel('Filters Ranked by Importance\n')
    # fig5.tight_layout()
    # fig5.show()
    #
    #
    # # incorrect
    # fig5, ax5_ = plt.subplots(1, figsize=(5, 5.5))
    # ax5_.axis('off')
    # ax5 = fig5.add_axes([0.2, 0.12, 0.7, 0.68])
    # p = model_0.state_dict()
    # p_low = p['sinc_cnn2d.filt_b1']
    # p_band = p['sinc_cnn2d.filt_band']
    # #
    # filt_beg_freq0 = (torch.abs(p_low) + 1 / 500)
    # filt_end_freq0 = (filt_beg_freq0 + torch.abs(p_band) + 2 / 500)
    #
    # filt_beg_freq0 = filt_beg_freq0.cpu().numpy() * 500
    # filt_end_freq0 = filt_end_freq0.cpu().numpy() * 500
    #
    # p = model.state_dict()
    # p_low = p['sinc_cnn2d.filt_b1']
    # p_band = p['sinc_cnn2d.filt_band']
    # #
    # filt_beg_freq = (torch.abs(p_low) + 1 / 500)
    # filt_end_freq = (filt_beg_freq + torch.abs(p_band) + 2 / 500)
    #
    # filt_beg_freq = filt_beg_freq.cpu().numpy() * 500
    # filt_end_freq = filt_end_freq.cpu().numpy() * 500
    # alpha = 1
    # ms = 5
    # for i in range(0, 32):
    #     w = np.argsort(attentionTs_mean_incorr)[-(i + 1)]
    #     lines0, = ax5.plot([filt_beg_freq0[w], filt_end_freq0[w]], [31 - i] * 2, ls='dashed', color='Blue')
    #     print(filt_beg_freq[w], filt_end_freq[w])
    #     lines1, = ax5.plot([filt_beg_freq[w], filt_end_freq[w]], [31 - i] * 2, linewidth=ms, color='orange', alpha=alpha)
    #     alpha -= 0.02
    #     ms -= 0.1
    #     print(31 - i)
    #
    # ax5.set(yticklabels=[])
    # ax5.set(yticks=[])
    # ax5.xaxis.set_ticks(np.arange(0, 41, 10))
    # props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    # # Create a Rectangle patch
    #
    # lines0.set_label('Initialized Filters')
    # lines1.set_label('Learned Filters')
    #
    # # ax5_.text(0.8, 0.8, 'Most \nImportant', horizontalalignment='center',
    # #      verticalalignment='center', transform=ax5_.transAxes)
    #
    # # ax5_.text(1, 0.15, 'Least \nImportant', horizontalalignment='center',
    # #
    # #      verticalalignment='center', transform=ax5_.transAxes, bbox=props)
    # ax5_.annotate('', xy=(0.12, 0.2), xycoords='axes fraction', xytext=(0.12, 0.8),
    #               arrowprops=dict(arrowstyle="simple", color='black'))
    # ax5_.set_zorder(1)
    # fig5.legend(bbox_to_anchor=(0.68, 1.0), fontsize='small', handlelength=2)
    # ax5.set_xlabel('Frequency (Hz) ')
    # ax5.set_ylabel('Filters Ranked by Importance\n')
    # fig5.tight_layout()
    # fig5.show()

    #%%
    # now that's lets the important time series


    ### visulize training
    timeTr = np.empty((0,len(y_att_test[0][0])))
    for i in y_att_train_list:
        timeTr = np.vstack((timeTr, np.array(torch.squeeze(i[0].cpu().detach()))))

    timeTr_mean = np.mean(timeTr,axis=0)



    weightsD = np.squeeze(np.array(model.module.fc.weight.cpu().detach()))
    weightsD = weightsD/np.max(np.abs(weightsD))

    weightsA = np.squeeze(np.array(model.module.fc_bound.weight.cpu().detach()))
    weightsA = weightsA/np.max(np.abs(weightsA))


    # attentionTt drift and alpha

    # 4 most important frequencies weighted that are used to predict drift
    timeV = np.arange(0,1000,125)+125/2
    fig6,ax6 = plt.subplots(4,2,figsize= (10,10))
    # ax6_ = fig6.add_axes([0.1, 0.1, 0.2, 1])
    timeTr_chopped = (timeTr_mean * weightsD).reshape(-1, 8)
    timeTr_chopped_A = (timeTr_mean * weightsA).reshape(-1,8)

    # for i in range(0,4):
    #     imp = -(i+1)
    #     att_Ind = np.argsort(attentionTr_mean)
    #     f1,f2 = filt_beg_freq[att_Ind[imp]], filt_end_freq[att_Ind[imp]]
    #     ax6[i][0].plot(timeV,(np.mean(timeTr_chopped[att_Ind[imp] * 3 :att_Ind[imp] * 3+3],axis=0).T))
    #     ax6[i][0].set_ylabel('\n%.0f Hz to              \n'% f1 + '%.0f Hz              '% f2,rotation=0)
    #     ax6[i][0].set_xlabel('Time(ms)')
    #     ax6[i][1].set_xlabel('Time(ms)')
    #     ax6[i][1].plot(timeV, np.mean(timeTr_chopped_A[att_Ind[imp] * 3:att_Ind[imp] * 3 + 3], axis=0).T)
    # ax6[0][0].set_title('Weighted Time Course \nof Drift')
    # ax6[0][1].set_title('Weighted Time Course \n of Boundary')
    # fig6.suptitle('Weights of FCL of 4 Most Important Bands')
    # fig6.tight_layout()
    # fig6.show()
    # fig6.savefig('figures_final/' +  finalsubIDs[s] + 'fcp_weights_tr' +'.png')
    #





    ### visulize testing
    timeTs = np.empty((0,len(y_att_test[0][0])))
    for i in y_att_test_list:
        timeTs = np.vstack((timeTs, np.array(torch.squeeze(i[0].cpu().detach()))))

    timeTs_mean = np.mean(timeTs,axis=0)




    ### visulize testing

    def getInd(attentionVec):
        '''returns sorted index, from important to least imporatn'''
        indV =np.argsort(attentionVec)[::-1]
        return indV

    def getfreq(ind):
        f1 = filt_beg_freq[ind]
        f2 = filt_end_freq[ind]
        return np.round(f1), np.round(f2)
    ### visulize testing
    topoTs = np.empty((0,32,98,370))
    for i in y_att_test_list:
        topoTs = np.vstack((topoTs, np.array(torch.squeeze(i[2].cpu().detach()))))


    # make topo over time  ######################3

    # topotTs is the topo plot after sincnet
    topoTs_mean = np.mean(topoTs,axis=0)   # average across subjects
    Ind0,Ind1 = np.arange(0,370,model.module.stride_window), np.arange(0,370,model.module.stride_window) + model.module.pool_window
    pooled = np.zeros((32,num_chan,model.module.pool_window,model.module.output_size))
    for pl in range(model.module.output_size):
        pooled[:,:,:,pl] = topoTs_mean[:,:,Ind0[pl]:Ind1[pl]]
    pooled_mean = np.mean(pooled,axis=-2)
    ind = getInd(attentionTs_mean)

    # fig8,ax8 = plt.subplots(1,8,figsize = (12,1.9))
    window = model.module.pool_window
    stride = model.module.stride_window
    outsize = model.module.output_size
    windowOrig = (window *500)/370 *2 #window length ins
    strideOrig =  (stride *500)/370 *2
    windowVec = np.arange(0,outsize) * strideOrig

    # imp = 0
    # for a, j in enumerate(ax8.flat):
    #     # plottopomap((pooled_mean[ind[imp],:,a]), ax=j) #0-1 normalization or not
    #
    #     plottopomap((pooled_mean[ind[imp],:,a]), ax=j) #0-1 normalization or not
    #     j.set_xlabel('%.0fms to \n '%windowVec[a] +'%.0fms'%np.round(windowVec+windowOrig)[a])
    #
    # # fig8.suptitle('Topoplots over time for %.0f Hz to '%getfreq(ind[imp])[0] +'%.0f Hz' %getfreq(ind[imp])[1])
    # fig8.show()
    # fig8.savefig('figures_final/' + finalsubIDs[s] + 'topo_%sfiltered_overtime_ts'%imp + '.png')
    #

    outsize = model.module.output_size
    windowOrig = (window *500)/370 *2 #window length ins
    strideOrig =  (stride *500)/370 *2
    windowVec = np.arange(0,outsize) * strideOrig




    # make topo over time WEIGHTED

    weights = np.squeeze(np.array(model.module.separable_conv.depthwise.weight.detach().cpu()))

    ##############################
    ##############################
    # make time series over time WEIGHTED
    ##############################
    ##############################



    # make topo with weights   ###

    topoTs_mean = np.mean(topoTs, axis=0)
    Ind0,Ind1 = np.arange(0,370,model.module.stride_window), np.arange(0,370,model.module.stride_window) + model.module.pool_window
    pooled = np.zeros((32,num_chan,model.module.pool_window,model.module.output_size))
    for pl in range(model.module.output_size):
        pooled[:,:,:,pl] = topoTs_mean[:,:,Ind0[pl]:Ind1[pl]]
    pooled_mean = np.mean(pooled,axis=-2)  #average pooled on time winodow



########## weights and time seires plot ##############3
    cmap = hotcold(neutral=0, interp='linear', lutsize=2048)
    mynorm = matplotlib.colors.Normalize(vmin=-1, vmax=1)
    depth = model.module.separable_conv.depthwise.weight.shape[1]
    def getaxis(depth,fig):
        gs = GridSpec(depth, 3, figure=fig)
        axl = []
        if depth >=1:
            ax0 = fig15.add_subplot(gs[0, 0:2])
            ax3 = fig15.add_subplot(gs[0, 2])
            axl.extend([ax0,ax3])
        if depth >= 2:
            ax1 = fig15.add_subplot(gs[1, 0:2])
            ax4 = fig15.add_subplot(gs[1, 2])
            axl.extend([ax1, ax4])
        if depth ==3:
            ax2 = fig15.add_subplot(gs[2, 0:2])
            ax5 = fig15.add_subplot(gs[2, 2])
            axl.extend([ax2, ax5])
        return axl

    for imp in [0,1,2,3]:

        fig15 = plt.figure(figsize=(12, 4*depth))
        axl = getaxis(depth,fig15)

        f1,f2 = getfreq(ind[imp])
        weightsSelected = weights[ind[imp]*depth:ind[imp]*depth+depth,:]
        # weight the time series
        topoTs_mean = np.mean(topoTs,axis=0)
        # topoTs_mean = topoTs[24,:,:]
        topoTs_mean_freq = topoTs_mean[ind[imp],:,:]  # the frequency band of interest

        for a, j in enumerate(axl[int(-len(axl)/2):]):
            im, cn = plottopomap(norm(weightsSelected[a , :]), ax=j)
            ax_pos = j.get_position().get_points()
            ymin_cb, xmin_cb = ax_pos[~np.eye(2, 2, dtype=bool)]
            cb_ax = fig15.add_axes([xmin_cb-0.08, ymin_cb-0.05,0.1,(0.2*1/depth)/0.33])
            cb_ax.axis('off')
            fig15.colorbar(plt.cm.ScalarMappable(norm=mynorm, cmap=cmap),  ax=cb_ax)

            j.set_title('Depth %s Weights'% int(a+1))
        xVec =np.linspace(0,1000,topoTs_mean_freq.shape[1])

        for w, j in enumerate(axl[:int(len(axl)/2)]):
            amp = topoTs_mean_freq.T @ norm(weightsSelected[w, :])
            j.plot(xVec, amp-amp[0])
            j.set_title('Weighted ERP (Depth %s)'%s)
        # j[0].plot(xVec,np.mean(topoTs_mean_freq,axis=0))
        # amp0 = topoTs_mean_freq.T @ norm(weightsSelected[0, :])
        # amp1 = topoTs_mean_freq.T @ norm(weightsSelected[1, :])
        # amp2 = topoTs_mean_freq.T @ norm(weightsSelected[2, :])
        # ax0.plot(xVec,amp0 - amp0[0])
        # ax1.plot(xVec,amp1 - amp1[0])
        # ax2.plot(xVec,amp2 - amp2[0])
        # j[0].set_title('Average ERP without Weights')
        # ax0.set_title('Weighted ERP (Depth 1)')
        # ax1.set_title('Weighted ERP (Depth 2)')
        # ax2.set_title('Weighted ERP (Depth 3)')
        for a, j in enumerate(axl[:int(len(axl)/2)]):
            j.set_xlabel('Times (ms)')
            j.set_ylabel('Amplitude (Normalized)')
            j.axvline(np.median(np.abs(test_target)) * 1000, label = 'median RT',linewidth = 4, alpha=0.8,\
                      color = 'tab:purple')
            if a ==0 :
                j.legend()
        # fig15.text(0.005,0.005,'Weights are normalized on [-1,1]',color= 'red')
        fig15.suptitle('%i Hz to ' % f1 + '%i Hz' % f2 + ' (Drift Predictions)')
        fig15.text(0.5, 0.004, '*Weights are normalized on [-1,1]', color='black',fontsize = 22)

        fig15.tight_layout()
        fig15.show()
        fig15.savefig(figurepath + '/' + finalsubIDs[s] +'erp_weights_%s_filt.png'%imp)



# add more detaisl to the config file

t2 = time.time()
print(t2-t1)

if createConfig:
    config_object.read(modelpath + '/config.ini')
    config_object["loss_func"] = {
        "loss_function": "wfpt, corr(drift, rt), BCEloss, 0.5*sum(boundary)",
        "optimizer_drift": optimizer_drift,
        "optimizer_boundary": optimizer_alpha,
        "optimizer_choice": optimizer_choice
    }
    config_object["time_complexity"] = {
        "time":t2-t1
    }
    with open(modelpath + '/config.ini', 'w') as conf:
        config_object.write(conf)