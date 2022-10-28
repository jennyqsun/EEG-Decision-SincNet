# Created on 11/22/21 at 6:46 AM 

# Author: Jenny Sun
import os
import random

import numpy as np
import matplotlib.pyplot as plt
# the following import is required for matplotlib < 3.2:
from mpl_toolkits.mplot3d import Axes3D  # noqa
import mne
import numpy as np
from hdf5storage import loadmat
import scipy.io
from bipolar import hotcold
hotcoldmap = hotcold(neutral=0, interp='linear', lutsize=2048)


def chansets_new():
    chans= np.arange(0,128)
    chans_del = np.array([56,63,68,73,81,88,94,100,108,114,49,43,48,38,32,44,128,127,119,125,120,121,126,
                  113,117,1,8,14,21,25]) -1
    chans = np.delete(chans, chans_del)
    return chans



def plottopo(data, ax, maskchan, dcolor, dsize):
    group = np.zeros(98)
    for m in range(len(maskchan)):
        group[maskchan[m]] = 1
    # group[maskchan[1]] = 1
    d = dict(marker='o', markerfacecolor= dcolor, markeredgecolor='k', linewidth=0, markersize=dsize, alpha=0.9)
    # biosemi_montage = mne.channels.make_standard_montage('biosemi128')
    # n_channels = len(biosemi_montage.ch_names)
    # np.random.seed(2)
    # data = np.random.uniform(-2,2,(98,1))
    chanlist = chansets_new()

    locdic = scipy.io.loadmat('eginn128hm.mat')['EGINN128']
    # locdic_flu = locdic[0][0][-1][0][0][0]
    nasion_chan = 17-1
    lpa_chan = 48-1
    rpa_chan = 113-1
    locdicchan = locdic[0][0][0][0][0][4][0:-1]
    chan_pos = dict()

    for i, j in enumerate(locdicchan[chanlist]):
        chan_pos[str(chanlist[i]+1)] = j*0.1

    mont = mne.channels.make_dig_montage(ch_pos=chan_pos,
                                         nasion=locdicchan[nasion_chan]*0.1, lpa= locdicchan[lpa_chan]*0.1, rpa = locdicchan[rpa_chan]*0.1)

    fake_info = mne.create_info(ch_names=mont.ch_names, sfreq=500,
                                ch_types='eeg')
    fake_evoked = mne.EvokedArray(data, fake_info)


    fake_evoked.set_montage(mont)
    #
    # mont.plot(show_names=True)
    #
    #     #
    # # fig, ax = plt.subplots(ncols=1, figsize=(8, 4), gridspec_kw=dict(top=0.9),
    # #                        sharex=True, sharey=True)
    #
    mne.viz.plot_topomap(fake_evoked.data[:, 0], fake_evoked.info, axes=ax,names=fake_info['ch_names'],
                         show=False, vmin =0, vmax =0, cmap = 'terrain', mask = group.astype(bool), mask_params =d)






def plottopomap(data, ax, cmap=hotcoldmap):
    '''inputs: '''
    data = np.reshape(data,(-1,1))
    chanlist = chansets_new()

    locdic = scipy.io.loadmat('eginn128hm.mat')['EGINN128']
    # locdic_flu = locdic[0][0][-1][0][0][0]
    nasion_chan = 17-1
    lpa_chan = 48-1
    rpa_chan = 113-1
    locdicchan = locdic[0][0][0][0][0][4][0:-1]
    chan_pos = dict()

    for i, j in enumerate(locdicchan[chanlist]):
        chan_pos[str(chanlist[i]+1)] = j*0.1

    mont = mne.channels.make_dig_montage(ch_pos=chan_pos,
                                         nasion=locdicchan[nasion_chan]*0.1, lpa= locdicchan[lpa_chan]*0.1, rpa = locdicchan[rpa_chan]*0.1)

    basic_info = mne.create_info(ch_names=mont.ch_names, sfreq=500,
                                ch_types='eeg')
    data = np.reshape(data,(98,1))
    evoked = mne.EvokedArray(data, basic_info)

    #
    evoked.set_montage(mont)
    # #
    # mont.plot(show_names=True)

        #
    # fig, ax = plt.subplots(ncols=1, figsize=(8, 4), gridspec_kw=dict(top=0.9),
    #                        sharex=True, sharey=True)

    im, cm = mne.viz.plot_topomap(evoked.data[:,0], evoked.info, axes=ax,names=basic_info['ch_names'],cmap=cmap, show=False)
    return im,cm
