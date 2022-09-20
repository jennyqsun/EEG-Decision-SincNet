# Created on 2/8/22 at 8:29 PM 

# Author: Jenny Sun
# Created on 2/6/22 at 9:17 AM

# Author: Jenny Sun

# Author: Jenny Sun
from hdf5storage import loadmat
import numpy as np
import os
import numpy as np
from sklearn.model_selection import train_test_split
import scipy.io as sio
import pickle
from my_wfpt import neg_wfpt as wfpt
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import scipy
# plt.rcParams.update({'font.size': 24})

postname = '_1000_ddm_2param_attention_bound_uncorr1'
# postname = '1000_ddm_2param_onebd'

resultspath = 'results' + postname + '/'


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


def loadsubjdict(subID):
    path = '/home/jenny/pdmattention/task3/final_interp/'
    datadict = loadmat(path + subID + 'final_interp.mat')
    return datadict, subID


def get_indata(datadict, sub):
    indata_dict = dict()
    rt_train, rt_test = train_test_split(datadict['rt'][0], test_size=0.2, random_state=42)
    acc_train, acc_test = train_test_split(datadict['correct'][0], test_size=0.2, random_state=42)
    condition_train, condition_test = train_test_split(datadict['condition'][0], test_size=0.2, random_state=42)

    indata_dict['rt'] = rt_train
    indata_dict['acc'] = acc_train
    indata_dict['condition'] = condition_train
    indata_dict['participant'] = np.ones(indata_dict['rt'].shape[0], dtype='int64')
    indata_dict['true_participant'] = np.ones(indata_dict['rt'].shape[0], dtype='int64') * int(sub[1:4])

    indata_dict['rt_test'] = rt_test
    indata_dict['acc_test'] = acc_test
    indata_dict['condition_test'] = condition_test

    return indata_dict


sublist, finalsubIDs = getIDs()

fig3, ax10 = plt.subplots(4, 4, figsize=(15, 15))
slist = [4, 26, 32, 33, 24,23,21]
count = 0
# for subid in slist[-2:-1]:
for subid in range(32,33):
    datadict, sub = loadsubjdict(finalsubIDs[subid])
    indata_dict = get_indata(datadict, sub)
    # ddmparams = loadmat('/home/jenny/pdmattention/sincnet/single_nocond_' + finalsubIDs[subid] + '.mat')
    # alpha, ndt, drift = ddmparams['alpha'][0][0][2][0][0], ddmparams['ndt'][0][0][2][0][0], ddmparams['delta'][0][0][2][0][
    #     0]
    sub = finalsubIDs[subid]
    filename = format(sub) + 'results' + postname + '.pkl'
    d = pickle.load(open(resultspath + filename, 'rb'))
    rt_train = d['target_rt_train']
    rt_test = d['target_rt_test']
    drift_train = d['delta_train']
    alpha_train = d['alpha_train']
    drift_test = d['delta_test']
    alpha_test = d['alpha_test']
    ndt = np.min(rt_train) * 0.93

    # d_acc = pickle.load(open(resultspath + format(sub) + 'results_' + postname + 'acc' + '.pkl', 'rb'))
    # rt_train_corr= d_acc['target_rt_train'] > 0
    # rt_test_corr = d_acc['target_rt_test'] > 0

    dparam = pickle.load(open('/home/jenny/pdmattention/sincnet/single_nocond_complete' + sub + '.pkl', 'rb'))
    plt.rcParams["font.serif"] = "Times New Roman"
    plt.rcParams.update({'font.size': 12+6})
    font = 17+6
    plt.rcParams['lines.markersize'] = 5.0

    fig = plt.figure(figsize=(14, 8))
    gs = GridSpec(2, 4, figure=fig)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[0, 2])
    ax3 = fig.add_subplot(gs[0, 3])

    ax4 = fig.add_subplot(gs[1, 0])
    ax5 = fig.add_subplot(gs[1, 1])
    ax6 = fig.add_subplot(gs[1, 2])
    ax7 = fig.add_subplot(gs[1, 3])

    # rt_train_corr = np.ones_like(rt_train, dtype=bool)
    # rt_test_corr = np.ones_like(rt_test, dtype=bool)
    rt_train_corr = rt_train >0
    rt_test_corr = rt_test>0
    ax0.scatter(1 / rt_train[rt_train_corr], drift_train[rt_train_corr], label='correct', color='tab:purple')
    ax0.scatter(-1 / (rt_train[~rt_train_corr]), drift_train[~rt_train_corr], label='incorrect', color='#db2728')
    ax0.set_xlabel('1/RT')
    ax0.set_ylabel(r'$\delta$ Fitted')
    corr_rho_train_delta = scipy.stats.spearmanr(np.abs(1 / rt_train), np.abs(drift_train))
    ax0.set_title('\n\nSpearman 'r'$\rho = %.2f$' % corr_rho_train_delta[0], fontsize=font)
    fig12xlim = ax0.get_xlim()
    fig12ylim = ax0.get_ylim()

    ax1.scatter((rt_train[rt_train_corr]), (alpha_train[rt_train_corr]), label='correct', color='tab:purple')
    ax1.scatter((-(rt_train[~rt_train_corr])), (alpha_train[~rt_train_corr]), label='incorrect', color='#db2728')
    #
    # ax1.scatter(np.log(rt_train[rt_train_corr]), np.log(alpha_train[rt_train_corr]), label='correct', color='tab:purple')
    # ax1.scatter(np.log(-(rt_train[~rt_train_corr])), np.log(alpha_train[~rt_train_corr]), label='incorrect', color='#db2728')
    # ax1.set_xscale('log')
    # ax1.set_yscale('log')
    ax1.set_xlabel('RT')
    ax1.set_ylabel(r'($\alpha$ Fitted)')
    corr_rho_train_alpha = scipy.stats.spearmanr(np.abs(rt_train), np.abs(alpha_train))
    ax1.set_title('\n\nSpearman 'r'$\rho = %.2f$' % corr_rho_train_alpha[0], fontsize=font)
    # fig.text(0.3, 0.95, 'Training Results (Subject %s)'% sub[1:4], transform=fig.transFigure, horizontalalignment='center',fontsize=font)
    # fig.text(0.3, 0.48, 'Test Results (Subject %s)' % sub[1:4], transform=fig.transFigure, horizontalalignment='center',fontsize=font)
    fig34xlim = ax1.get_xlim()
    fig34ylim = ax1.get_ylim()

    # fig.suptitle('Distribution from Training Results (Blue) \nvs. \n Posterior Distribution from JAGS Estimates (Orange)', x=0.75, y=0.9)
    # fig.text(0.75, 0.92, 'Distribution from Training Results (Blue) \nvs. \n Posterior Distribution from JAGS Estimates (Orange)', transform=fig.transFigure, horizontalalignment='center',fontsize=font)
    # fig.text(0.75, 0.45, 'Distribution from Test Results', transform=fig.transFigure, horizontalalignment='center', fontsize=font)

    # v2 = ax2.violinplot(np.abs(drift_train), positions=[0.5], showmedians=True)
    # v2['bodies'][0].set_facecolor('#D43F3A')
    # v2['bodies'][0].set_edgecolor('red')
    # a = np.random.choice(np.reshape(dparam['delta'], 30000), 70, replace=False)
    # ax2.violinplot(np.reshape(dparam['delta'], 30000), positions=[1.2], showmedians=True)
    # ax2.set_ylabel( r'$\delta$ Fitted')
    # ax2.set_xticks([0.5, 1.2])
    # ax2.set_xticklabels(['Decision \nSincNet', 'Bayesian \nMCMC'], rotation=0)
    # fig56ylim = ax2.get_ylim()
    #
    # v3 = ax3.violinplot(alpha_train[rt_train > 0], positions=[0.5], showmedians=True)
    # # v3['bodies'][0].set_facecolor('#D43F3A')
    #
    # ax3.violinplot(np.reshape(dparam['alpha'], 30000), positions=[1.2], showmeans=True)
    # ax3.set_xticks([0.5, 1.2])
    # ax3.set_xticklabels(['Decision \nSincNet', 'Bayesian \nMCMC'], rotation=0)
    # ax3.set_ylabel( r'$\alpha$ Fitted')
    # fig78ylim = ax3.get_ylim()

    ax4.scatter(1 / rt_test[rt_test_corr], drift_test[rt_test_corr], label='correct', color='tab:purple')
    ax4.scatter(-1 / rt_test[~rt_test_corr], drift_test[~rt_test_corr], label='incorrect', color='#db2728')

    ax4.set_xlabel('1/RT')
    ax4.set_ylabel(r'$\delta$ Predicted')
    corr_rho_test_delta = scipy.stats.spearmanr(np.abs(1 / rt_test), np.abs(drift_test))
    ax4.set_xlim(fig12xlim)
    ax4.set_ylim(fig12ylim)

    ax4.set_title('\n\nSpearman 'r'$\rho = %.2f$' % corr_rho_test_delta[0], fontsize=font)
    #
    # ax5.scatter(np.log(rt_test[rt_test_corr]),np.log(alpha_test[rt_test_corr]), color='tab:purple')
    # ax5.scatter(np.log(-(rt_test[~rt_test_corr])), np.log(alpha_test[~rt_test_corr]), color='#db2728')

    ax5.scatter((rt_test[rt_test_corr]),(alpha_test[rt_test_corr]), color='tab:purple')
    ax5.scatter((-(rt_test[~rt_test_corr])),(alpha_test[~rt_test_corr]), color='#db2728')

    # ax1.scatter(-(rt_train[rt_train<0]),alpha_train[rt_train<0], label = 'incorrect', color ='green')
    ax5.set_xlabel('RT')
    ax5.set_ylabel(r'$\alpha$ Predicted')
    corr_rho_test_alpha = scipy.stats.spearmanr(np.abs(rt_test), np.abs(alpha_test))
    ax5.set_title('\n\nSpearman 'r'$\rho = %.2f$' % corr_rho_test_alpha[0], fontsize=font)

    ax5.set_xlim(fig34xlim)
    ax5.set_ylim(fig34ylim)

    # ax6.violinplot(np.abs(drift_test), positions=[0.5], showmedians=True)
    # # v2['bodies'][0].set_facecolor('#D43F3A')
    # # v2['bodies'][0].set_edgecolor('red')
    # # ax2.violinplot(np.reshape(dparam['delta'],30000), positions=[1.2],showmedians=True)
    # ax6.set_ylabel( r'$\delta$ Predicted')
    # ax6.set_xticks([0.5])
    # ax6.set_xticklabels(['Decision \nSincNet'], rotation=0)
    # ax6.set_ylim(fig56ylim)
    #
    # ax7.violinplot(alpha_test[rt_test > 0], positions=[0.5], showmedians=True)
    # # v2['bodies'][0].set_facecolor('#D43F3A')
    # # v2['bodies'][0].set_edgecolor('red')
    # # ax2.violinplot(np.reshape(dparam['delta'],30000), positions=[1.2],showmedians=True)
    # ax7.set_ylabel(r'$\alpha$ Predicted')
    # ax7.set_xticks([0.5])
    # ax7.set_xticklabels(['Decision \nSincNet'], rotation=0)
    # ax7.set_ylim(fig78ylim)

    # ax2.text(1, 0, 'Distribution from Test Results',fontsize=17)
    # fig.suptitle('Subject %s'% sub[1:4] +'\n')

    #### add panel label ####
    # fig.text(0.04, 0.93, 'A', transform=fig.transFigure, horizontalalignment='center', fontsize=font, weight='bold')
    # fig.text(0.28, 0.93, 'B', transform=fig.transFigure, horizontalalignment='center', fontsize=font, weight='bold')
    # fig.text(0.52, 0.93, 'C', transform=fig.transFigure, horizontalalignment='center', fontsize=font, weight='bold')
    # fig.text(0.76, 0.93, 'D', transform=fig.transFigure, horizontalalignment='center', fontsize=font, weight='bold')
    # fig.text(0.04, 0.46, 'E', transform=fig.transFigure, horizontalalignment='center', fontsize=font, weight='bold')
    # fig.text(0.28, 0.46, 'F', transform=fig.transFigure, horizontalalignment='center', fontsize=font, weight='bold')
    # fig.text(0.52, 0.46, 'G', transform=fig.transFigure, horizontalalignment='center', fontsize=font, weight='bold')
    # fig.text(0.76, 0.46, 'H', transform=fig.transFigure, horizontalalignment='center', fontsize=font, weight='bold')
    # ax0.legend()
    # fig.tight_layout()
    # fig.show()
    print(subid)
    print(sub)
    print('train delta', corr_rho_train_delta)
    print('test_delta', corr_rho_test_delta)
    print('train alpha', corr_rho_train_alpha)
    print('test_alpha', corr_rho_test_alpha)

    print('========================================================')
    print('======================END===============================')
    print('========================================================')

    # fig.savefig('result_figures' + '/1000_2param_onebound'+ '/%s' % sub)
    # fig.savefig(resultspath + 'fig_' + '%s' % sub + 'png')
    ax10[0][count].scatter(np.abs(rt_train), drift_train, label='correct', color='tab:purple')
    ax10[2][count].scatter(np.abs(rt_train), (alpha_train), label='correct', color='tab:purple')
    ax10[1][count].scatter(np.abs(rt_test), drift_test, label='correct', color='tab:purple')
    ax10[3][count].scatter(np.abs(rt_test), (alpha_test), color='tab:purple')

    xlimd = ax10[0][count].get_xlim()
    ylimd = ax10[0][count].get_ylim()
    xlima = ax10[2][count].get_xlim()
    ylima = ax10[2][count].get_ylim()
    ax10[2][count].set_xlim(xlima)
    ax10[2][count].set_ylim(ylima)
    ax10[1][count].set_xlim(xlimd)
    ax10[1][count].set_ylim(ylimd)
    ax10[3][count].set_xlim(xlima)
    ax10[3][count].set_ylim(ylima)
    count += 1

font =26
ax10[0][0].set_ylabel(' \nFitted', fontsize=font)

fig3.text(0.5, 0.78, '1/RT', transform=fig3.transFigure, horizontalalignment='center', fontsize=font)
fig3.text(0.02, 0.98, 'A', transform=fig3.transFigure, horizontalalignment='center', fontsize=font, weight='bold')

ax10[1][0].set_ylabel('Drift \nPredicted', fontsize=font)
fig3.text(0.5, 0.54, '1/RT', transform=fig3.transFigure, horizontalalignment='center', fontsize=font)
fig3.text(0.02, 0.74, 'B', transform=fig3.transFigure, horizontalalignment='center', fontsize=font, weight='bold')

ax10[2][0].set_ylabel('Boundary \nFitted (Logged)', fontsize=font)
fig3.text(0.5, 0.3, 'RT', transform=fig3.transFigure, horizontalalignment='center', fontsize=font)
fig3.text(0.02, 0.5, 'C', transform=fig3.transFigure, horizontalalignment='center', fontsize=font, weight='bold')

ax10[3][0].set_ylabel('Boundary \nPredicted (Logged)', fontsize=font)
fig3.text(0.5, 0.05, 'RT', transform=fig3.transFigure, horizontalalignment='center', fontsize=font)
fig3.text(0.02, 0.26, 'D', transform=fig3.transFigure, horizontalalignment='center', fontsize=font, weight='bold')

fig3.tight_layout(h_pad=4)
fig3.subplots_adjust(bottom=0.1)
fig.show()
# fig3.show()
# fig3.savefig('/home/jenny/sincnet_eeg/corr_all.png')
import numpy as np
from scipy.special import rel_entr


# def kl_divergence(a, b):
#     return sum(a[i] * np.log(a[i] / b[i]) for i in range(len(a)))

def kl_divergence(a, b):
    a = np.asarray(a, dtype=np.float)
    b = np.asarray(b, dtype=np.float)

    return np.sum(np.where(a != 0, a * np.log(a / b), 0))


# def kl_divergence(p, q):
# 	return sum(p[i] * log2(p[i]/q[i]) for i in range(len(p)))

def kl(p, q):
    """Kullback-Leibler divergence D(P || Q) for discrete distributions
    Parameters
    ----------
    p, q : array-like, dtype=float, shape=n
    Discrete probability distributions.
    """
    p = np.asarray(p, dtype=np.float)
    q = np.asarray(q, dtype=np.float)

    return np.sum(np.where(p != 0, p * np.log(p / q), 0))


import numpy as np


# def softmax(x):
#     """Compute softmax values for each sets of scores in x."""
#     return np.exp(x) / np.sum(np.exp(x), axis=0)
#
#
# a = softmax(np.random.choice((np.reshape(dparam['delta'], 30000)), size=drift_train.shape[0], replace=False))
# b = softmax(drift_train)
# c = softmax(np.random.choice((np.reshape(dparam['alpha'], 30000))))
# d = softmax(alpha_train)
#
# KL_delta = kl(a, b)
# KL_alpha = kl(c, d)
#

# adding a best fitting line

ax0.margins(x=0)
ax1.margins(x=0)

m, b = np.polyfit(np.abs(1/rt_train), drift_train, 1)
ax0.plot(np.abs(1/rt_train), m*np.abs(1/rt_train) +b,color = 'black')
ax0.plot(np.linspace(ax0.get_xlim()[0],ax0.get_xlim()[1]), m*np.linspace(ax0.get_xlim()[0],ax0.get_xlim()[1]) +b,color = 'grey')

m, b = np.polyfit((np.abs(rt_train)), (alpha_train), 1)
ax1.plot((np.abs(rt_train)), m*(np.abs(rt_train)) +b,color = 'black')
ax1.plot(np.linspace(ax1.get_xlim()[0],ax1.get_xlim()[1]), m*np.linspace(ax1.get_xlim()[0],ax1.get_xlim()[1]) +b,color = 'grey')



m, b = np.polyfit(np.abs(1/rt_test), drift_test, 1)
ax4.plot(np.abs(1/rt_test), m*np.abs(1/rt_test) +b,color = 'black')
ax4.plot(np.linspace(ax4.get_xlim()[0],ax4.get_xlim()[1]), m*np.linspace(ax4.get_xlim()[0],ax4.get_xlim()[1]) +b,color = 'grey')


m, b = np.polyfit((np.abs(rt_test)), (alpha_test), 1)
# ax5.plot(np.log(np.abs(rt_train)), m*np.log(np.abs(rt_train)) +b,color = 'black')
ax5.plot(np.linspace(ax5.get_xlim()[0],ax5.get_xlim()[1]), m*np.linspace(ax5.get_xlim()[0],ax5.get_xlim()[1]) +b,color = 'grey')

fig.tight_layout(h_pad=-1)
fig.show()

# fig.savefig(resultspath + finalsubIDs[subid] + 'final.png')
fig.savefig('figures_final/final_small.png')