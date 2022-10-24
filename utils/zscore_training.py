# Created on 10/3/22 at 4:06 PM 

# Author: Jenny Sun
import sklearn
from sklearn.preprocessing import StandardScaler
import numpy as np


def concat_trials(data):
    ''':arg:  trial x channel x time EEG data
       :return: channel x time EEG data, original data
        the function also checks whether the concatenante is correct or not.
    '''
    data_concat = np.concatenate(data,axis=1)
    return data_concat

def zscore(data):
    ''':param  trial x channel x time original EEG data
       :return: transformed data, origianl data, zscore model
    '''
    # zscore the data
    data_concat = concat_trials(data)
    numTrial,numChan, n = data.shape[0], data.shape[1],data.shape[2]
    scaler = StandardScaler()
    scaler.fit(data_concat.T)
    dataNew = scaler.transform(data_concat.T)

    dataTrans = np.empty_like(data)
    for trial in range(numTrial):
        dataTrans[trial, :,:] = dataNew[n*trial:n*trial+n,:].T
    return dataTrans,data, scaler

def ztransform(data, model):
    '''
    :param data: trial x channel x time original EEG data
    :param model: zscore model
    :return: transformeddata, data
    '''
    data_concat = concat_trials(data)
    numTrial,numChan, n = data.shape[0], data.shape[1],data.shape[2]
    dataNew = model.transform(data_concat.T)

    dataTrans = np.empty_like(data)
    for trial in range(numTrial):
        dataTrans[trial, :,:] = dataNew[n*trial:n*trial+n,:].T
    return dataTrans, data


