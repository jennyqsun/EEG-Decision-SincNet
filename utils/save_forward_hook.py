# Created on 10/23/22 at 6:58 PM 

# Author: Jenny Sun


import numpy as np
import torch



def saveForwardHookTest(FEAT, TARGET, resultpath):
    # convert everyhing to be an array and save
    # save each layer
    for k in FEAT.keys():
        FEAT[k] = np.array(FEAT[k])
    for k in FEAT.keys():
        dat = FEAT[k]
        np.save(resultpath + '/' + 'feature_test_' + k, dat)
    # save the key
    keys = []
    for n in FEAT.keys():
        keys.append(n)
        np.save(resultpath + '/' + 'feature_test_keys', keys)

    # save the corresponding rt
    np.save(resultpath + '/' + 'feature_test_rt', np.array(TARGET))
    return



def saveForwardHookTrain(FEAT, TARGET, resultpath):
    # convert everyhing to be an array and save
    # save each layer
    for k in FEAT.keys():
        FEAT[k] = np.array(FEAT[k])
    for k in FEAT.keys():
        dat = FEAT[k]
        np.save(resultpath + '/' + 'feature_train_' + k, dat)
    # save the key
    keys = []
    for n in FEAT.keys():
        keys.append(n)
        np.save(resultpath + '/' + 'feature_train_keys', keys)

    # save the corresponding rt
    np.save(resultpath + '/' + 'feature_train_rt', np.array(TARGET))
    return
