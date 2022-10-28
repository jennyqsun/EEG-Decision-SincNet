# Created on 10/28/22 at 12:56 PM 

# Author: Jenny Sun

import numpy as np

def norm(vec):
    '''normalize the vectors to -1 to 1'''
    f_min, f_max = np.min(vec), np.max(vec)
    newV = 2 * (vec - f_min) / (f_max - f_min) - 1
    return newV


def normZeroOne(vec):
    '''normalize the vectors to 0 to 1'''
    f_min, f_max = np.min(vec), np.max(vec)
    newV = (vec - f_min) / (f_max - f_min)
    return newV