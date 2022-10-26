# Created on 1/31/22 at 1:53 PM 

# Author: Jenny Sun
import numpy as np
def neg_wfpt(t,v, t0,a):
    tt = (t - t0) / (a**2)
    if t< 0:

        tt = (-t -t0) / (a**2)
        t = -t
        v = -v
    # a-z
    w = 0.5
    # ks = 2 + np.sqrt(-2*tt * np.log(2*np.sqrt(2*np.pi*tt)*err))
    # ks = np.max((ks, np.sqrt(tt)+1))

    K=10
    k = np.arange(-4,6)
    p = np.cumsum((w+2*k)*np.exp(-((w+2*k)**2)/2/tt))
    p = p[-1]/np.sqrt(2*np.pi*tt**3)
    p = (p * np.exp(-v*a*w - (v**2)*t/2)/(a**2))

    # print(p)
    p = np.log(p)
    # print(p)
    # if t < 0:
    #     p = -p
    return(-(p))

def neg_wfpt_flip(t,v, t0,a):
    '''
    flip sign version of the correct wfpt
    :param t:
    :param v:
    :param t0:
    :param a:
    :return:
    '''
    tt = (t - t0) / (a**2)
    if t< 0:
        t = -t
        tt = (t -t0) / (a**2)
    else:
        v=-v


    w = 0.5

    K=10
    k = np.arange(-4,6)
    p = np.cumsum((w+2*k)*np.exp(-((w+2*k)**2)/2/tt))
    p = p[-1]/np.sqrt(2*np.pi*tt**3)
    p = (p * np.exp(-v*a*w - (v**2)*t/2)/(a**2))

    p = np.log(p)
    return(-(p))


def wfpt(t, v, t0, a):
    '''
    simple one bound wfpt negative log likelihood
    :param t: rt
    :param v: drifit rate (negative)
    :param t0: ndt
    :param a: boundary
    :return: negative log likelihood
    '''
    w = 0.5
    tt = (t - t0) / (a**2)
    K=10
    k = np.arange(-4,6)
    p = np.cumsum((w+2*k)*np.exp(-((w+2*k)**2)/2/tt))
    p = p[-1]/np.sqrt(2*np.pi*tt**3)
    p = (p * np.exp(-v*a*w - (v**2)*t/2)/(a**2))

    # print(p)
    p = np.log(p)
    return (-p)

def wfpt_vec(t, v, t0, a):
    '''
    one bound wfpt that accepts vectors.
    vector form of wfpt.
    :param t: rt vector [n]
    :param v: drifit rate (negative) vector []
    :param t0: ndt vector   [n]
    :param a: boundary vector
    :return: negative log likelihood vector
    '''
    if len(t.shape) == 1:
        t = t.reshape(-1, 1)
    if len(v.shape) == 1:
        v = v.reshape(-1, 1)
    if len(a.shape) == 1:
        a = a.reshape(-1, 1)
    t0 = np.array(t0)
    t0 = np.ones_like(t) * t0
    w = np.array(0.5)
    kk = np.arange(-4, 6)
    tt = (t - t0) / (a**2)
    try:
        k = np.tile(kk,(t.shape[0],1))
    except IndexError:
        k = kk.copy()

    err = np.array(0.01)
    tt = np.maximum(np.array(np.abs(t) - t0),err) / np.maximum(err, a) ** 2  # normalized time

    tt_vec = np.tile(tt, (1, len(kk)))
    pp = np.cumsum((w+2*k)*np.exp(-(((w+2*k)**2)/2)/tt_vec),axis=1)
    pp = pp[:,-1]/np.sqrt(2*np.array(np.pi)*np.squeeze(tt)**3)
    pp = pp[:, None]
    # t = torch.where(torch.tensor(t).cuda() > 0, torch.tensor(t).cuda(), torch.tensor(-t).cuda())
    p = (pp * (np.exp(-v * np.maximum(err, a) * w - (v ** 2) * np.array(t) / 2) / (
                np.maximum(err, a) ** 2)))
    # print(p)
    # p = torch.where(torch.tensor(v).cuda()>0, 1*p, 6*p)
    p = np.log(p)
    # p = torch.where(torch.tensor(v).cuda()>0, p, -p)
    # print(t,a,v)
    # print('probability is ', p)

    return -(p.sum())


# t = test_target.detach().squeeze().cpu().numpy()
# v = pred_copy.detach().squeeze().cpu().numpy()
# t0 = ndt.detach().cpu().squeeze().numpy()
# a = pred_1.detach().cpu().squeeze().numpy()