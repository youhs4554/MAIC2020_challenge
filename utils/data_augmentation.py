import numpy as np
from scipy.interpolate import CubicSpline      # for warping


def DA_Jitter(X, sigma=0.05):
    myNoise = np.random.normal(loc=0, scale=sigma, size=X.shape)
    return X+myNoise


def DA_Scaling(X, sigma=0.1):
    scalingFactor = np.random.normal(
        loc=1.0, scale=sigma, size=(1, X.shape[1]))  # shape=(1,3)
    myNoise = np.matmul(np.ones((X.shape[0], 1)), scalingFactor)
    return X*myNoise


def DA_Permutation(X, nPerm=4, minSegLength=100):
    X_new = np.zeros(X.shape)
    idx = np.random.permutation(nPerm)
    bWhile = True
    while bWhile == True:
        segs = np.zeros(nPerm+1, dtype=int)
        segs[1:-1] = np.sort(np.random.randint(minSegLength,
                                               X.shape[0]-minSegLength, nPerm-1))
        segs[-1] = X.shape[0]
        if np.min(segs[1:]-segs[0:-1]) > minSegLength:
            bWhile = False
    pp = 0
    for ii in range(nPerm):
        x_temp = X[segs[idx[ii]]:segs[idx[ii]+1], :]
        X_new[pp:pp+len(x_temp), :] = x_temp
        pp += len(x_temp)
    return(X_new)


def GenerateRandomCurves(X, sigma=0.2, knot=4):
    xx = (np.ones((X.shape[1], 1))*(np.arange(0,
                                              X.shape[0], (X.shape[0]-1)/(knot+1)))).transpose()
    yy = np.random.normal(loc=1.0, scale=sigma, size=(knot+2, X.shape[1]))
    x_range = np.arange(X.shape[0])
    cs_x = CubicSpline(xx[:, 0], yy[:, 0])
    return np.array([cs_x(x_range)]).transpose()


def DistortTimesteps(X, sigma=0.2):
    # Regard these samples aroun 1 as time intervals
    tt = GenerateRandomCurves(X, sigma)
    # Add intervals to make a cumulative graph
    tt_cum = np.cumsum(tt, axis=0)
    # Make the last value to have X.shape[0]
    t_scale = [(X.shape[0]-1)/tt_cum[-1, 0], ]
    tt_cum[:, 0] = tt_cum[:, 0]*t_scale[0]
    return tt_cum


def DA_TimeWarp(X, sigma=0.2):
    tt_new = DistortTimesteps(X, sigma)
    X_new = np.zeros(X.shape)
    x_range = np.arange(X.shape[0])
    X_new[:, 0] = np.interp(x_range, tt_new[:, 0], X[:, 0])
    return X_new
