import numpy as np
import pandas as pd
from scipy.optimize import minimize
import scipy

def yhat_my_self(p,x):
    y = np.zeros(len(x[0,:]))
    x[-1,0] = p[-1]
    # print(x.shape) print(p.shape)
    y[0] = np.dot(x[:,0].T,p[0:-1])
    # print(y[0]) print(len(x[0,:])-1)
    for ii in range(len(x[0,:])-1):
        x[-1,ii+1] = y[ii]
        y[ii+1] = np.dot(x[:,ii].T, p[0:-1])
    return y



def obj_my_self(p,x,y,tau):
    yfit = yhat_my_self(p,x)
    r = y-yfit
    obj = np.sum(np.abs(r*(tau-(r<0))))
    # print(obj)
    return obj


def quantreg(x,y,tau):
    x_one = np.ones(len(x[0,:]))
    x = np.vstack((x_one,x))
    # print(len(x[0,:]))
    # print(len(y))
    # print(x.shape)
    # p0 = np.linalg.solve(x,y)

    # p0 = np.dot(np.dot(np.linalg.inv(np.dot(x,x.T)), x), y.T)
    # p0 = np.linalg.lstsq(x.T, y, rcond=None)[0]
    x_zeros = np.zeros(len(x[0,:]))
    x = np.vstack((x,x_zeros))
    # p0 = np.hstack((p0,0))
    # p0 = np.hstack((p0,0))
    p0 = np.array([-0.0887,-0.1083,0.2495,-0.0747,0.0689])
    p = minimize(fun = obj_my_self, x0=p0, args=(x,y,tau), method='Nelder-Mead', options={'xatol':1e-8, 'disp':True, 'maxiter':10, 'maxfev':1000})

    return p


