import numpy as np
from numpy.core.fromnumeric import _all_dispatcher
import pandas as pd
from scipy.optimize import minimize
import scipy
from math import log, sqrt,exp

def llhngarch(par, rfr, x):
    h = x
    z = x
    q = x
    lamb = par[0]
    sigma2 = 1/(1+exp(-par[1]))
    alpha = 1/(1+exp(-par[2]))
    beta = 1/(1+exp(-par[3]))
    rho = 1/(1+exp(-par[4]))
    varphi = 1/(1+exp(-par[5]))

    gam1 = par[6]
    gam2 = par[7]

    q[0] = sigma2
    h[0] = sigma2
    z[0] = (x[0] - rfr - lamb*h[0])/sqrt(h[0])
    for i in range(1:length(z)):
        q[i] = sigma2+rho*(q[i-1]-sigma2)+varphi*(z[i-1]^2-1-2*gam2*sqrt(h[i-1])*z[i-1])
        h[i] = q[i] + beta*(h[i-1]-q[i-1])+alpha*(z[i-1]^2-1-2*gam1*sqrt(h[i-1])*z[i-1])
        z[i] = (x[i] - rfr -lamb*h[i])/sqrt(h[i])

    llsum = 0.5*sum(np.log(h)+z^2)
    
    return llsum



def agarchfit(lamb, sigma2, alpha, beta, rho, varphi, gamma1, gamma2, rfr):
    par = list()
    par[0] = lamb
    par[1] = sigma2
    par[2] = alpha
    par[3] = beta
    par[4] = rho
    par[5] = varphi
    par[6] = gamma1
    par[7] = gamma2

    result = minimize(fun = llhngarch, x0 = par, args = par, method = 'Nelder-Mead', options = options={'xatol':1e-8, 'disp':True, 'maxiter':10, 'maxfev':1000})