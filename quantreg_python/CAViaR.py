import numpy as np
import pandas as pd
from scipy.optimize import minimize
from quantreg import quantreg
from quantreg import yhat_my_self

import matplotlib.pylab as plt
import matplotlib.dates as mdate
from matplotlib.pylab import rcParams
plt.rcParams['axes.unicode_minus']=False
rcParams['font.sans-serif'] = 'kaiti'


df = pd.read_csv('.\indices_since95.csv')
indices = df['000001.SH']
indices = indices.values
indices = indices[0:-1]

y = np.log(indices[1:])-np.log(indices[0:-1])

x_positive = np.abs(y[0:-1])*(y[0:-1]>0)
x_negative = np.abs(y[0:-1])*(y[0:-1]<0)

x = np.vstack((x_positive,x_negative))
y0 = y[1:]

tau = 0.1

p = quantreg(x,y0,tau)
print("p的值是：")
print(p.x)

xf = x
x_ones = np.ones(len(xf[0,:]))
xf = np.vstack((x_ones,xf))
x_zeros = np.zeros(len(x[0,:]))
xf = np.vstack((xf,x_zeros))

yfit = yhat_my_self(p.x,xf)

plot(y0)
plot(yfit,'r')