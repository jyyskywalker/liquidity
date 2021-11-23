import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdate
import matplotlib


VaR_list = []
log_yield_list = []
f = open('stock_list.txt')
for line in f.readlines():
    line = line.strip('\n')
    VaR_list.append(line+'VaR')
    log_yield_list.append(line+'yield')

VaR = pd.read_csv('./VaR.csv',index_col=0)
log_yield = pd.read_csv('./log_yield.csv',index_col=0)
VaR.columns = VaR_list
log_yield.columns = log_yield_list

VaR.index = pd.to_datetime(VaR.index,format='%Y%m%d')
time = VaR.index
fig = plt.figure(figsize=(12,9))
# ax = plt.subplot(111)
# plt.xticks(pd.date_range(time[0],time[-1],freq='6M'),rotation=45)
for name in VaR.columns:
    plt.plot(time, VaR[name])

plt.legend(VaR_list)
plt.show()

# 画对数收益率的代码
# log_yield.index = pd.to_datetime(log_yield.index,format='%Y%m%d')
# time = log_yield.index
# fig = plt.figure(figsize=(12,9))
# for name in log_yield.columns:
#    plt.plot(time, log_yield[name])
#plt.legend(log_yield_list)
#plt.show()