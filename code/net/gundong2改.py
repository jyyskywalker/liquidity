# -*- coding: utf-8 -*-
"""
Created on Sat Jan 11 16:44:40 2020

@author: 哎哟喂
"""

import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from numpy import *
from matplotlib.pylab import rcParams
from statsmodels.tsa.stattools import adfuller

import statsmodels.api as sm
import statsmodels.stats.diagnostic
from statsmodels.tsa.api import VAR

save_data_fai1 = np.zeros((11,11,979))
save_data_cov = np.zeros((11,11,979))
save_data_result = np.zeros((11,11,979))
np.save("D:/liquidity/shujv/daima/save_data_fai1",save_data_fai1)
np.save("D:/liquidity/shujv/daima/save_data_cov",save_data_cov)
#save_data[:,:,0].

data= pd.read_csv('D:/liquidity/shujv/now/zong.csv',index_col='index') 
for i in range(360,1339,1):
    datai = data.iloc[i-360:i,:]
    model = VAR(datai)
    results = model.fit(1)
    coef = results.params
    fai1i = coef.iloc[1:12,:].T#系数书写是得出矩阵的转置
    save_data_fai1[:,:,i-360] = fai1i
    cov = results.sigma_u
    save_data_cov[:,:,i-360] = cov

#递归函数
def caculate_A(i, fai1):
    if i == 0:
        return mat(eye(11,11,dtype=int))
    else:
        return fai1 * caculate_A(i-1,fai1)
# 变量 x_j 对变量 x_i 的向前 H 期的广义预测误差方差   
for n in range(979):
    fai1_data = save_data_fai1[:,:,n]
    Covariance_mat =  save_data_cov[:,:,n]
    for i in range(11): 
        ei = mat(eye(11, 11, dtype=int))[:,i]
        for j in range(11):
            sum_top = 0
            sum_bottom = 0
            sigma_jj = Covariance_mat[j,j]
            ej = mat(eye(11, 11, dtype=int))[:,j]
            # 预期天数为 10 天
            for h in range(10):
                A_h = caculate_A(h, fai1 = fai1_data)
                sum_bottom += ei.T * A_h * Covariance_mat * A_h.T * ei
                W = ei.T * A_h * Covariance_mat * ej
                sum_top += 1 / sigma_jj * W * W
            result = sum_top/sum_bottom
            save_data_result[i,j,n]= result
np.save("D:/liquidity/shujv/daima/save_data_result",save_data_result)

#读取滚动原数据

            
#行标准化        
import numpy as np
def standard_data(each):
    orig_data = each
    # 计算比重
    sum_line = np.sum(orig_data, axis=1)
    result = np.divide(orig_data,sum_line)
    return sum_line, result
data = np.load("D:/liquidity/shujv/daima/save_data_result.npy")
save_data_result = np.zeros(shape=data.shape)
for i in range(979):
    each = data[:,:,i]
    SUM, result = standard_data(each)
    save_data_result[:,:,i] = result   
    

gundongdata = zeros(979)
for i in range(979):
    xishu_i = save_data_result[:,:,i]
    for x in range(11):#对角变为0，求非对角元素
        xishu_i[x,x] = 0
    liehe = xishu_i.sum(axis = 0)#列和
    ave = liehe.mean()
    gundongdata[i] = ave
np.save("D:/liquidity/shujv/daima/save_data_result1",gundongdata)#保存对角为0的滚动矩阵
'''
series = zeros(980)
for i in range(978):
    series[i] = (gundongdata[i+1]-gundongdata[i])/gundongdata[i]
np.save("D:/liquidity/shujv/daima/k",series)
k = np.load("D:/liquidity/shujv/daima/k.npy")
k1 = abs(k)
k1.sort()
k1[979]
print (np.where(k==0.8823306709948058))
'''
gundongdata = gundongdata[0:969]
gundongdata1 = pd.DataFrame(columns = ['values'])
gundongdata1['values'] = gundongdata
#转换索引为datatimeindex形式才能画图
gundongdata1.index = pd.to_datetime(data.index[369:])

#画出滚动360天，预测10天的全国动态图
import matplotlib.pyplot as plt
import matplotlib.dates as mdate
from matplotlib.pyplot import rcParams 
rcParams['font.sans-serif'] = 'kaiti'
#???坐标显示问题
values = gundongdata1['values']
time = gundongdata1.index
fig = plt.figure(figsize=(12,9))
ax = plt.subplot(111)
ax.xaxis.set_major_formatter(mdate.DateFormatter('%Y-%m-%d'))#显示日期
plt.xticks(pd.date_range(time[0],time[-1],freq='6M'),rotation=45)
ax.plot(time,values,color='#054E9F')

plt.plot(gundongdata1)
plt.show()

#画出滚动360天，预测10天的公用事业动态图
gundonggongyong = zeros(979)
for i in range(979):
    xishu_i = save_data_result[:,:,i]
    for x in range(11):#对角变为0，求非对角元素
        xishu_i[x,x] = 0
    out_ = xishu_i.sum(axis = 0)[4]#第5列和（公用事业列）
    in_ = xishu_i.sum(axis = 1)[4]#第5行和
    net = out_ - in_
    gundonggongyong[i] = net

gundonggongyong1 = gundongdatagongyong[0:969]
gundonggongyong1 = pd.DataFrame(columns = ['values'])
gundonggongyong1['values'] = gundonggongyong

#转换索引为datatimeindex形式才能画图
gundonggongyong1.index = pd.to_datetime(data.index[369:])

import matplotlib.pyplot as plt
import matplotlib.dates as mdate
from matplotlib.pyplot import rcParams 
plt.rcParams['axes.unicode_minus']=False
rcParams['font.sans-serif'] = 'kaiti'
values = gundonggongyong1['values']
time = gundonggongyong1.index
fig = plt.figure(figsize=(12,9))
ax = plt.subplot(111)
ax.xaxis.set_major_formatter(mdate.DateFormatter('%Y-%m-%d'))#显示日期
plt.xticks(pd.date_range(time[0],time[-1],freq='6M'),rotation=45)
ax.plot(time,values,color='#054E9F')

#画出滚动360天，预测10天的金融动态图
gundongjr = zeros(979)
for i in range(979):
    xishu_i = save_data_result[:,:,i]
    for x in range(11):#对角变为0，求非对角元素
        xishu_i[x,x] = 0
    out_ = xishu_i.sum(axis = 0)[5]#第6列和（金融列）
    in_ = xishu_i.sum(axis = 1)[5]#第6行和
    net = out_ - in_
    gundongjr[i] = net

gundongjr1 = gundongjr1[0:969]    
gundongjr1 = pd.DataFrame(columns = ['values'])
gundongjr1['values'] = gundongjr
#转换索引为datatimeindex形式才能画图
gundongjr1.index = pd.to_datetime(data.index[369:])
import matplotlib.pyplot as plt
import matplotlib.dates as mdate
from matplotlib.pyplot import rcParams 
plt.rcParams['axes.unicode_minus']=False
rcParams['font.sans-serif'] = 'kaiti'
values = gundongjr1['values']
time = gundongjr1.index
fig = plt.figure(figsize=(12,9))
ax = plt.subplot(111)
ax.xaxis.set_major_formatter(mdate.DateFormatter('%Y-%m-%d'))#显示日期
plt.xticks(pd.date_range(time[0],time[-1],freq='6M'),rotation=45)
ax.plot(time,values,color='#054E9F')     

#画出滚动360天，预测10天的地产动态图
gundongdichan = zeros(979)
for i in range(979):
    xishu_i = save_data_result[:,:,i]
    for x in range(11):#对角变为0，求非对角元素
        xishu_i[x,x] = 0
    out_ = xishu_i.sum(axis = 0)[1]#第二列和（地产列）
    in_ = xishu_i.sum(axis = 1)[1]#第二行和
    net = out_ - in_
    gundongdichan[i] = net
gundongdichan1 = gundongdichan1[0:969]    
gundongdichan1 = pd.DataFrame(columns = ['values'])
gundongdichan1['values'] = gundongdichan
#转换索引为datatimeindex形式才能画图
gundongdichan1.index = pd.to_datetime(data.index[369:])

import matplotlib.pyplot as plt
import matplotlib.dates as mdate
from matplotlib.pyplot import rcParams 
plt.rcParams['axes.unicode_minus']=False
rcParams['font.sans-serif'] = 'kaiti'
values = gundongdichan1['values']
time = gundongdichan1.index
fig = plt.figure(figsize=(12,9))
ax = plt.subplot(111)
ax.xaxis.set_major_formatter(mdate.DateFormatter('%Y-%m-%d'))#显示日期
plt.xticks(pd.date_range(time[0],time[-1],freq='6M'),rotation=45)
ax.plot(time,values,color='#054E9F')

#获取2015-12-29 (表格的490 数据的488) 2016-1-4(491) 的索引
gundongdata1.loc['2016-01-01','values']
import numpy as np
#np.where(gundongdata == 6.791032409527788)#索引205
data_1228 = save_data_result[:,:,127]#索引127
data_1229 = save_data_result[:,:,128]#索引128
data_0104 = save_data_result[:,:,129]
data_0105 = save_data_result[:,:,130]
#data_0510 = save_data_result[:,:,206]
#计算NS组成的矩阵 data_xxxx_
data_1229_ = zeros((11,11))
for i in range(11):
    for j in range(11):
        data_1229_[i,j] = data_1229[i,j] - data_1229[j,i]

data_0104_ = zeros((11,11))
for i in range(11):
    for j in range(11):
        data_0104_[i,j] = data_0104[i,j] - data_0104[j,i]
data_0105_ = zeros((11,11))
for i in range(11):
    for j in range(11):
        data_0105_[i,j] = data_0105[i,j] - data_0105[j,i]
        
data_1228_ = zeros((11,11))
for i in range(11):
    for j in range(11):
        data_1228_[i,j] = data_1228[i,j]-data_1228[j,i]
        
#计算MNS边际溢出矩阵
data_0105__= data_0105_ - data_0104_
data_0104__= data_0104_ - data_1229_
data_1229__= data_1229_ - data_1228_

#创建新数组保存净边际溢出矩阵
save_data_result2 = zeros((11,11,979))#NS矩阵
save_data_result3 = zeros((11,11,980))#NS前一期矩阵
save_data_result4 = zeros((11,11,980))#MNS矩阵
for n in range(979):
    for i in range(11):
        for j in range(11):
            save_data_result2[i,j,n] = save_data_result[i,j,n]- save_data_result[j,i,n]#NS矩阵
            save_data_result3[i,j,n+1] = save_data_result[i,j,n] - save_data_result[j,i,n]
save_data_result3 = save_data_result3[:,:,0:979]
save_data_result4 = save_data_result2 - save_data_result3

#计算上1 5 分位数
import numpy as np
percent1 = zeros((11,11))
percent5 = zeros((11,11))
percent10 = zeros((11,11))
save_data_result4abs = np.abs( save_data_result4)

for i in range(11):
    for j in range(11):
        percent = save_data_result4[i,j,:]
        percent1[i,j] = np.percentile(percent,99)
        percent5[i,j] = np.percentile(percent,95)
        percent10[i,j] = np.percentile(percent,90)
        
a = save_data_result[1,1,:]
np.percentile(a,1)#1%分位数
np.percentile(a,5)#5%分位数


xishu_1 = save_data_result[:,:,0]
liehe = xishu_1.sum(axis = 0)#列和
ave = liehe.mean()
ave
xishu_1
xishu1
xishu0
for i in range(10):
    print(i)
    
a = save_data_result[:,:,1068]
Covariance_mat =  save_data_cov[:,:,0]
Covariance_mat[1,1]

#验证第一个 第二个
fai1_data = save_data_fai1[:,:,0]
fai2_data = save_data_fai2[:,:,0]
Covariance_mat =  save_data_cov[:,:,0]
save_data_result = zeros((11,11))
for i in range(11):
    ei = mat(eye(11, 11, dtype=int))[:,i]
    for j in range(11):
        sum_top = 0
        sum_bottom = 0
        sigma_jj = Covariance_mat[j,j]
        ej = mat(eye(11, 11, dtype=int))[:,j]
        for h in range(10):
            A_h = caculate_A(h, fai1=fai1_data, fai2=fai2_data)
            sum_bottom += ei.T * A_h * Covariance_mat * A_h.T * ei
            W = ei.T * A_h * Covariance_mat * ej
            sum_top += 1 / sigma_jj * W * W
        result = sum_top/sum_bottom
        save_data_result[i][j]= result
result1=save_data_result
#data1=data.iloc[0:360,:]
#print(adfuller(data1['cl']))
#print(adfuller(data['dc']))
#print(adfuller(data['dx']))
#print(adfuller(data['gy']))
#print(adfuller(data['gg']))
#print(adfuller(data['jr']))
#print(adfuller(data['kx']))
#print(adfuller(data['ny']))
#print(adfuller(data['rc']))
#print(adfuller(data['xx']))
#print(adfuller(data['yl']))