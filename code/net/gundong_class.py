import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import matplotlib.dates as mdate
from numpy import *
from matplotlib.pylab import rcParams
plt.rcParams['axes.unicode_minus']=False
rcParams['font.sans-serif'] = 'kaiti'

from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm
import statsmodels.stats.diagnostic
from statsmodels.tsa.api import VAR

file_path=''
data = pd.read_csv(file_path, index_col = 0)

class gundong():
    def __init__(self, data, gundong_time, k_lag):
        self.row = data.shape[0] # 行长度
        self.column = data.shape[1] # 列长度
        self.data = data
        self.gundong_time = gundong_time
        self.k_lag = k_lag
        self.save_data_fai1 = np.zeros((self.column, self.column, self.row-self.gundong_time+1, k_lag))
        self.save_data_cov = np.zeros((self.column, self.column, self.row-self.gundong_time+1))
        self.save_data_result = np.zeros((self.column, self.column, self.row-self.gundong_time+1))

        
    
    def VAR(self):
        '''
        实现滚动计算 k-lag 的 VAR 模型
        并且保存矩阵的系数以及相关系数矩阵
        实现了 k-lag>1 时的向量值回归模型
        '''
        for i in range(gundong_time, self.row+1,1):
            datai = self.data.iloc[i-self.gundong_time:i,:]
            model = VAR(datai)
            # 滞后 k_lag 个单位计算
            results = model.fit(self.k_lag)
            coef = results.params
            for j in range(k_lag):
                fai1i = coef.iloc[1+j*self.column:self.column+1+j*self.column,:].T
                self.save_data_fai1[:,:,i-self.gundong_time, j] = fai1i
            self.save_data_cov[:,:,i-self.gundong_time] = results.sigma_u


    def calculate_A(self, h, fai1):
        if h == 0:
            return mat(eye(self.column, self.column, dtype=int))
        else:
            temp = mat(np.zeros((self.column, self.column)))
            for p in range(self.k_lag):
                temp += fai1[:,:,p]*calculate_A(h-p, fail)
            return temp


    def save_result(self, predict_time):
        '''
        适用于不同 k_lag 的向量自回归模型

        predict_time 代表动态预测的天数

        ??? 循环可做矩阵运算优化
        '''
        self.predict_time = predict_time
        for n in range(self.row-self.gundong_time+1):
            fai1_data = self.save_data_fai1[:,:,n,:]
            Covarinace_mat = self.save_data_cov[:,:,n]
            for i in range(self.column):
                ei = mat(eye(self.column, self.column, dtype=int))[:,i]
                for j in range(self.column):
                    sum_top = 0
                    sum_bottom = 0
                    sigma_jj = Covarinace_mat[j,j]
                    ej = mat(eye(self.column, self.column, dtype=int))[:,j]
                    for h in range(self.predict_time):
                        A_h = caculate_A(h, fail = fai1_data)
                        sum_bottom += ei.T*A_h*Covarinace_mat*A_h.T*ei
                        W = ei.T * A_h * Covariance_mat*ej
                        sum_top += 1/sigma_jj * W * W
                    result = sum_top/sum_bottom
                    self.save_data_result[i,j,n] = result
        



    def standard_yichu(self):
        '''
        计算溢出指数的比重

        ??? 循环可做矩阵运算优化
        '''
        for i in range(self.row-self.gundong_time+1):
            sum_line = np.sum(self.save_data_result[:,:,i], axis=1)
            self.save_data_result[:,:,i] = np.divide(self.save_data_result[:,:,i], sum_line)

                
    def save_data(self, path):
        np.save(path+'save_data_fai1',self.save_data_fai1)
        np.save(path+'save_data_cov',self.save_data_cov)
        np.save(path+'save_data_result',self.save_data_result)
    

    def process_data(self):
        '''
        对角为 0 的滚动矩阵
        '''
        self.gundongdata = np.zeros(self.row-self.gundong_time+1)
        for i in range(self.row-self.gundong_time+1):
            xishu_i = self.ave_data_result[:,:,i]
            for x in range(self.column):#对角变为0，求非对角元素
                xishu_i[x,x] = 0
            liehe = xishu_i.sum(axis = 0)#列和
            ave = liehe.mean()
            self.gundongdata[i] = ave
        np.save("save_data_result1",self.gundongdata)
    
    def final_data(self):
        '''
        '''
        self.gundongdata1 = pd.DataFrame(columns = ['values'])
        self.gundongdata1['values'] = self.gundongdata[0:self.row-self.gundong_time+1-self.predict_time]
        self.gundongdata1.index = pd.to_datetime(self.data.index[self.gundong_time+self.predict_time-1:])

    def plot_industry(self):
        '''
        '''
        pass
