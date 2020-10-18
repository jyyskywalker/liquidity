# -*- coding: utf-8 -*-
'''
    进行动态预测的代码，使用循环，没有优化的版本
'''
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


class gundong():
    def __init__(self, data, gundong_time, k_lag):
        self.row = data.shape[0] # 行长度
        self.column = data.shape[1] # 列长度
        self.data = data
        self.gundong_time = gundong_time # 滚动选择的时间
        self.k_lag = k_lag
        self.save_data_coef = np.zeros((self.column, self.k_lag*self.column, self.row-self.gundong_time+1))
        self.save_data_cov = np.zeros((self.column, self.column, self.row-self.gundong_time+1))
        self.save_data_result = np.zeros((self.column, self.column, self.row-self.gundong_time+1))

        
    
    def VAR(self):
        '''
        实现滚动计算 k-lag 的 VAR 模型
        并且保存矩阵的系数以及相关系数矩阵
        实现了 k-lag>1 时的向量值回归模型
        '''
        for i in range(self.gundong_time, self.row+1,1):
            datai = self.data.iloc[i-self.gundong_time:i,:]
            model = VAR(datai)
            # 滞后 k_lag 个单位计算
            results = model.fit(self.k_lag)
            coef = results.params
            self.save_data_coef[:,:,i-self.gundong_time]= coef.iloc[1:1+self.k_lag*self.column,:].T
            self.save_data_cov[:,:,i-self.gundong_time] = results.sigma_u


    def calculate_A(self, h, coef):
        '''
        从小到大矩阵迭代
        '''
        A_h = mat(np.zeros((self.k_lag*self.column, self.column)))
        A_h[0:self.column, :] = mat(np.identity(self.column))
        matrix_identity = mat(np.zeros((self.k_lag*self.column, self.column)))
        matrix_identity[0:self.column, :] = mat(np.identity(self.column))
        matrix_left = matrix_identity*coef
        matrix_right = mat(np.zeros((self.k_lag*self.column, self.k_lag*self.column)))
        for j in range(1,self.k_lag):
            matrix_right[j*self.column:(j+1)*self.column, (j-1)*self.column:j*self.column] = mat(np.identity(self.column))
        matrix_multiple = matrix_left+matrix_right
        for i in range(h):
            A_h = matrix_multiple*A_h
        return A_h[0:self.column, :]


    def cal_overflow(self, predict_time):
        '''
        适用于不同 k_lag 的向量自回归模型

        predict_time 代表动态预测的天数

        ??? 循环可做矩阵运算优化
        '''
        self.predict_time = predict_time
        for n in range(self.row-self.gundong_time+1):
            coef_data = self.save_data_coef[:,:,n]
            Covariance_mat = self.save_data_cov[:,:,n]
            for i in range(self.column):
                ei = mat(eye(self.column, self.column, dtype=int))[:,i]
                for j in range(self.column):
                    sum_top = 0
                    sum_bottom = 0
                    sigma_jj = Covariance_mat[j,j]
                    ej = mat(eye(self.column, self.column, dtype=int))[:,j]
                    for h in range(self.predict_time):
                        A_h = self.calculate_A(h, coef=coef_data)
                        sum_bottom += ei.T*A_h*Covariance_mat*A_h.T*ei
                        W = ei.T * A_h * Covariance_mat*ej
                        sum_top += 1/sigma_jj * W * W
                    result = sum_top/sum_bottom
                    self.save_data_result[i,j,n] = result
        



    def standard_overflow(self):
        '''
        计算溢出指数的比重

        ??? 循环可做矩阵运算优化
        '''
        for i in range(self.row-self.gundong_time+1):
            sum_line = np.sum(self.save_data_result[:,:,i], axis=1)
            self.save_data_result[:,:,i] = np.divide(self.save_data_result[:,:,i], sum_line)

                
    def save_data(self, path):
        np.save(path+'save_data_coef',self.save_data_coef)
        np.save(path+'save_data_cov',self.save_data_cov)
        np.save(path+'save_data_result',self.save_data_result)
    

    def overflow_matrix(self):
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
