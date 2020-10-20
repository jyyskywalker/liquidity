# -*- coding: utf-8 -*-
'''
    进行动态预测的代码，用张量优化的版本
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


class gundong_tensor():
    def __init__(self, data, gundong_time, k_lag):
        self.row = data.shape[0] # 行长度
        self.column = data.shape[1] # 列长度
        self.data = data
        self.gundong_time = gundong_time # 滚动选择的时间
        self.k_lag = k_lag
        self.save_data_coef = np.zeros((self.row-self.gundong_time+1, self.column, self.k_lag*self.column))
        self.save_data_cov = np.zeros((self.row-self.gundong_time+1, self.column, self.column))
        self.save_data_result = np.zeros((self.row-self.gundong_time+1, self.column, self.column))

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
            self.save_data_coef[i-self.gundong_time,:,:]= coef.iloc[1:1+self.k_lag*self.column,:].T
            self.save_data_cov[i-self.gundong_time,:,:] = results.sigma_u


    def calculate_multiply(self):
        # 初始的 A_0,...A_{1-p}

        # 第一个分块矩阵是单位阵
        matrix_identity = np.zeros((self.k_lag*self.column, self.column))
        matrix_identity[0:self.column,:] = np.identity(self.column)
        matrix_identity = np.expand_dims(matrix_identity,0).repeat(self.row-self.gundong_time+1, axis=0)
        matrix_left = np.matmul(matrix_identity, self.save_data_coef)
        matrix_right = np.zeros((self.k_lag*self.column, self.k_lag*self.column))
        for j in range(1,self.k_lag):
            matrix_right[j*self.column:(j+1)*self.column, (j-1)*self.column:j*self.column] = np.identity(self.column)
        matrix_right = np.expand_dims(matrix_right,0).repeat(self.row-self.gundong_time+1, axis=0)
        matrix_multiply = matrix_left+matrix_right
        return matrix_multiply


    def cal_overflow(self, predict_time):
        '''
            适用于不同 k_lag 的向量自回归模型
            
            张量乘法运算 a*b*c 维张量 matmul a*c*d 维张量结果是 a*b*d 维张量

            Args:
                predict_time: 预测天数
        '''
        self.predict_time = predict_time
        # 初始的 A_h 矩阵 
        self.A_h = np.zeros((self.row-self.gundong_time+1, self.k_lag*self.column, self.column))
        self.A_h[:, 0:self.column,:] = np.identity(self.column)
        # 得到
        matrix_multiply = self.calculate_multiply()
        temp = np.matmul(self.A_h[:,0:self.column,:],self.save_data_cov)
        sum_top = temp*temp
        # 得到一个对角阵
        sigma_jj = self.save_data_cov.diagonal(axis1=1, axis2=2)
        sigma_jj = np.apply_along_axis(np.diag, 1, sigma_jj)
        # A_h * cov * A_h'
        temp_bottom = np.matmul(temp, self.A_h[:,0:self.column,:].transpose(0,2,1))
        # 每行元素都是对角线元素
        temp_bottom = temp_bottom.diagonal(axis1=1, axis2=2)[:,np.newaxis].transpose(0,2,1).repeat(self.column,2)
        # * sigma_jj
        sum_bottom = np.matmul(temp_bottom, sigma_jj)
        for h in range(self.predict_time-1):
            self.A_h = np.matmul(matrix_multiply, self.A_h)
            temp = np.matmul(self.A_h[:,0:self.column,:], self.save_data_cov)
            sum_top = sum_top + temp*temp
            temp_bottom = np.matmul(temp, self.A_h[:,0:self.column,:].transpose(0,2,1))
            temp_bottom = temp_bottom.diagonal(axis1=1, axis2=2)[:,np.newaxis].transpose(0,2,1).repeat(self.column,2)
            sum_bottom = sum_bottom + np.matmul(temp_bottom, sigma_jj)

        self.save_data_result = sum_top/sum_bottom

        def standard_overflow(self):
            '''
            计算溢出指数的比重


            '''
            pass


    def save_data(self, path):
        np.save(path+'save_data_coef',self.save_data_coef)
        np.save(path+'save_data_cov',self.save_data_cov)
        np.save(path+'save_data_result',self.save_data_result)

