# -*- coding: utf-8 -*-
'''
    进行动态预测的代码

    amihud 的单位对于方差分解没有影响
'''
import pandas as pd
import numpy as np
from numpy import *

import matplotlib.pylab as plt
import matplotlib.dates as mdate
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
        self.column_list = [column for column in self.data]
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


    def __calculate_multiply(self):
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
        matrix_multiply = self.__calculate_multiply()
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
        self.__standard_overflow()
        # return self.save_data_result

    def __standard_overflow(self):
        '''
        计算溢出指数的比重
        '''
        sum_line = np.sum(self.save_data_result, axis=2)
        sum_line = sum_line.reshape(self.save_data_result.shape[0],self.save_data_result.shape[1],1)
        self.save_data_result = np.divide(self.save_data_result, sum_line)
        self.save_data_result = self.save_data_result*100
        return self.save_data_result


    def save_data(self, path='./'):
        '''
        保存三种数据
        '''
        np.save(path+'save_data_coef',self.save_data_coef)
        np.save(path+'save_data_cov',self.save_data_cov)
        np.save(path+'save_data_result',self.save_data_result)
    
# -------------------------


    def __xishu_data_result(self):
        '''
        将每个矩阵的 [j,j] 元素变成0
        '''
        xishu = self.save_data_result.diagonal(axis1=1, axis2=2)
        xishu = np.apply_along_axis(np.diag, 1, xishu)
        self.xishu_data = self.save_data_result-xishu

    def __plot_data_process(self, name):
        '''
        '''
        # gundongdata = zeros(self.row-self.gundong_time+1)

        if name == 'total':
            liehe = self.xishu_data.sum(axis=1)
            ave = liehe.mean(axis=1)
            return ave
        else:
            if name in self.column_list:
                index = self.column_list.index(name)
                out_ = self.xishu_data.sum(axis=1)[:,index]
                in_ = self.xishu_data.sum(axis=2)[:,index]
                net = out_-in_
                return net
            else:
                return None


    def plot(self, name):
        '''

        '''
        self.__xishu_data_result()
        gundongdata = self.__plot_data_process(name)
        gundongdata1 = pd.DataFrame(columns = ['values'])
        gundongdata1['values'] = gundongdata[0:self.row-self.gundong_time-self.predict_time+1]
        gundongdata1.index = pd.to_datetime(self.data.index[self.gundong_time+self.predict_time-1:])

        values = gundongdata1['values']
        time = gundongdata1.index
        fig = plt.figure(figsize=(12,9))
        ax = plt.subplot(111)
        ax.xaxis.set_major_formatter(mdate.DateFormatter('%Y-%m-%d'))#显示日期
        plt.xticks(pd.date_range(time[0],time[-1],freq='6M'),rotation=45)
        plt.title(name)
        ax.plot(time,values,color='#054E9F')
        
        plt.plot(gundongdata1)
        plt.show()

    def static_analysis(self, predict_time, path='xishu.csv'):
        '''
        如果做静态分析直接调用这个函数

        Args:
            predict_time: 预测时间长度
        '''
        if self.row-self.gundong_time+1==1:
            self.VAR()
            self.cal_overflow(predict_time=predict_time)
            ##
            self.__xishu_data_result()
            xishu_data = self.xishu_data[0]
            df = pd.DataFrame(xishu_data, columns = self.column_list)
            df.index = self.column_list
            df.loc['out']=df.apply(lambda x: x.sum())
            df['in']=df.apply(lambda x: x.sum(),axis=1)
            df.to_csv(path)
            return df


    def get_data_result(self):
        '''
        获取计算得到的结果
        '''
        return self.save_data_result

    def get_data_xishu(self):
        '''
        获取xishu data
        '''
        return self.xishu_data


