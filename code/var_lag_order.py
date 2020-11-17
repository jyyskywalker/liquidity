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


class var_lag_order():
    
    def __init__(self, data):
        '''
        '''
        self.row = data.shape[0]
        self.column = data.shape[1]
        self.initial_data = data
        # self.stationary_p_value = stationary_p_value


    def judge_percent(self, test):
        '''
        添加可以选择置信度的功能
        '''
        if test:
            if test[0]<=test[4]['1%']:
                return '1%'
            elif test[0]<=test[4]['5%']:
                return '5%'
            elif test[0]<=test[4]['10%']:
                return '10%'
            else:
                return None
        return None    

    def test_stationary(self, degree='1%'):
        '''
        都是在 1% 的水平条件下满足

        Args: 
            degree: 置信度，1%，5%，10%三种
        '''
        self.column_list = [column for column in self.initial_data]
        self.column_test = {}
        for column in self.column_list:
            test = sm.tsa.stattools.adfuller(self.initial_data[column])
            temp_list = [test[1]]
            temp_list.append(self.judge_percent(test))
            self.column_test[column] = temp_list
        test_list_1 = []
        test_list_2 = []
        for i in range(int(len(self.column_list)/2)):
            if degree=='1%':
                if self.column_test[self.column_list[i]][1]=='1%' and self.column_test[self.column_list[i+int(len(self.column_list)/2)]][1]=='1%':
                    test_list_1.append(self.column_list[i])
                    test_list_2.append(self.column_list[i+int(len(self.column_list)/2)])
            elif degree=='5%':
                if self.column_test[self.column_list[i]][1]=='1%' or '5%' and self.column_test[self.column_list[i+int(len(self.column_list)/2)]][1]=='1%' or '5%':
                    test_list_1.append(self.column_list[i])
                    test_list_2.append(self.column_list[i+int(len(self.column_list)/2)])         
            elif degree=='10%':
                if self.column_test[self.column_list[i]][1]=='1%' or '5%' or '10%' and self.column_test[self.column_list[i+int(len(self.column_list)/2)]][1]=='1%' or '5%' or '10%':
                    test_list_1.append(self.column_list[i])
                    test_list_2.append(self.column_list[i+int(len(self.column_list)/2)])   
            else:
                print('degree selection false')
        self.column_list = test_list_1+test_list_2
        self.process_data = self.initial_data[self.column_list]

    def var_regression(self, k_lag=1):
        '''
        进行 k_lag 滞后的向量自回归并且返回结果
        '''
        self.model = VAR(self.process_data)
        self.results = self.model.fit(k_lag)
        return self.results
        


    def choice_of_order(self, max_lag):
        '''
        返回选取的最优滞后阶数 k-lag
        '''
        self.model.select_order(max_lag)
        self.results = self.model.fit(maxlags = max_lag, ic='aic')
        return self.results

        
        

    def granger_test(self):
        '''
        对每个数据变量进行 Granger 因果性检验
        '''
        


