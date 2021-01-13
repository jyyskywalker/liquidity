import numpy as np
import pandas as pd
import tushare as ts

class process_stock():

    def __init__(self, stock_list, token):
        self.stock_list = stock_list
        self.token = token
        ts.set_token(token)
        self.pro = ts.pro_api()
        
    def get_stock(self, name, market_type):
        '''
        日交易额单位：千元*N_amount

        Args:
            name: stock name
            market_type: nolimit or limit
        '''
        if market_type == 'nolimit':
            df = self.pro.daily(ts_code = name, start_date = self.start_date, end_date=self.end_date)
            df['yield_rate'] = df['change']/df['pre_close']
            df['abs_yield_rate'] = abs(np.log(df['close']/df['pre_close']))
            df['amihud'] = df['abs_yield_rate']/(df['amount']/self.N_amount)
            return df
        else:
            df = self.pro.daily(ts_code = name, start_date = self.start_date, end_date=self.end_date)
            df['yield_rate'] = df['change']/df['pre_close']
            df['max'] = df.apply(lambda x:max(x['pre_close'], x['high']), axis=1)
            df['min'] = df.apply(lambda x:min(x['pre_close'], x['low']), axis=1)
            df['abs_yield_rate'] = abs(np.log(df['max']/df['min']))
            df['amihud'] = df['abs_yield_rate']/(df['amount']/self.N_amount)
            return df
    

    
    def process_stock(self, name, df):
        '''
        '''
        df = df[['trade_date','yield_rate','amihud']]
        df.rename(columns={'yield_rate':'yield_rate_'+name, 'amihud':'amihud_'+name},inplace=True)
        return df

    def create_time_series(self, start_date, end_date, N_amount=1, market_type='nolimit'):
        '''
        获取 stock_list 中每只 stock 的 yield rate 以及 amihud 指标并得到时间序列
        
        Args:
            name: stock name
            start_date: trade start date
            end_date: trade end date
            N_amount: 成交额的单位
        '''
        # 先获取第一个
        self.start_date = start_date
        self.end_date = end_date
        self.N_amount = N_amount
        df = self.get_stock(self.stock_list[0], market_type)
        df = self.process_stock(self.stock_list[0], df)
        # 将第一个和后面的合并
        for i in range(len(self.stock_list)-1):
            df_temp = self.get_stock(self.stock_list[i+1], market_type)
            df_temp = self.process_stock(self.stock_list[i+1], df_temp)
            df = pd.merge(df,df_temp)
        self.df = df
        # 改变顺序
        row_list1 = ['trade_date']
        row_list2 = []
        for name in self.stock_list:
            row_list1.append('yield_rate_'+name)
            row_list2.append('amihud_'+name)
        self.df = self.df[row_list1+row_list2]
        self.df = self.df.dropna(axis=1, how='any')
        self.df = self.df.iloc[::-1]
        self.df['trade_date'] = pd.to_datetime(self.df['trade_date'])
        return self.df

    def save_df(self, path):
        '''
        保存为csv格式
        '''
        self.df.to_csv(path, index = False)
        
