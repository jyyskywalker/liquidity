import numpy as np
import pandas as pd
import tushare as ts

class getdata():

    def __init__(self, stock_list, token):
        self.stock_list = stock_list
        self.token = token
        ts.set_token(token)
        self.pro = ts.pro_api()

    def get_stock(self, name):
        df = self.pro.daily(ts_code = name, start_date = self.start_date, end_date=self.end_date)
        df = df.iloc[::-1]
        df['loss']=100*(np.log(df['close'])-np.log(df['pre_close']))
        df = df[['trade_date','loss']]
        df.rename(columns={'loss':'loss'+name}, inplace=True)
        return df

    def create_time_series(self, start_date, end_date):
        self.start_date = start_date
        self.end_date = end_date

        df = self.get_stock(self.stock_list[0])
        for i in range(len(self.stock_list)-1):
            df_temp = self.get_stock(self.stock_list[i+1])
            df = pd.merge(df, df_temp)
        self.df = df
        self.df = self.df.dropna(axis=1, how='any')
        self.df['trade_date'] = pd.to_datetime(self.df['trade_date'])
        return self.df

    def save_df(self, path):
        '''
        保存为csv格式
        '''
        self.df.to_csv(path, index = False)
