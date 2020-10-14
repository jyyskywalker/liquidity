import numpy as np
import pandas as pd
import tushare as ts

ts.set_token('094f15d71394516b730602faa77b1c708007b8d05df300590b4445ed')
pro = ts.pro_api()
# 多元化银行
diverse_bank_list = ['000001.SZ','600000.SH','600015.SH','600016.SH','600036.SH','601166.SH',
                     '601288.SH','601328.SH','601398.SH','601818.SH',
                     '601939.SH','601988.SH','601998.SH']
# 区域性银行
regional_bank_list = ['002142.SZ','002807.SZ','002839.SZ','002936.SZ','002948.SZ','002958.SZ',
                      '002966.SZ','600908.SH','600919.SH','600926.SH','600928.SH','601009.SH',
                      '601077.SH','601128.SH','601169.SH','601229.SH','601577.SH','601838.SH',
                      '601860.SH','601997.SH','603323.SH']
# df_1 = pro.daily(ts_code='000001.SZ', start_date='20190101', end_date='20191231')
# df_1.head()



def get_stock(name, start_date, end_date, N_amount=1):
    '''
    日交易额单位：千元（日成交额是否需要改变单位？）
    
    
    
    Args:
        name: stock name
        start_date: trade start date
        end_date: trade end date
        N_amount: 成交额的单位 
    
    '''
    df = pro.daily(ts_code=name, start_date=start_date, end_date=end_date)
    df['yield_rate'] = df['change']/df['pre_close']
    df['abs_yield_rate'] = abs(np.log(df['close']/df['pre_close']))
    df['amihud'] = df['abs_yield_rate']/df['amount']
    return df



def process_stock(name, df):
    df = df[['trade_date','yield_rate','amihud']]
    df.rename(columns={'yield_rate':'yield_rate_'+name, 'amihud':'amihud_'+name},inplace=True)
    return df
    
    
    
def create_time_series(stock_list, start_date, end_date):
    '''
    获取 stock_list 中每只 stock 的 yield rate 以及 amihud 指标并得到时间序列
    
    Args:
    
    '''
    # 先获取第一个
    df = get_stock(stock_list[0], start_date, end_date)
    df = process_stock(stock_list[0], df)
    # 将第一个和后面的合并
    for i in range(len(stock_list)-1):
        df_temp = get_stock(stock_list[i+1], start_date, end_date)
        df_temp = process_stock(stock_list[i+1], df_temp)
        df = pd.merge(df,df_temp)
    return df

def main():
    df = create_time_series(diverse_bank_list,'20190101','20191231')
    return df
    

if __name__ == '__main__':
    df = main()
    ## 改变顺序
    row_list = ['trade_date']
    for name in diverse_bank_list:
        row_list.append('yield_rate_'+name)
    for name in diverse_bank_list:
        row_list.append('amihud_'+name)
    df = df[row_list]
    df.to_csv('stock.csv')