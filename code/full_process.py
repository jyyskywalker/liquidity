from process import process_stock
from var_lag_order import var_lag_order
from gundong_class import gundong_class


# 获取列表
stock_list = []
f = open('./VaR/stock_list.txt')
for line in f.readlines():
    line = line.strip('\n')
    stock_list.append(line)


token = '094f15d71394516b730602faa77b1c708007b8d05df300590b4445ed'
starttime = '20180101'
endtime = '20191232'
process = process_stock(stock_list,token)
df = process.create_time_series(starttime, endtime,1000,'limit')
process.save_df(path = './stock2.csv')


data = pd.read_csv('stock2.csv', index_col=0)
varlag = var_lag_order(data)
varlag.test_stationary('5%')
print(varlag.column_list)
print(varlag.get_stock_index())

lag_order = varlag.choice_of_order()
lag_order.summary()

results = varlag.var_regression(k_lag=3, ics='aic')
results.summary()


amihud = pd.read_csv('amihud.csv',index_col = 0)
gundong_amihud = gundong_tensor(amihud,670,3)
gundong_amihud.static_analysis(predict_time=10)

gundong_amihud = gundong_tensor