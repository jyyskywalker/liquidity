from gundong_class import gundong_class
import numpy as np
import pandas as pd

import matplotlib.pylab as plt
import matplotlib.dates as mdate
from matplotlib.pylab import rcParams
plt.rcParams['axes.unicode_minus']=False
rcParams['font.sans-serif'] = 'kaiti'

path_data='stock.csv' # 读取包含收益率和流动性的文件
data = pd.read_csv(path_data, index_col=0)

path_yield='yield.csv' # 读取只含收益率的文件
yield_data = pd.read_csv(path_yield, index_col=0)

gundong_time=360 # 滚动窗口时间
k_lag = 3

gundong_data = gundong_class(data, gundong_time, k_lag)
gundong_yield = gundong_class(yield_data, gundong_time, k_lag)


gundong_data.VAR()
gundong_data.cal_overflow(predict_time=10)

gundong_yield.VAR()
gundong_yield.cal_overflow(predict_time=10)

data_xishu = gundong_data.get_data_xishu()
yield_xishu = gundong_yield.get_data_xishu()

data_time = gundong_data.data_time()

# yield_xishu 所有求和得到 O
# Y 表示 yield 对yield自己的
# B 表示 yield 对 amihud
# A 表示 amihud 对 yield

length = yield_xishu.shape[1]
# yield 对 yield 的求和
yield_to_yield = data_xishu[:, 0:length, 0:length]
yield_to_yield_sum = yield_to_yield.sum(axis=1)
yield_to_yield_sum = yield_to_yield_sum.sum(axis=1)

# yield 对 amihud 的求和
yield_to_amihud = data_xishu[:, length:2*length, 0:length]
yield_to_amihud_sum = yield_to_amihud.sum(axis=1)
yield_to_amihud_sum = yield_to_amihud_sum.sum(axis=1)

# amihud 对 yield 的求和
amihud_to_yield = data_xishu[:, 0:length, length:2*length]
amihud_to_yield_sum = amihud_to_yield.sum(axis=1)
amihud_to_yield_sum = amihud_to_yield_sum.sum(axis=1)

# 原始的 yield 的求和
yield_sum = yield_xishu.sum(axis=1)
yield_sum = yield_sum.sum(axis=1)

plot_data = pd.DataFrame(columns=['yield'])
plot_data['yield'] = yield_sum
plot_data['yield_to_yield'] = yield_to_yield_sum
plot_data['yield_to_amihud'] = yield_to_amihud_sum
plot_data['amihud_to_yield'] = amihud_to_yield_sum
plot_data.index = pd.to_datetime(data_time)


time = plot_data.index
fig = plt.figure(figsize=(12,9))
ax = plt.subplot(111)
ax.xaxis.set_major_formatter(mdate.DateFormatter('%Y-%m-%d'))
plt.xticks(pd.date_range(time[0],time[-1],freq='6M'),rotation=45)
plt.title('compare')
column_list = [columns for columns in plot_data]
for name in column_list:
    ax.plot(time, plot_data[name],label=name)
plt.show()