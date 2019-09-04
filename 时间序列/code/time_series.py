import pandas as pd
import numpy as np
import datetime as dt

# 1.创建一个时间序列
# 1.1 date_range
# H M D 小时 月 天
rng = pd.date_range('2016/07/01', periods=10, freq='2D3H')

time = pd.Series(np.random.rand(20), index=pd.date_range(dt.datetime(2019, 1, 1), periods=20))

# 1.2 truncate过滤掉
truncate = time.truncate(before='2019-01-08')

# 1.3 使用时间索引
print(time['2019-01-02'])

print(time['2019-01-03':'2019-01-06'])

# 1.4 时间戳
print(pd.Timestamp('2016-07-01'))

# 可以指定更多的细节
print(pd.Timestamp('2016-07-01 10'))

print(pd.Timestamp('2016-07-01 10:15'))

# 1.5 时间区间
print(pd.Period('2016-01'))

print(pd.Period('2016-01-01'))

# 1.6 Time offsets
print(pd.Timedelta('1 day'))

print(pd.Period('2019-01-01') + pd.Timedelta('1 day'))
print(pd.Timestamp('2019-01-01 10:10') + pd.Timedelta('15 ns'))

period = pd.Period('2019-01-01', freq='M')

# 1.7 时间戳和周期可以相互转换
ts = pd.Series(range(10), pd.date_range('2019-09-01', periods=10, freq='D'))
ts_period = ts.to_period()
# 时间戳与时间周期的区别在于如果有一点时间包含在那个时间周期中，切片时那个时间周期全部选中
