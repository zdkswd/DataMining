import pandas as pd
import numpy as np

# 2. 数据重采样
rng = pd.date_range('1/1/2011', periods=90, freq='D')
ts = pd.Series(np.random.rand(len(rng)), index=rng)

# 2.1 降采样
month_resample = ts.resample('2M').sum()

# 2.2 升采样
day3Ts = ts.resample('3D').mean()
print(day3Ts.resample('D').asfreq())

# 插值填充
# ffill空值取前面的值
# bfill空值取后面的值
# interpolate线性取值
ffill = day3Ts.resample('D').ffill(2)
bfill = day3Ts.resample('D').bfill(1)
linear = day3Ts.resample('D').interpolate('linear')
