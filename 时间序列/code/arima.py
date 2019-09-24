import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pylab import style
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
import numpy as np


import warnings

warnings.filterwarnings('ignore')



# 4.1 用时间来当作索引,并将日期处理成标准格式
data = pd.read_csv('../../data/all.csv', sep='\t',index_col=[0], parse_dates=[0])
temp_df=data['pre']

#temp_df=temp_df['temp'].resample('Y').mean()
temp_train = temp_df['1988-01':'2017-12']

plt.plot(temp_train)
plt.show()

# 4.2 此时可以发现数据的平稳性较差，进行差分

temp_diff = temp_train.diff(1)
temp_diff.dropna(inplace=True)
temp_diff=temp_diff.diff(12)
temp_diff.dropna(inplace=True)
#stationarity(temp_diff)
plt.figure()
plt.plot(temp_diff)
#plt.title('一阶差分')
plt.show()


# 此时平稳性已经差不多了

# 4.3计算acf值 pacf值

plt.figure()
acf = plot_acf(temp_diff.iloc[13:], lags=20)
plt.title('ACF')
plt.show()
plt.figure()
pacf = plot_pacf(temp_diff.iloc[13:], lags=20)
plt.title('PACF')
plt.show()


model = sm.tsa.statespace.SARIMAX(temp_train,order=(1,0,1),seasonal_order=(1,1,1,12),trend='n')
result = model.fit()
summary = result.summary()

# 4.4 预测，起始时间必须是从已知的时间开始
pred = result.predict('2010-01', '2044-01',dynamic=True)
plt.plot(pred['2019-01':]+2*np.random.rand(301))
#plt.plot(temp_df)
plt.show()
s=pred['2019-01':]
s.to_csv('test.csv')


