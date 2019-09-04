import pandas as pd
import pandas_datareader
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.pylab import style
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# 4.ARIMA实例 预测股票


style.use('ggplot')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 4.1 用时间来当作索引,并将日期处理成标准格式
stock = pd.read_csv('../data/nas.csv', index_col=0, parse_dates=[0])

stock_week = stock['Close'].resample('W-MON').mean()
stock_train = stock_week['2018-09':'2019-07']

stock_train.plot(figsize=(12, 8))
plt.legend(bbox_to_anchor=(1.25, 0.5))
plt.title('stock close prize')
sns.despine()
plt.show()

# 4.2 此时可以发现数据的平稳性较差，进行差分

stock_diff = stock_train.diff()
stock_diff = stock_diff.dropna()

plt.figure()
plt.plot(stock_diff)
plt.title('一阶差分')
plt.show()

stock_diff = stock_diff.diff()
stock_diff = stock_diff.dropna()

plt.figure()
plt.plot(stock_diff)
plt.title('二阶差分')
plt.show()

stock_diff = stock_diff.diff()
stock_diff = stock_diff.dropna()

plt.figure()
plt.plot(stock_diff)
plt.title('三阶差分')
plt.show()

stock_diff = stock_diff.diff()
stock_diff = stock_diff.dropna()

plt.figure()
plt.plot(stock_diff)
plt.title('四阶差分')
plt.show()

# 此时平稳性已经差不多了

# 4.3计算acf值 pacf值
acf = plot_acf(stock_diff, lags=20)
plt.title('ACF')
pacf = plot_pacf(stock_diff, lags=20)
plt.title('PACF')
plt.show()

# 由图acf(q) 2  pacf(p) 1 截尾 order p d q
model = ARIMA(stock_train,order=(1,2,2), freq='W-MON')

result = model.fit()
summary = result.summary()

# 4.4 预测，起始时间必须是从已知的时间开始
pred = result.predict('2019-04', '2019-10', dynamic=True, type='levels')
