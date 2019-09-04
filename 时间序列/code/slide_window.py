import matplotlib.pylab
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 3.滑动窗口

df = pd.Series(np.random.rand(600), index=pd.date_range('7/1/2016', freq='D', periods=600))

r = df.rolling(window=10)
window_mean = r.mean()

plt.figure(figsize=(15,5))

df.plot(style='r--')
df.rolling(window=10).mean().plot(style='b')
plt.show()
