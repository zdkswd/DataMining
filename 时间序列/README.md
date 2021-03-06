# 1.时间序列

## 1.1 date_range

## 1.2 truncate过滤掉

## 1.3 使用时间索引

## 1.4 时间戳

## 1.5 时间区间

## 1.6 Time offsets

## 1.7 时间戳和周期可以相互转换
时间戳与时间周期的区别在于如果有一点时间包含在那个时间周期中，切片时那个时间周期全部选中

# 2.数据重采样

## 2.1 降采样

## 2.2 升采样
插值填充
ffill空值取前面的值
bfill空值取后面的值
interpolate线性取值
# 3.滑动窗口

![slide_window](assets/slide_window.png)

# 4.ARIMA实例 预测股票
## 4.1 用时间来当作索引,并将日期处理成标准格式
## 4.2 此时可以发现数据的平稳性较差，进行差分
## 4.3计算acf值 pacf值
## 4.4 预测，起始时间必须是从已知的时间开始

# ARIMA概念

## 自回归模型(AR)

1. 描述当前值与历史值之间的关系，用变量自身的历史时间数据对自身进行预测。
2. 自回归模型必须满足平稳性的要求。

## 自回归模型的限制

1. 自回归模型是用自身的数据进行预测。
2. 必须具有平稳性。
3. 必须具有自相关性，如果自相关系数小于0.5，则不宜采用。
4. 自回归只适用于预测与自身前期相关的现象。

## 移动平均模型（MA）

1. 移动平均模型关注的是自回归模型中误差项的累加。
2. 移动平均法能有效地消除预测中的随机波动。

## 自回归移动平均模型(ARMA)

1. 自回归与移动平均的结合。

## ARIMA 差分自回归移动平均模型

AR是自回归，p为自回归项，MA为移动平均，q为移动平均数，d为时间序列称为平稳时所做的差分次数。

原理：将非平稳时间序列转化为平稳时间序列然后将因变量仅对它的滞后值以及随机误差项的现值和滞后值进行回归所建立的模型。

## 自相关系数ACF

1. 有序的随机变量序列与其自身相比较，自相关函数反映了同一序列在不同时序的取值之间的相关性。
2. 取值范围是[-1,1]

## 偏自相关函数 PACF

1. ACF还包括了其他变量的影响，偏自相关系数PACF是严格这两个变量之间的相关性。

