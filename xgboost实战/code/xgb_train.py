import xgboost as xgb
import pandas as pd
import numpy as np
import pickle
import sys
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, make_scorer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from scipy.sparse import csr_matrix, hstack
from sklearn.model_selection import KFold, train_test_split
from xgboost import XGBRegressor

import warnings

warnings.filterwarnings('ignore')

# 7.数据预处理
train = pd.read_csv('../data/train.csv')
# 做对数变换
train['log_loss'] = np.log(train['loss'])
# 数据分为离散和连续特征
features = [x for x in train.columns if x not in ['id', 'loss', 'log_loss']]
cat_features = [x for x in train.select_dtypes(include=['object']).columns if x not in ['id', 'loss', 'log_loss']]
num_features = [x for x in train.select_dtypes(exclude=['object']).columns if x not in ['id', 'loss', 'log_loss']]

ntrain = train.shape[0]
train_x = train[features]
train_y = train['log_loss']

# 将类别特征转换
for c in range(len(cat_features)):
    train_x[cat_features[c]] = train_x[cat_features[c]].astype('category').cat.codes


# 8.训练xgb模型，
# 训练一个基本的xgb模型，进行参数调节通过交叉验证来观察结果的变换
# 使用平均绝对误差来衡量
# xgboost自定义了一个数据矩阵类DMatrix，会在训练开始时进行一遍预处理，从而提高之后每次迭代的效率
def xg_eval_mae(yhat, dtrain):
    y = dtrain.get_label()
    return 'mae', mean_absolute_error(np.exp(y), np.exp(yhat))


dtrain = xgb.DMatrix(train_x, train['log_loss'])

# 调参的初始指标，经验值
xgb_params = {
    'seed': 0,
    'eta': 0.1,
    'colsample_bytree': 0.5,
    'silent': 1,
    'subsample': 0.5,
    'object': 'reg:linear',
    'max_depth': 5,
    'min_child_weight': 3
}

# 9.使用交叉验证
''''
bst_cv1 = xgb.cv(xgb_params, dtrain, num_boost_round=50, nfold=3, seed=0,
                 feval=xg_eval_mae, maximize=False, early_stopping_rounds=10)
'''

# 此时得到了第一个基准结果

# 画图查看
''''
plt.figure()
bst_cv1[['train-mae-mean', 'test-mae-mean']].plot()
plt.show()
'''

# 10.总结第一个模型
# 没有发生过拟合
# 只建立了50个树模型


# 11.试着改进模型
# 建立100个树模型
''''
bst_cv2 = xgb.cv(xgb_params, dtrain, num_boost_round=100, nfold=3, seed=0,
                 feval=xg_eval_mae, maximize=False, early_stopping_rounds=10)
'''

# 绘图查看结果
'''
fig, (ax1, ax2) = plt.subplots(1, 2)
fig.set_size_inches(16, 4)

ax1.set_title('100 rounds of training')
ax1.set_xlabel('rounds')
ax1.set_ylabel('loss')
ax1.grid(True)
ax1.plot(bst_cv2[['train-mae-mean', 'test-mae-mean']])
ax1.legend(['training loss', 'test loss'])

ax2.set_title('last 60 rounds')
ax2.set_xlabel('rounds')
ax2.set_ylabel('loss')
ax2.grid(True)
ax2.plot(bst_cv2.iloc[40:][['train-mae-mean', 'test-mae-mean']])
ax2.legend(['training loss', 'test loss'])

plt.show()
'''


# 放大来看，后60轮测试效果没有训练效果好，有一丢丢过拟合，但是问题不大

# 此时也得到了新的mae，比第一个模型的效果要好，接下来要改变其他的参数了


# 12.xgb参数调节
# step1：选择一组初始参数
# step2：改变max_depth和min_child_weight 控制树模型的复杂程度
# step3：调节gamma降低模型过拟合风险
# step4：调节subsample和colsample_bytree改变数据采样策略
# step5：调节学习率eta
class XGBoostRegressor(object):
    def __init__(self, **kwargs):
        self.params = kwargs
        if 'num_boost_round' in self.params:
            self.num_boost_round = self.params['num_boost_round']
        self.params.update({'silent': 1, 'objective': 'reg:linear', 'seed': 0})

    def fit(self, x_train, y_train):
        dtrain = xgb.DMatrix(x_train, y_train)
        self.bst = xgb.train(params=self.params, dtrain=dtrain, num_boost_round=self.num_boost_round,
                             feval=xg_eval_mae, maximize=False)

    def predict(self, x_pred):
        dpred = xgb.DMatrix(x_pred)
        return self.bst.predict(dpred)

    def kfold(self, x_train, y_train, nfold=5):
        dtrain = xgb.DMatrix(x_train, y_train)
        cv_rounds = xgb.cv(params=self.params, dtrain=dtrain, num_boost_round=self.num_boost_round,
                           nfold=nfold, feval=xg_eval_mae, maximize=False, early_stopping_rounds=10)
        return cv_rounds.iloc[-1, :]

    def plot_feature_importance(self):
        feat_imp = pd.Series(self.bst.get_fscore()).sort_values(ascending=False)
        feat_imp.plot(title='feature importance')
        plt.ylabel('feature importance score')
        plt.show()

    def get_params(self, deep=True):
        return self.params

    def set_params(self, **kwargs):
        self.params.update(kwargs)
        return self


# 基准模型
bst = XGBoostRegressor(eta=0.1, colsample_bytree=0.5, subsample=0.5,
                       max_depth=5, min_child_weight=3, num_boost_round=50)
base = bst.kfold(train_x, train_y, nfold=5)

# step2:树的深度和节点权重
# 这些参数对xgboost性能的影响最大，因此应当第一个调整
# max_depth:树的最大深度，增加这个值会让模型更复杂，也容易出现过拟合，深度3-10是合理的
# min_child_weight:正则化参数，如果树分区中实例权重小于定义的总和，则停止树的构建
'''
xgb_param_grid = {'max_depth': list(range(4, 6)), 'min_child_weight': list((1, 3, 6))}
'''

def mae_score(y_true, y_pred):
    return mean_absolute_error(np.exp(y_true), np.exp(y_pred))


mae_scorer = make_scorer(mae_score, greater_is_better=False)

'''
import time

t1 = time.time()
grid = GridSearchCV(XGBoostRegressor(eta=0.1, num_boost_round=50, colsample_bytree=0.5, subsample=0.5),
                    param_grid=xgb_param_grid, cv=3, scoring=mae_scorer)
grid.fit(train_x, train_y.values)
t2 = time.time()
grid_search_time = t2 - t1

# 交叉验证结果
means = grid.cv_results_['mean_test_score']
params = grid.cv_results_['params']
best_param = grid.best_params_
# 这个值得到的是带负号的最大值，也就是去掉负号的最小值
best_score = grid.best_score_
'''

# step3:调节gamma去降低过拟合风险
'''
xgb_param_grid={'gamma':[0.1*i for i in range(5,7)]}

grid=GridSearchCV(XGBoostRegressor(eta=0.1, num_boost_round=50, colsample_bytree=0.5, subsample=0.5,max_depth=8,min_child_weight=6),
                  param_grid=xgb_param_grid,cv=3,scoring=mae_scorer)
grid.fit(train_x,train_y.values)

# 交叉验证结果
means = grid.cv_results_['mean_test_score']
params = grid.cv_results_['params']
best_param = grid.best_params_
# 这个值得到的是带负号的最大值，也就是去掉负号的最小值
best_score = grid.best_score_
# 结论：对于这个数据集来说gamma值对于结果的影响很小
'''

# step4：调节采样方式subsample和colsample_bytree
''''
xgb_param_grid={'subsample':[0.1*i for i in range(6,8)],'colsample_bytree':[0.1*i for i in range(6,7)]}

grid=GridSearchCV(XGBoostRegressor(eta=0.1,gamma=0.2,num_boost_round=50,max_depth=8,min_child_weight=6),
                  param_grid=xgb_param_grid,cv=3,scoring=mae_scorer)

grid.fit(train_x,train_y.values)

# 交叉验证结果
means = grid.cv_results_['mean_test_score']
params = grid.cv_results_['params']
best_param = grid.best_params_
# 这个值得到的是带负号的最大值，也就是去掉负号的最小值
best_score = grid.best_score_
# 结论：对于该数据集这组参数的作用不大
'''

# step5:减小学习率增大树的个数
# 学习率设置小一些，树设置多一些，过拟合风险会低一些
xgb_param_grid={'eta':[0.5,0.3,0.1,0.05,0.01]}

grid=GridSearchCV(XGBoostRegressor(eta=0.1,gamma=0.2,num_boost_round=50,max_depth=8,min_child_weight=6,subsample=0.9,colsample_bytree=0.6),
                  param_grid=xgb_param_grid,cv=3,scoring=mae_scorer)
grid.fit(train_x,train_y.values)

# 交叉验证结果
means = grid.cv_results_['mean_test_score']
params = grid.cv_results_['params']
best_param = grid.best_params_
# 这个值得到的是带负号的最大值，也就是去掉负号的最小值
best_score = grid.best_score_