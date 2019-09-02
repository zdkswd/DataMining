import pandas as pd

df1 = pd.read_csv('../data/JData_Action_201602.csv')
df10 = df1[:10000]
df10.to_csv('../data/JData_Action_201602_sample.csv', index=False)
df2 = pd.read_csv('../data/JData_Action_201603.csv')
df20 = df2[:10000]
df20.to_csv('../data/JData_Action_201603_sample.csv', index=False)
df3 = pd.read_csv('../data/JData_Action_201604.csv')
df30 = df3[:10000]
df30.to_csv('../data/JData_Action_201604_sample.csv', index=False)
