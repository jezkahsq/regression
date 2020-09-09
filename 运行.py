# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 16:55:43 2020

@author: Administrator
"""


import pandas as pd
from sklearn.model_selection import train_test_split
import sys
sys.path.append('C:\\Users\\Administrator\\Desktop\\7.21\\MODEL')
from sklearn.externals import joblib
from sklearn.preprocessing import PolynomialFeatures
import numpy
from MODEL.preprocess import Pre_process
from MODEL.eval import ModelEvaluation
from MODEL.model import RegressionModel

#数据预处理
datafile1 = u'C:\\Users\\Administrator\\Desktop\\7.21\\1\\1chiller6test.csv'
data = pd.read_csv(datafile1)
data = data.iloc[:,2:12]
My_process = Pre_process(data)
My_process.WindowMean()
#My_process.Describe()

#建模
model_names = ['LR', 'Ridge', 'ElasticNet', 'BayesianRidge', 'SVR', 'SGDRegressor'] 
#model_names = ()
model_num = 6
testsize = 0.2
X = My_process.data_result.iloc[:, 1:10]
y = My_process.data_result.iloc[:, -1]
My_model = RegressionModel(X, y, model_names, testsize)
My_model.Features()
#My_model.Svr()
My_model.Models(model_names)

#原始数据特征扩充、模型保存
polynomy = PolynomialFeatures(degree=2)#多项式特征扩充
poly_features = polynomy.fit_transform(X)
dt_ = pd.DataFrame()
for i in range(6):
    model = joblib.load( f'C:\\Users\\Administrator\\Desktop\\7.21\\models-saved\\1期 device6 模型{model_names[i]}.pkl')
    dt_[i] = model.predict(poly_features)

#模型实现与评估
y_test = y
y_predict1 = dt_.iloc[:,0:6]
y_predict = y_predict1.T.values
sort_num = 20#y_test划分的区间数
My_model = ModelEvaluation(y_test, y_predict, model_names, model_num, sort_num, sort_bin=40)
My_model.EvaluationIndex()
My_model.EvaluationPlot()
My_model.RelativeError()
My_model.RelativeErrorSorted()