# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 14:26:00 2020

@author: 29434
"""


import numpy as np
from sklearn import linear_model
from sklearn import svm
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn import model_selection
from sklearn.externals import joblib
#from sklearn.pipeline import Pipeline
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import SGDRegressor

class RegressionModel:

    def __init__(self, X, y, model_names, testsize):
        self.X = X
        self.y = y
        self.model_names = model_names
        self.testsize = testsize
        
    def Features(self):
        self.log_features = np.log(self.X)#对数特征转换
        polynomy = PolynomialFeatures(degree=2)#多项式特征扩充
        self.poly_features = polynomy.fit_transform(self.X)
        return self.poly_features, self.log_features
        
    def Svr(self):
        #寻找最优参数c
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(self.poly_features, self.y, test_size=self.testsize)
        C_range = np.linspace(0.1, 10, 10)
        max_c = []
        #max_perc = 0.8
        predicts = []
        for c in C_range:
            svr = svm.LinearSVR(epsilon=3, C=c).fit(self.X_train, self.y_train)
            perc_within_eps = 100*np.sum(self.y_val - svr.predict(self.X_val) < 1) / len(self.y_val)
            #max_perc = max(max_perc,perc_within_eps)
            predicts.append(perc_within_eps)
            max_c.append(c)
        self.max_C = max_c.index(max(predicts))
        plt.figure()
        plt.plot(C_range, predicts1, c="blue")
        plt.xlabel('C')
        plt.ylabel('predict')
        plt.show()
        return self.C
        '''
        #寻找最优参数gamma
        gamma_range = np.linspace(0.001,10,1000)
        max_perc = 0.8
        predicts2 = []
        for gamma in gamma_range:
            svr = SVR(kernel='rbf',epsilon=3,gamma=gamma,C=1).fit(self.X_train,self.y_train)
            perc_within_eps = 100*np.sum(self.y_val - svr.predict(self.X_val) <1) / len(self.y_val)
            max_perc = max(max_perc,perc_within_eps)
            predicts2.append(perc_within_eps)
        plt.figure()
        plt.plot(gamma_range,predicts2,c="red")
        plt.xlabel('gamma')
        plt.ylabel('predict')
        plt.show()
        '''       
        
    def Models(self, model_names):
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(self.poly_features, self.y, test_size=0.3)
        
        #model_lasso = linear_model.LassoCV(alphas=[0.0001,0.001,0.01,0.05,0.1,0.2,0.5,1,10],cv=10)
        model_linear = linear_model.LinearRegression()
        model_ridge = linear_model.RidgeCV(alphas=[0.0001, 0.001, 0.01, 0.05, 0.1, 0.2, 0.5, 1, 10], cv=10)
        model_en = linear_model.ElasticNetCV(alphas=[0.0001, 0.0005, 0.001, 0.01, 0.1, 1, 10], l1_ratio=[.01, .1, .5, .9, .99],  max_iter=5000)
        model_br = linear_model.BayesianRidge(alpha_1=1e-06, alpha_2=1e-06, compute_score=False, copy_X=True,
                                              fit_intercept=True, lambda_1=1e-06, lambda_2=1e-06, n_iter=300,
                                              normalize=False, tol=0.001, verbose=False)                                                
        #model_svr = model_selection.GridSearchCV(LinearSVR(random_state=0, tol=1e-5),  
        #                                         param_grid={"epsilon":[0,0.2],"C": [0,1]},cv = 5)
        model_svr = make_pipeline(StandardScaler(), svm.LinearSVR(random_state=0, tol=1e-5))
        model_sgdr = make_pipeline(StandardScaler(), SGDRegressor(max_iter=1000, tol=1e-3))
        model_list = [model_linear, model_ridge, model_en, model_br, model_svr, model_sgdr]
        #print('岭回归最优alpha值:', model_ridge.alpha_, '\n', '-'*50)

        for i in range(6):
            model_predict = model_list[i].fit(self.X_train, self.y_train)
            self.pre_y_list = model_predict.predict(self.X_val)  
            joblib.dump(model_predict, f'C:\\Users\\Administrator\\Desktop\\7.21\\models-saved\\1期 device6 模型{model_names[i]}.pkl')
        
        #self.pre_y = [model.predict(self.X) for model in model_list]
        #print('线性回归权重值:','\n', model_linear.coef_, '\n', '截距:', model_linear.intercept_, '\n', '-'*50)
        #print('岭回归权重值:', '\n', model_ridge.coef_, '\n', '-'*50)
        #print('贝叶斯岭回归权重值:', '\n', model_br.coef_, '\n', '-'*50)
        #self.n_samples, self.n_features = self.X.shape  
        #print('all samples: %d \t features: %d' % (self.n_samples, self.n_features))
    

