# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 18:38:28 2020

@author: Admin
"""

import pandas as pd
import numpy as np
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as sci
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] =False


class ModelEvaluation:

    def __init__(self, y_test, y_predict, model_names, model_num, sort_num, sort_bin=10):
        self.y_test = y_test
        self.y_predict = y_predict
        self.model_names = model_names
        self.model_num = model_num
        self.sort_num = sort_num
        self.sort_bin = sort_bin
        
    def EvaluationIndex(self):
        if not self.model_names:
            self.model_names = list()
            for i in range(self.model_num):
                self.model_names.append(f'{i}')                                                               
        model_metrics_functions = [explained_variance_score, mean_absolute_error, mean_squared_error, r2_score]
        model_metrics_list = [[m(self.y_test, self.y_predict[i]) for m in model_metrics_functions] for i in range(len(self.y_predict))] #回归评估指标列表
        self.regression_score = pd.DataFrame(model_metrics_list, index=self.model_names, columns=['explained_variance', 'mae', 'mse', 'r2']) #建立回归指标的数据框
#        print('all samples: %d \t features: %d' % (self.n_samples, self.n_features), '\n', '-'*60)
        print('\n', 'regression metrics:', '\n', '-'*60)
        print(self.regression_score)

    def EvaluationPlot(self):
        plt.figure(figsize=(10, 10))
        if not self.model_names:
            self.model_names = list()
            for i in range(self.model_num):
                self.model_names.append(f'{i}')
        for i, pre_y in enumerate(self.y_predict):
            plt.subplot(len(self.y_predict)+1, 1, i+1)
            plt.plot(np.arange(len(self.y_test)), self.y_test, color='k', label='true y')
            plt.plot(np.arange(len(pre_y)), pre_y, 'g--', label=self.model_names[i])
            plt.title('True and {} result comparison'.format(self.model_names[i]))
            plt.legend(loc='upper right')
            plt.tight_layout()
        #plt.savefig('.\\figures\\LineComparison'+str(self.model_names[i])+'.png', dpi=300)
        plt.show()
           
        #每种模型预测值与真实值的散点密度图
        for i, pre_y in enumerate(self.y_predict):
            sns.jointplot(x=self.y_predict[i], y=self.y_test, kind = 'hex', color = 'k', stat_func=sci.pearsonr,
                          marginal_kws = dict(bins = 20))
            plt.title('True and {} result'.format(self.model_names[i]))
            #plt.savefig('.\\figures\\ScatterHex'+str(self.model_names[i])+'.png', dpi=300)
        
        for i, pre_y in enumerate(self.y_predict):
            plt.plot([1, 2, 3, 4, 5, 6])
            plt.subplot(3,2,i+1)
            plt.scatter(x=self.y_predict[i], y=self.y_test,s=16.,color='teal',alpha=0.4)
            plt.title('True and {} result'.format(self.model_names[i]))
            x = self.y_predict[i]
            y = x
            plt.plot(x, y, color='red', linewidth=1)
            #plt.savefig('.\\figures\\Scatter'+str(self.model_names[i])+'.png', dpi=300)
        plt.show()
     
     
    #相对误差频次分布直方图
    def RelativeError(self):
        relative_error = pd.DataFrame()
        if not self.model_names:
            self.model_names = list()
            for i in range(self.model_num):
                self.model_names.append(f'{i}')
        for i, pre_y in enumerate(self.y_predict):
            relative_error[i] = (self.y_predict[i] - self.y_test)/self.y_test
            fig, ax = plt.subplots()
            N,bins,patches = ax.hist(relative_error[i], bins = 100)
            #plt.xlim(-1,1)
            for j in range(25):
                patches[j].set_facecolor('#EEA2AD')
            for j in range(25, 40):
                patches[j].set_facecolor('#6CA6CD')
            for j in range(40, len(patches)):
                patches[j].set_facecolor('#FFA07A')
            plt.title('{}'.format(self.model_names[i]))
            #plt.savefig('.\\figures\\RelativeErrorDist'+str(self.model_names[i])+'.png', dpi=300)
            
    def RelativeErrorSorted(self):
        relative_error = pd.DataFrame()
        if not self.model_names:
            self.model_names = list()
            for i in range(self.model_num):
                self.model_names.append(f'{i}')
        a=self.y_test.max()
        b=self.y_test.min()
        abrange=np.linspace(b, a, self.sort_num+1)
        
        for i, pre_y in enumerate(self.y_predict):
            relative_error[i] = (self.y_predict[i]- self.y_test)/self.y_test
            plt.plot()
            for j in range(0, self.sort_num):
                plt.subplot(5, 4, j+1)
                plt.hist(relative_error[i][(self.y_test>=abrange[j]) & (self.y_test<=abrange[j+1])], bins=self.sort_bin, alpha=0.5, histtype='stepfilled',color='steelblue')
                #plt.xlim(-0.5, 0.5)
                down = abrange[j]
                up = abrange[j+1]
                #plt.title(f'{down}到{up}区间{self.model_names[i]}相对误差图')
                # plt.savefig('.\\figures\\RelativeErrorDist'+str(self.model_names[i])+'.png', dpi=300)
                plt.tick_params(labelsize=9)

            plt.show()
        # for j in range(0,self.sort_num):
        #     relative_error = pd.DataFrame()
        #     if not self.model_names:
        #         self.model_names = list()
        #         for i in range(self.model_num):
        #             self.model_names.append(f'{i}')
        #     for i, pre_y in enumerate(self.y_predict_sorted[j]):
        #         plt.figure()
        #         relative_error[i] = (self.y_predict_sorted[j][i]- self.y_test_sorted[j])/self.y_test_sorted[j]
        #         plt.hist(relative_error[i], bins=10,alpha=0.5,histtype='stepfilled',color='steelblue')
        #         plt.title(f'第{j}区间相对误差图' + '{}'.format(self.model_names[i]))
        #         plt.show()
    
    #单变量分析
    def PredictionAnalysis(self, model, X_test):
        for i, pre_y in enumerate(self.y_predict):
            print(f'{self.model_names[i]}', '预测值数据示例:', pd.DataFrame(self.y_predict[i]).head(5),'\n', '-'*60)
            print(f'{self.model_names[i]}', '预测值描述性统计分析:', pd.DataFrame(self.y_predict[i]).describe(), '\n', '-'*60)