# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 18:20:58 2020

@author: 29434
"""


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as sci
import pandas as pd

class Pre_process:
    
    def __init__(self, data):
        self.data = data

    def Normalization(self):
        data_fea = self.data.iloc[:, :] #取数据中指标所在的列
        self.data_normalized = data_fea.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))#归一化
        self.data_normalized.to_csv('C:\\Users\\Administrator\\Desktop\\lj\\归一化.csv')
        
    def DataAnalysis(self):
        self.cor = self.data.corr() #全部特征计算
        names = list(self.data.columns)
        indice=1#子图索引
        sns.heatmap(self.cor, cmap='Reds', center=None, linewidths=0.05, vmax=1, vmin=0, annot=True, annot_kws={'size':6, 'weight':'bold'})
        plt.xticks(np.arange(len(names))+0.5, names) #横坐标标注点
        plt.yticks(np.arange(len(names))+0.5, names)
        plt.title('Characteristic correlation')
        plt.show()
        
        with sns.axes_style('white'):
            cols = self.data.columns
            for i in cols:
                for j in cols:
                    if i<j:
                        sns.jointplot(x=self.data[i], y=self.data[j], data = self.data, kind = 'hex',
                                  color = 'k',stat_func=sci.pearsonr,marginal_kws = dict(bins = 20))
                        indice+=1
                        plt.savefig('C:\\Users\\Administrator\\Desktop\\lj\\figures\\'+'频次直方图_'+str(indice), dpi=300)

    def Describe(self):
        print('数据示例:', '\n', self.data.head(5), '\n', self.data.tail(5), '\n', '-'*60)
        print('描述性统计分析:', '\n', self.data.describe(), '\n', '-'*60)
        
        #频次直方图
        cols = self.data.columns
        indice=1#子图索引
        for i in cols:
            plt.subplot(2, 3, indice)#绘制子图
            plt.subplots_adjust(wspace =0.3, hspace =0.4)
            #可改为hist
            #plt.figure(i)#重置画布
            sns.distplot(self.data[i], color=sns.desaturate("indianred", .8), bins=40)
            indice+=1
            plt.axvline(x=self.data[i].mean(), ls=":", c="green")
            plt.axvline(x=self.data[i].median(), ls=":", c="blue")


    def Plot(self, plot_path):
        #箱线图
        plt.figure()
        sns.boxplot(data=self.data.iloc[:, 1:], fliersize=2, flierprops={'color':'purple', 'markeredgecolor':"purple"})
        #pd.set_option('display.max_columns',10)
        if plot_path:
            try:
                plt.savefig(plot_path + '箱线图', dpi=300)
            except Exception:
                print("输入的路径有错")
        else:
            plt.savefig('.\\figures\\' +'箱线图', dpi=300)

    
    #异常值处理
    #四分位法
    def OutlierQuantile(self):
        cols = self.data.columns
        for i in cols:
            a = self.data[i].quantile(0.75)
            b = self.data[i].quantile(0.25)
            c = self.data[i]
            c[(c>=(a-b)*1.5+a)|(c<=b-(a-b)*1.5)]=np.nan
            self.data.dropna()
            print('四分位法有效数据量：', len(self.data))
            self.lower_quartile = b-(a-b)*1.5
            self.upper_quartile = (a-b)*1.5+a
            print('四分位法阈值:', self.lower_quartile, '~', self.upper_quartile, '\n', '-'*60)
            return self.data #删除后的有效数据
    
    #四倍标准差法
    def OutlierStd(self):
        cols = self.data.columns
        for i in cols:
            self.lower_std = self.data[i].mean()+self.data[i].std()*4
            self.upper_std = self.data[i].mean()-self.data[i].std()*4
            qua_std = self.data[i]
            qua_std[(qua_std>=self.lower_std)|(qua_std<=self.upper_std)]=np.nan
            self.data.dropna()
            print('四倍标准差法有效数据量：', len(self.data))
            print('四倍标准差法阈值:', self.lower_std, '~', self.upper_std, '\n', '-'*60)
        self.data.to_csv(u'C:\\Users\\Administrator\\Desktop\\1\\四倍标准差筛选.csv')
        
    #滑动窗口法
    def WindowMean(self, X=10, thre_list = [0.02, 0.02, 0.02, 0.02, 0.02, 0.02]):
        window_mean = self.data.rolling(window = X).mean()
        a = abs((self.data - window_mean)/self.data)
        self.data_window=self.data.copy()
        self.data_window.loc[:, :]=True
        for i,col in enumerate(thre_list):
            if col>0:
                self.data_window.iloc[:, i] = (a.iloc[:, i]<thre_list[i])
        self.data_result=self.data[self.data_window]
        self.data_result=self.data_result.dropna()
        self.data_result.to_csv(u'C:\\Users\\Administrator\\Desktop\\7.21\\滑动窗口筛选.csv')
        '''
        b = abs((self.data - window_mean)/self.data_5)
        self.data_5_window = self.data_5
        self.data_5_window.loc[:,:]=True
        for i,col in enumerate(thre_list):
            if col>0:
                self.data_5_window.iloc[:,i] = (b.iloc[:,i]<thre_list[i])
        self.data_5_result=self.data_5[self.data_5_window]
        self.data_5_result=self.data_5_result.dropna()
        self.data_5_result.to_csv(u'C:\\Users\\Administrator\\Desktop\\7.21\\按5号滑动窗口筛选.csv')
        '''
        
    def WindowStd(self, X=10, thre_list = [0.2, 0, 2, 0, 0, 0, 0]):
        window_std = self.data.rolling(window = X).std()
        self.data_window=self.data.copy()
        self.data_window.loc[:, :]=True
        for i, col in enumerate(thre_list):
            if col>0:
                self.data_window.iloc[:, i] = (window_std.iloc[:, i]<thre_list[i])
        self.data_result=self.data[self.data_window]
        self.data_result=self.data_result.dropna()
        print('滑动窗口标准差法：', '\n', self.data_result, '\n', '-'*60)
        return self.data_result