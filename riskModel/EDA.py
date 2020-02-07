# -*- coding: utf-8 -*-
# @Time    : 2018-01-21 11:22
# @Author  : HuangSir
# @FileName: EDA.py
# @Software: PyCharm

"""
Exploratory analysis
"""

import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import missingno as msno
import pandas_profiling as pandas_pf
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['backend'] = 'Agg'
from scipy.stats import norm
from .utils.InputCondition import check_df,check_str,check_str_list

class BasicStat(object):
    """数据概述"""
    def __init__(self,df:pd.DataFrame):
        """
        基本指标统计类
        :param df:dfframe 数据集
        """
        df = check_df(df)
        self._df = df
        self.dtypes = df.dtypes.to_dict()

    def feature_describe(self,x=None,percentiles=None,include=None):
        """
        字段基本信息
        :param x: str or list 统计变量, 默认None
        :param percentiles: list-like of float 默认 None,[0.25,0.5,0.75]
        :param include: 'all', list-like of dtypes or None (default)
        :return:feature_describe 字段基本信息
        """
        # x = check_str_list(x)
        if x:
            feature_describe = self._df[x].describe(percentiles=percentiles,include=include).T
        else:
            feature_describe = self._df.describe(percentiles=percentiles,include=include).T
        return feature_describe


    def df_report(self,fname):
        """
        数据报告
        :param fname: str 保存文件名
        :return:数据报告 html格式
        """
        fname = check_str(fname)
        warnings.warn('there are will cost a lot of time')
        profile = pandas_pf.ProfileReport(self._df)
        profile.to_file(output_file=fname)
        print("report done")

class Plot(object):
    """
    作图
    """
    def __init__(self,df:pd.DataFrame):
        """公共属性"""
        df = check_df(df)
        self._df = df

    def draw_pie(self,var:str,fname:str):
        """
        字符型变量饼图
        -------------------------------------
        Params
        s: pandas Series
        lalels:labels of each unique value in s
        dropna:bool obj
        fname: 保存图路径及文件名
        -------------------------------------
        Return
        show the plt object
        """
        var = check_str(var)
        fname = check_str(fname)
        counts = self._df[var].value_counts(dropna=True)
        labels = counts.index
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.pie(counts, labels=labels, autopct='%1.2f%%', shadow=True, startangle=90)
        ax.axis('equal')
        ax.set_title(r'pie of {}'.format(var))
        plt.savefig(fname)
        plt.close()

    def draw_bar(self,var:str, fname:str, pct=False, horizontal=False):
        """
        字符型变量条形图
        -------------------------------------------
        Params
        s: pandas Series
        x_ticks: list, ticks in X axis
        pct: bool, True means trans df to odds
        dropna: bool obj,True means drop nan
        horizontal: bool, True means draw horizontal plot
        -------------------------------------------
        Return
        show the plt object
        """
        var = check_str(var)
        fname = check_str(fname)

        counts = self._df[var].value_counts(dropna=False)
        if pct:
            counts = counts / self._df[var].shape[0]
        ind = np.arange(counts.shape[0])
        plt.figure(figsize=(8, 6))

        if not horizontal:
            plt.bar(ind, counts)
            plt.ylabel('frequecy')
            plt.xticks(ind, tuple(counts.index))
        else:
            plt.barh(ind, counts)
            plt.xlabel('frequecy')
            plt.yticks(ind, tuple(counts.index))
        plt.title('Bar plot for {}'.format(var))
        plt.savefig(fname)
        plt.close()

    def draw_histogram(self,var:str,fname:str,num_bins:int=20):
        """
        连续变量分布图
        ---------------------------------------------
        Params
        s: pandas series
        num_bins: number of bins
        fname png name
        ---------------------------------------------
        Return
        show the plt object
        """
        var = check_str(var)
        fname = check_str(fname)

        fig, ax = plt.subplots(figsize=(14, 7))
        mu = self._df[var].mean()
        sigma = self._df[var].std()

        n, bins, patches = ax.hist(self._df[var], num_bins, normed=1, rwidth=0.95, facecolor="blue")

        y = norm.pdf(bins, mu, sigma)
        ax.plot(bins, y, 'r--')
        ax.set_xlabel(var)
        ax.set_ylabel('Probability density')
        ax.set_title(r'Histogram of %s: $\mu=%.2f$, $\sigma=%.2f$' % (var, mu, sigma))
        plt.savefig(fname)
        plt.close()

    def plot_miss(self,fname:str,asc=0,figsize=(10,6)):
        """
        缺失可视化
        :param df:df
        :param fname:str 路径及文件名
        :param asc: int 统计方法,Matrix(asc=0),BarChart(asc=1),Heatmap(asc=2)
        :param figsize tupe 图片大小
        :return:保存结果
        """
        filename = check_str(fname)
        if asc == 0:
            msno.matrix(df=self._df)
        elif asc == 1:
            msno.bar(df=self._df, figsize=figsize)
        else:
            msno.heatmap(df=self._df, figsize=figsize)
        plt.savefig(filename)

    def plot_scatter(self,var1:str,var2:str,fname:str,x1label=None,x2label=None):
        """
        散点图
        :param df:dataframe
        :param fname str 路径及文件名
        :param x1: str x1变量
        :param x2: str x2 变量
        :param x1label: x1 str 标签
        :param x2label: x2 str 标签
        :return:保存图
        """
        var1 = check_str(var1)
        var2 = check_str(var2)
        fname = check_str(fname)

        plt.scatter(x=self._df[var1].values,y=self._df[var2].values,s=20,c='g')
        plt.xlabel(x1label)
        plt.ylabel(x2label)
        plt.savefig(fname)
        plt.close()

    def mult_boxplot(self,variable:str,category:str,fname:str,xlabel=None, ylabel=None, title=None):
        """
        箱线图,探索数据分布
        :param df: df
        :param variable: str 统计变量
        :param category: str 分组值
        :param fname: str 路径及文件名
        :param xlabel: str
        :param ylabel: str
        :param title: str
        :return: 保存图
        """
        variable = check_str(variable)
        category = check_str(category)
        fname = check_str(fname)

        self._df[[variable, category]].boxplot(by=category) #作图

        if xlabel:
            plt.xlabel(xlabel)
        if ylabel:
            plt.ylabel(ylabel)
        if title:
            plt.title(title)

        plt.savefig(fname)
        plt.close()