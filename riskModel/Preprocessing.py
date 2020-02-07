# -*- coding: utf-8 -*-
# @Time    : 2018-01-21 11:22
# @Author  : HuangSir
# @FileName: Preprocessing.py
# @Software: PyCharm

"""
1. 变量类型划分--异常值检测--异常值处理--重复行(列)处理--众数处理--缺失处理
2. 样本采样: 随机采样, 随机平衡采样, Borderline-SMOTE过采样, Easy Ensemble 欠采样.
"""
import warnings
import pandas as pd
import numpy as np
import numbers
from sklearn.utils import shuffle
from .utils.tools import str2num
import re
from imblearn.over_sampling import BorderlineSMOTE,ADASYN,KMeansSMOTE,RandomOverSampler,SVMSMOTE
from imblearn.ensemble import BalanceCascade,EasyEnsemble
from collections import Counter

class Preprocess(object):
    """
    数据预处理
    """

    def split_col(self,df,no_split,min_size=5):
        """
        变量类型划分, 无序类别变量, 有序类别变量,有序离散数值变量,连续数值变量.
        :param no_split: list 不参与分类变量
        :param min_size: int 连续变量最小类别数,默认5
        :return: num_col and cat_col list 分类结果
        """
        num_col,cat_col = [],[]
        for i in [i for i in df.columns.tolist() if i not in no_split]:
            unique_list = df[i].unique().tolist()
            if len(unique_list) > min_size and isinstance(str2num(unique_list[0]), numbers.Real):
                num_col.append(i)
            else:
                cat_col.append(i)
        return num_col,cat_col

    def find_str(self, df,num_col):
        """
        异常值检测之特殊字符串
        :param num_col:list 需要检测的变量集
        :return:str_set 特殊值序列
        """
        str_set = set()
        for var in num_col:
            for i in df[var].unique():
                try:
                    if i is np.nan or isinstance(str2num(i),numbers.Real):
                        pass
                except:
                    str_set.add(i)
        return str_set

    def special_char(self,df,feature_col):
        """
        异常值检测之识别特殊符号
        :param df: df
        :param feature_col: list 特征列表集
        :return: special_set
        """
        special_set = set()
        for col in feature_col:
            for s in df[col].unique():
                try:
                    special = r"\$|\(|\)|\*|\+|\[|\]|\?\\|\^|\||\@|\&|\{|\}|u"
                    temp_set = set(re.findall(special,str(s)))
                    special_set.add(temp_set)
                except:
                    pass
        return special_set


    def outlier(self,df, col, low_percent=0.05, up_percent=0.95, cat_percent=0.001, special_value=None):
        """
        异常值处理之删除
        :param df:dataframe 数据集
        :param col:str 处理的数值变量 或分类变量
        :param low_percent:下限分位数
        :param up_percent:上限分位数
        :param cat_percnet:分类变量单个字段占比
        :param special_value: list 特殊值,不参与异常处理
        :return: dataframe 删除异常值后的数据
        """
        if  col in df.select_dtypes(include=["object", "bool"]).columns.tolist():  # 字符型变量
            cat_value_percent = df[col].value_counts() / len(df[col])  # 统计分类变量个数占比
            del_values = cat_value_percent[cat_value_percent < cat_percent].index.tolist()  # 删除值
            if [i for i in del_values if i not in special_value]:
                for i in del_values:
                    print('分类变量{},删除异常值:{}'.format(col,i))
                    df = df.loc[df[col] != i, :]  # 删除异常值
        else:
            Q1 = df[col].quantile(low_percent)  # 下分为点
            Q2 = df[col].quantile(up_percent)  # 上分为点
            # 异常值替换
            print('连续变量:',col,'low:',Q1,'up:',Q2)
            df[col] = df[col].apply(lambda x:Q1 if x < Q1 else Q2 if x > Q2 else x)
        return df

    def drop_dupl(self,df,axis=0):
        """
        删除重复行或重复列
        :param axis: 0 or 1 0表示删除重复行,1表示删除重复列
        :return: 删除结果
        """
        if axis == 0:
            # 删除重复行
            df_new = df.drop_duplicates()
        else: # 删除重复列
            df_new = df.T.drop_duplicates().T
        return df_new

    def cnt_miss_mode(self,df,col_list,mode=None):
        """
        缺失处理-缺失率和众数率统计
        :param col_list:统计字段
        :param model: 返回结果, 默认 None,option miss,mode
        :return: dict 缺失率和众数率字典
        """
        miss_mode = {}
        miss_mode["miss_rate"],miss_mode["mode_rate"] = {},{}

        for i in col_list:
            miss_rate = df[i].isnull().sum() / len(df)  # 缺失率
            miss_mode["miss_rate"][i] = miss_rate
            try:
                mode_values = df[i].mode()[0]  # 众数
            except:
                mode_values = None

            mode_rate = len(df.loc[df[i] == mode_values, :]) / len(df)  # 众数率
            miss_mode["mode_rate"][i] = mode_rate
        # 输出模式
        if mode=="miss":
            return miss_mode["miss_rate"]
        elif mode =="mode":
            return miss_mode["mode_rate"]
        else:
            return miss_mode


    def drop_nan_mode(self,df,nan_percent,mode_percent,col_list,drop=False):
        """
        缺失处理-删除缺失率和众数率较大的变量,如果只删除其中一种,只需将另一种设置为1即可
        :param nan_percent: float 0~1 缺失率
        :param mode_percent: float 0~1 众数率
        :param col_list: list 作用变量列表
        :param drop:bool 是否删除
        :return: DataFrame
        """
        del_col = set()
        miss_model_rate = self.cnt_miss_mode(df=df,col_list=col_list)
        for i in col_list:
            miss_rate = miss_model_rate["miss_rate"][i] # 缺失率
            mode_rate = miss_model_rate["mode_rate"][i] # 众数率
            if miss_rate >= nan_percent:
                del_col.add(i)
            elif mode_rate >= mode_percent:
                del_col.add(i)

        # 是否删除操作
        if drop:
            for i in del_col:
                try:
                    del df[i]
                except:
                    print("变量:{}已删除".format(i))

            return df, del_col
        else:
            return del_col

    def fill_nan(self,df,value = None,method=None,col_list=None,min_miss_rate=0.05):
        """
        缺失处理-缺失值填充
        :param value:dict 列填充值字典 默认 None
        :param method:str,方法 backfill, bfill, pad, ffill, None, mode,mean,special_value,采用均值填充的字段必须为连续变量
        :param col_list:填充列,特殊方式填充
        :param min_miss_rate,最小缺失率
        :return: dataframe
        """
        if method in ["backfill", "bfill", "pad", "ffill"]:
            df = df.fillnan(method=method)

        elif method=="special_value": # 特殊值填充或列字典填充
            if value:
                df = df.fillnan(value=value)
            else:
                df = df.fillnan(value='-99')

        elif method in ["mode","mean"] and len(col_list)>0: # 特殊填充方法
            miss_model_rate = self.cnt_miss_mode(df=df,col_list=col_list)
            # 特殊填充方法
            for i in col_list:
                try:
                    mode = df[i].mode()[0]
                except:
                    mode = value

                if value:
                    if miss_model_rate["miss_rate"][i] > min_miss_rate:
                        df[i] = df[i].fillna(value=value)
                    elif method =="mode":
                        df[i] = df[i].fillna(value=mode)
                    elif method =="mean":
                        try:
                            df[i] = df[i].fillna(value=df[i].mean())
                        except TypeError:
                            warnings.warn('分类变量:{} 均值填充失败,使用众数填充')
                            df[i] = df[i].fillna(value=mode)
                elif not value:
                    if method == "mode":
                        df[i] = df[i].fillna(value=mode)
                    elif method =="mean":
                        try:
                            df[i] = df[i].fillna(value=df[i].mean())
                        except TypeError:
                            warnings.warn('分类变量:{} 均值填充失败,使用众数填充')
                            df[i] = df[i].fillna(value=mode)
        else:
            df = df.fillnan('-99') # 任何都不填的情况下
        return  df


    def factor_map(self,df,col_list=None):
        """
        字符变量数值化
        :param col_list:需要数值化字段
        :return:df, factor_dict 数值化映射
        """
        factorize_dict = {}
        if not col_list:
            col_list = df.select_dtypes(include=["object","bool"]).columns.tolist()
        for var in col_list:
            factorize_dict[var] = {}
            for i in np.arange(len(pd.factorize(df[var])[1])):
                factorize_dict[var][pd.factorize(df[var])[1][i]] = i  # 数值化映射字典
            df[var] = pd.factorize(df[var])[0]  # 数值化
        return df,factorize_dict # 数值数据, 映射字典



class DataSample(object):
    """
    数据抽样
    """
    def __init__(self,df,target='target'):
        self._df = df
        self._target = target

    def random_sample(self,n=None,fra=None):
        """
        随机抽样
        :param n:抽样样本量
        :param fra: float,抽样样本占比
        :return:dataframe 抽样结果
        """
        if n:
            new_df = self._df.sample(n = n,replace=True)
        elif fra:
            new_df = self._df.sample(frac=fra)
        else:
            new_df = self._df
        return new_df

    def balance_sample(self,odd,c=None):
        """
        根据好坏比抽样
        :param odd:int,float 好坏比
        :param label:str 标签
        :param c: option g, 默认对坏样本抽样
        :return: df 抽样后新数据集
        """
        bad_df = self._df.loc[self._df[self._target]==1,:]  # 坏样本
        good_df = self._df.loc[self._df[self._target]==0,:]  # 好样本
        bad = len(bad_df) # 坏样本数
        good = len(good_df)  # 好样本数
        if c == 'g':
            good_new = int(bad*odd) # 所需好样本数
            if good_new > good:
                good_df_new = good_df.sample(n=good_new,replace=True)
            else:
                good_df_new = good_df.sample(n = good_new,replace=False)

            good_df = good_df_new
        else:
            bad_new = int(good/odd) # 所需坏样本数
            if bad_new > bad:
                bad_df_new = bad_df.sample(n=bad_new,replace=True)
            else:
                bad_df_new = bad_df.sample(n=bad_new,replace=False)

            bad_df = bad_df_new

        df = pd.concat([bad_df,good_df],axis=0) # 合并新样本
        df = shuffle(df) # 打乱顺序
        return df

    def over_sample(self,features,method="BorderLine",sampling_strategy="minority",random_state=42,
                    k_neighbors=5,n_neighbors=10,kind="borderline-1"):
        """
        过采样方法
        : param features: list 特征集
        :param method: str, option: ADASYN, BorderLine,Random,SVM
        :param sampling_strategy:str or dict, option: 'minority','not majority','all','auto', {1:n,0:m}
        :param random_state:int
        :param k_neighbors:int
        :param n_neighbors:int
        :param kind:str, borderline-1,borderline-2
        :return:df
        """
        X = self._df[features].values
        y = self._df[self._target].values

        print("Original label shape {}".format(Counter(y)))

        if method == "ADASYN":
            overSm = ADASYN(sampling_strategy=sampling_strategy,random_state=random_state,n_neighbors=k_neighbors)
        elif method == "BorderLine":
            overSm = BorderlineSMOTE(sampling_strategy=sampling_strategy,random_state=random_state,k_neighbors=k_neighbors,m_neighbors=n_neighbors,kind=kind)
        elif method == "Random":
            overSm = RandomOverSampler(sampling_strategy=sampling_strategy,random_state=random_state)
        elif method == "SVM":
            overSm = SVMSMOTE(sampling_strategy=sampling_strategy,random_state=random_state,k_neighbors=k_neighbors,m_neighbors=n_neighbors,out_step=0.5)
        else:
            print("不支持{}该抽样方法".format(method))
            return self._df

        X_res,y_res = overSm.fit_resample(X,y)
        print("overSample label shape {}".format(Counter(y_res)))
        _data = np.concatenate([X_res, y_res.reshape(len(X_res), 1)], axis=1)
        df_new = pd.DataFrame(data=_data,columns=features+[self._target])
        return df_new