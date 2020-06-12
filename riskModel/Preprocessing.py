# -*- coding: utf-8 -*-
# @Time    : 2018-01-21 11:22
# @Author  : HuangSir
# @FileName: Preprocessing.py
# @Software: PyCharm

"""
1. 变量类型划分--异常值检测--异常值处理--重复行(列)处理--众数处理--缺失处理
2. 样本采样: 随机采样, 随机平衡采样, Borderline-SMOTE过采样, Easy Ensemble 欠采样.
"""
import pandas as pd
import numpy as np
import numbers
from sklearn.utils import shuffle
from .utils.tools import str2num
import re
from imblearn.over_sampling import BorderlineSMOTE
from collections import Counter


class Preprocess(object):
    """
    数据预处理
    """

    def find_str(self, df: pd.DataFrame, cols: list):
        """
        连续数值变量包含字符串
        """
        s = set()
        for col in cols:
            if col in df.columns:
                for i in df[col].unique():
                    try:
                        if i is np.nan or isinstance(str2num(i), numbers.Real):
                            pass
                    except:
                        print(f'{col} ---> {i}')
                        s.add(i)
        return s

    def find_sign(self, df: pd.DataFrame, cols: list):
        """
        识别特殊符号
        """
        s = set()
        for col in cols:
            for v in df[col].unique():
                try:
                    special = r"\$|\(|\)|\*|\+|\[|\]|\?\\|\^|\||\@|\&|\{|\}|u"
                    t_s = set(re.findall(special, str(v)))
                    s.add(t_s)
                except:
                    pass
        return s

    def outlier_drop(self, df: pd.DataFrame, cols: list, low_percent=0.01, up_percent=0.99, cat_percent=0.001):
        """
        离群值
        """
        out_d = dict()
        object_cols = df.select_dtypes(include=['object', 'bool', 'category']).columns.tolist()
        for col in cols:
            if col in object_cols:  # 字符型变量
                cat_v_p = df[col].value_counts(normalize=True, dropna=False)  # 统计分类变量个数占比
                del_v = cat_v_p[cat_v_p < cat_percent].index.tolist()  # 删除值
                print(f'分类变量:{col},离群值:{del_v}')
                out_d[col] = del_v
            else:
                try:
                    q1 = df[col].quantile(low_percent)  # 下分为点
                    q2 = df[col].quantile(up_percent)  # 上分为点
                    print(f'连续变量:{col},low:{q1},up:{q2}')
                    out_d[col] = {'low': q1, 'up': q2}
                except TypeError:
                    pass
        out_d = {k: v for k, v in out_d.items() if v}
        return out_d

    def miss_mode_cnt(self, df: pd.DataFrame, cols: list):
        """
        缺失率和众数率统计
        """
        result = {}
        result['miss'], result['mode'] = {}, {}

        for col in cols:
            miss_rate = df[col].isnull().sum() / len(df)  # 缺失率
            result['miss'][col] = miss_rate

            mode_values = df[col].mode()[0]  # 众数
            mode_rate = len(df.loc[df[col] == mode_values, :]) / len(df)  # 众数率
            result['mode'][col] = mode_rate

        return result

    def fill_nan(self, df: pd.DataFrame, cols: list, values=-9999, min_miss_rate=0.05):
        """
        缺失值填充
        """
        miss_mode = self.miss_mode_cnt(df,cols)
        for col in cols:
            miss_rate = miss_mode['miss'][col]
            if miss_rate < min_miss_rate:
                mode = df[col].mode()[0]
                df[col] = df[col].fillna(mode)
            else:
                df[col] = df[col].fillna(values)

        return df

    def factor_map(self, df: pd.DataFrame, cols: list):
        """
        字符变量数值化
        """
        factorize_dict = {}
        for col in cols:
            factorize_dict[col] = {}
            l = pd.factorize(df[col])
            for i in np.arange(len(l[1])):
                factorize_dict[col][l[1][i]] = i  # 数值化映射字典
            df[col] = l[0]  # 数值化

        return df, factorize_dict  # 数值数据, 映射字典


class DataSample(BorderlineSMOTE):
    """
    数据抽样
    """

    def __init__(self, df: pd.DataFrame, cols: list, target='target',
                 sampling_strategy='minority', random_state=None, k_neighbors=5,
                 n_jobs=1, m_neighbors=10, kind='borderline-1'):
        super().__init__(sampling_strategy, random_state,
                         k_neighbors, n_jobs, m_neighbors, kind)
        self.df = df
        self.cols = cols
        self.target = target

    def balance_sample(self, odd, scalar: str = 'good'):
        """
        根据好坏比抽样
        """
        bad_df = self.df.loc[self.df[self.target] == 1, :]  # 坏样本
        good_df = self.df.loc[self.df[self.target] == 0, :]  # 好样本
        bad = len(bad_df)  # 坏样本数
        good = len(good_df)  # 好样本数
        if scalar == 'good':  # 对好样本进行抽样
            good_new = int(bad * odd)  # 所需好样本数
            if good_new > good:
                good_df_new = good_df.sample(n=good_new, replace=True)
            else:
                good_df_new = good_df.sample(n=good_new, replace=False)

            good_df = good_df_new

        else:
            bad_new = int(good / odd)  # 所需坏样本数
            if bad_new > bad:
                bad_df_new = bad_df.sample(n=bad_new, replace=True)
            else:
                bad_df_new = bad_df.sample(n=bad_new, replace=False)

            bad_df = bad_df_new

        self.df = pd.concat([bad_df, good_df], axis=0)  # 合并新样本
        self.df = shuffle(self.df)  # 打乱顺序
        return self.df

    def over_sample(self,int_cols:list):
        """
        过采样
        """
        print(f'Original label shape {Counter(self.df[self.target])}')

        X_res, y_res = self.fit_resample(self.df[self.cols], self.df[self.target])

        print(f'overSample label shape {Counter(y_res)}')

        data = np.concatenate([X_res, y_res.reshape(-1, 1)], axis=1)
        self.df = pd.DataFrame(data=data, columns=self.cols + [self.target])
        # 整数化
        for col in int_cols:
            self.df[col] = self.df[col].apply(lambda x:int(round(x,0)))

        return self.df
