# -*- coding: utf-8 -*-
# @Time    : 2018-01-21 11:22
# @Author  : HuangSir
# @FileName: RiskStragety.py
# @Software: PyCharm

# 风险策略,稳定性评估PSI,评分决策表,KS
# 单变量PSI

import pandas as pd
import numpy as np

def caculate_ks(df_score:pd.DataFrame, score:str='score', target:str='target'):
    """
    指标KS,不分组情况下
    :return:KS
    """
    df = df_score.sort_values(by=score)
    total_good = len(df.loc[df[target] == 0, :])
    total_bad = len(df.loc[df[target] == 1, :])
    df_good, df_bad = pd.DataFrame(), pd.DataFrame()

    df_good['score'] = df.loc[df[target] == 0, score]
    df_good['sum_good'] = [i + 1 for i in range(len(df.loc[df[target] == 0, target]))]
    df_good['sum_good_rate'] = df_good['sum_good'] / total_good
    df_good = df_good.drop_duplicates(['score'], keep='last')

    df_bad['sum_bad'] = df.loc[df[target] == 1, target].cumsum()
    df_bad['score'] = df.loc[df[target] == 1, score]
    df_bad['sum_bad_rate'] = df_bad['sum_bad'] / total_bad
    df_bad = df_bad.drop_duplicates(['score'], keep='last')

    df = pd.merge(df_bad, df_good, how='outer', on='score')
    df = df.sort_values(by='score')
    df = df.fillna(method='ffill')
    df = df.fillna(0)

    df['KS'] = df['sum_bad_rate'] - df['sum_good_rate']
    KS = abs(df['KS']).max()
    return KS, df

def var_psi(cols:list,train:pd.DataFrame,test:pd.DataFrame):
    """
    入模变量PSI
    """
    psi_dict = {}
    for col in cols:
        psi_dict[col] = (train[col].value_counts(normalize=True,dropna=False)-test[col].value_counts(normalize=True,dropna=False))*\
                        (np.log(train[col].value_counts(normalize=True,dropna=False))/test[col].value_counts(normalize=True,dropna=False))
    psi_dict_var = {}
    # 输出每个变量的PSI
    for  col in cols:
        psi_dict_var[col] = psi_dict[col].sum()
    return psi_dict_var

def bin_list(low_line:int,up_line:int,step:int):
    """
    分组
    """
    bin = [] # 分组方法
    bin.append(float('-inf')) # 极小值
    for i in range(int(round((up_line-low_line)/step,0))):
        bin.append(low_line+step*i)
    bin.append(up_line)
    bin.append(float('inf')) # 极大值
    return bin


def stragety_score(df:pd.DataFrame,step:int=50,score:str='score',label:str='target'):
    """
    风险决策报表
    """
    #分组
    low_line = round(df[score].min()/step)*step
    up_line = int(df[score].max()/step)*step

    bin = bin_list(low_line,up_line,step) #分组列表
    # total 每个分数段总计
    total_cut = pd.cut(x=df[score],bins=bin,right=False)
    total_df = pd.value_counts(total_cut) # 每个分数段总计数
    total_sum = len(total_cut) # 总计

    # good 每个分数段的好客户
    temp_df = df.loc[df[label]==0,score]
    good_cut = pd.cut(x=temp_df,bins=bin,right=False)
    good_df = pd.value_counts(good_cut) # 每个分数段好客户计数
    good_sum = len(good_cut) # 好客户总数

    # bad 每个分数段的坏客户
    temp_df = df.loc[df[label]==1,score]
    bad_cut = pd.cut(x=temp_df,bins=bin,right=False)
    bad_df = pd.value_counts(bad_cut) # 每个分数段的坏客户计数
    bad_sum = len(bad_cut) # 坏客户总数

    # 总计,好,坏 计数合并
    df = pd.concat([total_df,good_df,bad_df],axis=1)
    df.columns = ['total','good','bad']
    # 累积
    df_cumsum = df.cumsum()
    df_cumsum.columns = ['sum_total','sum_good','sum_bad'] # 累积总,累积好,累积坏 计数
    df = pd.concat([df,df_cumsum],axis=1)

    # 累积率
    df['sum_total_rate'] = df['sum_total']/total_sum # 累积坏比率
    df['sum_good_rate'] = df['sum_good']/good_sum # 累积好比率
    df['sum_bad_rate'] = df["sum_bad"]/bad_sum # 累积坏比率
    df['KS'] = df['sum_bad_rate'] - df['sum_good_rate'] # 区分度ks
    df['bad_rate'] = df['bad']/df['total'] # 每个分数段的坏比率
    df['good_rate'] = df['good']/df['total'] #每个分数段的好比率
    return df


def score_psi(train:pd.DataFrame,test:pd.DataFrame,low_line:int,up_line:int,step:int=50,score='score'):

    """
    分数PSI
    """
    
    #分组
    bin = bin_list(low_line=low_line,up_line=up_line,step=step)
    train_len = len(train) # 训练集总数
    test_len = len(test) # 测试集总数

    train_cut = pd.cut(x=train[score],bins=bin,right=False)  # 训练集分组
    test_cut = pd.cut(x=test[score],bins=bin,right=False)  # 测试集分组结果
    train_df = pd.value_counts(train_cut)  # 训练集分组计数
    test_df = pd.value_counts(test_cut)  # 测试集分组计数
    df = pd.concat([train_df,test_df],axis=1) # 合并
    df.columns = ['train','test']
    df['train_percent'] = df['train']/train_len #每个分数段的计数比例
    df['test_percent'] = df['test']/test_len
    df['PSI'] = (df['train_percent']-df['test_percent'])*np.log(df['train_percent']/df['test_percent']) #每个分段的psi
    PSI = df['PSI'].sum()
    return PSI,df
