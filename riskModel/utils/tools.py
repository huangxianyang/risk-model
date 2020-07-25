# -*- coding: utf-8 -*-
"""
小工具类
"""
import math
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve,auc,f1_score,recall_score,precision_score,accuracy_score

def Prob2Score(prob,basePoint=600,PDO=50,odds=20):
    # 将概率转化成分数且为正整数
    y = np.log(prob/(1-prob))
    a = basePoint - y * np.log(odds)
    y2 = a - PDO/np.log(2)*(y)
    score = y2.astype('int')
    return score

def str2num(s):
    """
    字符转换为数字
    """
    try:
        if '.' in str(s):
            return float(s)
        else:
            return int(s)
    except:
        return s

def feature_subgroup(df, index,columns):

    """
    特征分组
    :param index:list
    :param columns:list
    :return: df
    """
    g = df.groupby(index).agg({col: 'nunique' for col in columns})
    if g[g > 1].dropna().shape[0] != 0:
        print("index非唯一值.")
    return df.groupby(index).agg({col: 'max' for col in columns})

def groupby_key(df,user_id,content,label):
    """
    根据user_id 合并content 和label列
    :param data: dataframe
    :param user_id:
    :param content: 文本列
    :param label:目标文件
    :return: dataframe
    """
    df[content] = df[content].astype("str")
    content_Series = df.groupby(by=user_id)[content].sum()
    content_df = pd.DataFrame({"user_id":content_Series.index,"content":content_Series.values})
    label_df = df[[user_id,label]].drop_duplicates()
    df= pd.merge(content_df,label_df,on=user_id,how="inner")
    return df


def best_prob(y_true,y_prob):
    """
    cut best prob
    :param y_prob: y of prediction
    :param y_true: real y
    :return: ks_value and draw ks
    """
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    max_ks = 0
    cut_prob = 0.5
    for i in range(len(thr)):
        if abs(fpr[i] - tpr[i]) > max_ks:
            max_ks = abs(fpr[i] - tpr[i])
            cut_prob = thr[i]
    return cut_prob

def model_norm(y_true,y_prob):
    """
    计算模型指标,auc,ks,f1,recall,precision,accuracy,cut_prob
    :param y_true:like-array
    :param y_prob:like-array
    :return:norm dict
    """
    norm = dict()
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    KS = 0
    cut_prob = 0.5
    for i in range(len(thr)):
        if abs(fpr[i] - tpr[i]) > KS:
            KS = abs(fpr[i] - tpr[i])
            cut_prob = thr[i]
    norm["AUC"] = auc(fpr, tpr)
    norm["KS"] = KS
    norm["cut_prob"] = cut_prob
    y_pred = np.array([1 if i > cut_prob else 0 for i in y_prob])
    norm["recall"] = recall_score(y_true=y_true,y_pred=y_pred,pos_label=1,average='binary')
    norm["precision"] = precision_score(y_true=y_true,y_pred=y_pred,pos_label=1,average='binary')
    norm["accuracy"] = accuracy_score(y_true=y_true,y_pred=y_pred)
    norm["f1"] = f1_score(y_true=y_true,y_pred=y_pred,pos_label=1,average='binary')
    return norm

def calculate_AIC(X,y_true,y_prob):
    """
       赤池信息准则AIC计算
    :param X: like-array
    :param y_true:like-array
    :param y_prob:like-array
    :return: float AIC
    """
    aic = 2 * X.shape[1] + X.shape[0] * math.log(pow(y_true - y_prob,2).sum() / X.shape[1])
    return aic




