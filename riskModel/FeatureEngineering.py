# -*- coding: utf-8 -*-
# @Time    : 2018-01-21 11:22
# @Author  : HuangSir
# @FileName: FeatureEngineering.py
# @Software: PyCharm

"""
特征工程
1. 无序类别变量分箱组合, 有序数值变量分箱组合, 自定义分箱,WOE转换
2. 基于IV信息值特征选择,基于共线性特征选择,
基于VIF方差膨胀因子特征选择,基于逐步回归特征选择,基于L1正则化特征选择
"""
import time
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV,StratifiedKFold
from toad.transform import Combiner
from scorecardpy import woebin
from statsmodels.stats.outliers_influence import variance_inflation_factor  # 共线性检验
# 自定义
from .utils.btl.merge import monotonous_bin
from .utils.PltFunction import PlotFeatureEn

def cat_bin(df:pd.DataFrame,cols:list=None,target:str='target',specials:list=None,
            bin_num_limit:int=5,count_distr_limit:float=0.05,method:str='chimerge',**kwargs):
    if not cols:
        cols = df.columns.difference([target]).tolist()

    if specials:
        specials = {k:specials for k in cols}

    bind, ivd = dict(), dict()
    t0 = time.process_time()
    for col in cols:
        bind[col] = woebin(dt=df,x=col,y=target,special_values=specials,bin_num_limit=bin_num_limit,
                           count_distr_limit=count_distr_limit,method=method,print_info=False,
                           **kwargs)[col]
        ivd[col] = bind[col]['total_iv'].unique()[0]
    print(f'there are bing {len(cols)} using {int((time.process_time() - t0) * 100 / 60)} seconds')
    return bind, ivd

def num_bin(df:pd.DataFrame,cols:list=None,target:str='target',specials:list=None,
            bin_num_limit:int=5,count_distr_limit:float=0.05,sc_method='chimerge',
            non_mono_cols:list=None,init_bins=10,init_min_samples=0.05,init_method='chi',**kwargs):

    # 粗分箱,单调检验,分箱结果
    if not cols:
        cols = df.columns.difference([target]).tolist()

    if specials:
        specials = {k: specials for k in cols}

    if not non_mono_cols:
        non_mono_cols = []

    bind, ivd = dict(), dict()
    t0 = time.process_time()

    for col in cols:
        if col in non_mono_cols:
            bind[col] = woebin(dt=df, x=col, y=target, special_values=specials, bin_num_limit=bin_num_limit,
                               count_distr_limit=count_distr_limit, method=sc_method,print_info=False)[col]
            ivd[col] = bind[col]['total_iv'].unique()[0]

        else:
            c = Combiner()
            c.fit(X=df[col], y=df[target],n_bins=init_bins,min_samples=init_min_samples,method=init_method,**kwargs)
            init_points = c.export()[col]
            breaks_list = monotonous_bin(df=df, col=col, target=target,cutOffPoints=init_points, special_values=specials)

            bind[col] = woebin(dt=df, x=col, y=target, special_values=specials, breaks_list=breaks_list,
                               bin_num_limit=bin_num_limit,count_distr_limit=count_distr_limit,method=sc_method,
                               print_info=False)[col]
            ivd[col] = bind[col]['total_iv'].unique()[0]

    print(f'there are bing {len(cols)} using {int((time.process_time() - t0) * 100 / 60)} seconds')
    return bind, ivd

class SelectFeature(object):
    """特征选择"""

    def baseOn_iv(self, ivd: dict, thred: float = 0.02, is_draw=False):
        """
        选择IV高于阈值的变量, 一般说来,信息值0.02以下表示与目标变量相关性非常弱
        """
        high_IV = {k: v for k, v in ivd.items() if v >= thred}
        high_IV = {'_'.join([k,'woe']): v for (k, v) in sorted(high_IV.items(), key=lambda x: x[1], reverse=True)}  # 排序
        if is_draw:
            PltF = PlotFeatureEn()
            PltF.draw_IV(IV_dict=high_IV, path='./')
        return high_IV

    def baseOn_collinear(self, df: pd.DataFrame, high_iv: dict, thred: float = 0.7, is_draw=False):

        # 基于两两相关性共线性检验
        # 1,将候选变量按照IV进行降序排列
        # 2，计算第i和第i+1的变量的线性相关系数
        # 3，对于系数超过阈值的两个变量，剔除IV较低的一个
        deleted_feature = []  # 删除的变量
        for col1 in high_iv.keys():
            if col1 in deleted_feature:
                continue
            for col2 in high_iv.keys():
                if col1 == col2 or col2 in deleted_feature:
                    continue
                cor_v = np.corrcoef(df[col1], df[col2])[0, 1]
                if abs(cor_v) >= thred:
                    if high_iv[col1] >= high_iv[col2]:
                        deleted_feature.append(col2)
                        print(f'相关性检验,删除变量:{col2}')
                    else:
                        deleted_feature.append(col1)
                        print(f'相关性检验,删除变量:{col1}')

        last_feature = [i for i in high_iv.keys() if i not in deleted_feature]
        # 多变量分析：VIF
        X = df[last_feature].values
        VIF_list = [variance_inflation_factor(X, i) for i in range(X.shape[1])]
        max_VIF = max(VIF_list)
        print(f'最大方差膨胀因子:{max_VIF}')
        feature_VIF = {k: v for k, v in zip(last_feature, VIF_list)}
        # 相关性可视化
        if is_draw:
            PltF = PlotFeatureEn()
            PltF.draw_corr(df=df, figsize=(15, 15), path='./')
        return feature_VIF

    def baseOn_l1(self, X: pd.DataFrame, y: pd.Series,Kfold:int=5,drop_plus: bool = False):
        """
        基于L1正则选择特征
        """
        lr = LogisticRegression(penalty='l1', class_weight='balanced', solver='liblinear')
        k = StratifiedKFold(n_splits=Kfold, shuffle=True, random_state=42)
        gs = GridSearchCV(estimator=lr,param_grid={'C':np.arange(0.005,0.5,0.005)},scoring='roc_auc',cv=k)
        _ = gs.fit(X,y)
        print(f'best_scoring:{round(gs.best_score_,3)}')
        print(f'best_params:{gs.best_params_}')
        C = gs.best_params_['C']
        lr = LogisticRegression(penalty='l1', C=C, class_weight='balanced', solver='liblinear')
        _ = lr.fit(X, y)
        # 模型系数
        features = X.columns.tolist()
        paramsEst = pd.Series(lr.coef_.tolist()[0], index=features)
        feature_coe = paramsEst.to_dict()
        # 变量选择
        if drop_plus:
            ml_cols = {k: v for k, v in feature_coe.items() if v < 0}
        else:
            ml_cols = {k: v for k, v in feature_coe.items() if v > 0}
        return ml_cols,C
