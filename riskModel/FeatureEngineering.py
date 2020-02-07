# -*- coding: utf-8 -*-
# @Time    : 2018-01-21 11:22
# @Author  : HuangSir
# @FileName: FeatureEngineering.py
# @Software: PyCharm

"""
特征工程
1. 无序类别变量分箱组合, 有序数值变量分箱组合, 自定义分箱,WOE转换
2. 基于IV信息值特征选择, 基于重要度特征选择,基于共线性特征选择,
基于VIF方差膨胀因子特征选择,基于逐步回归特征选择,基于L1正则化特征选择
"""
import copy
import warnings
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['backend'] = 'Agg'
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from statsmodels.stats.outliers_influence import variance_inflation_factor # 共线性检验
from scorecardpy import woebin,woebin_ply,woebin_plot
# 自定义
from .utils.btl.merge import monotonous_bin
from .utils.PltFunction import PlotFeatureEn
from .utils.tools import calculate_AIC,model_norm
from .utils import InputCondition as IC

class FeatureBin(object):
    """
    变量分箱, 即特征离散化
    """
    def __init__(self,df,target="target",special_values=None,breaks_list=None,min_per_fine_bin=0.01,
                 stop_limit=0.02, min_per_coarse_bin=0.05, max_num_bin=5, method="tree"):
        """
        分箱类属性
        :param df: df
        :param target: str 标签名
        :param special_values: list, dict 特殊值, 为list是作用于所有变量, 也可以每个变量指定,
                               例如 {"A":[2600, 9960, "6850%,%missing"],"B":["education", "others%,%missing"]}
        :param breaks_list: list of dict 指定断点,变量指定,例如 {"A":[2600, 9960, "6850%,%missing"],"B":["education", "others%,%missing"]}
        :param min_per_fine_bin: float 初始分箱数,接受范围 0.01-0.2, 默认 0.01,即可初始分箱数为100
        :param stop_limit: float 最小IV值或卡方值,可接受范围 0-0.5, 默认 0.02.
        :param min_per_coarse_bin: float 最终每箱占比,接受范围 0.01-0.2, 默认 0.05, 即最小箱占比5%.
        :param max_num_bin: 最大分箱数 默认5
        :param method: str 方法,默认 tree 可选 chimerge 卡方分箱
        """
        self._df = IC.check_df(df)
        self._target = IC.check_str(target)
        if special_values is not None:
            self._special_values = IC.check_special_values(special_values)
        else:
            self._special_values = special_values
        if breaks_list is not None:
            self._breaks_list = IC.check_list_of_dict(breaks_list)
        else:
            self._breaks_list = breaks_list
        self._min_per_fine_bin = IC.check_float(min_per_fine_bin)
        self._stop_limit = IC.check_float(stop_limit)
        self._min_per_coarse_bin = IC.check_float(min_per_coarse_bin)
        self._max_num_bin = IC.check_int(max_num_bin)
        self._method = IC.check_str(method)

    def category_bin(self,bin_feature,max_num_bin=None):
        """
        无序类别变量分箱组合
        :param max_num_bin:int 最大分箱数
        :param bin_feature: list, 参与分箱的变量
        :return: bin_dict:dict, var_iv:dict
        """
        bin_feature = IC.check_list(bin_feature)
        t0 = time.process_time()
        bin_dict,var_iv = dict(),dict()
        df = self._df
        df[bin_feature] = df[bin_feature].astype("str") # 防止误将无需类别变量当做连续变量处理

        if max_num_bin == None:
            max_num_bin = self._max_num_bin
        else:
            max_num_bin = IC.check_int(max_num_bin)

        # 开始分箱
        for col in bin_feature:
            bin_dict[col] = woebin(dt=df[[col,self._target]],y=self._target,x=col,breaks_list=self._breaks_list,special_values=self._special_values,
                                   min_perc_fine_bin=self._min_per_fine_bin,min_perc_coarse_bin=self._min_per_coarse_bin,stop_limit=self._stop_limit,
                                   max_num_bin=max_num_bin,method=self._method)[col]
            var_iv[col] = bin_dict[col]["total_iv"].unique()[0]

        print("处理完{}个无序类别变量,耗时:{}秒".format(len(bin_feature),(time.process_time()-t0)*100/60))

        return bin_dict,var_iv

    def number_bin(self,bin_feature,max_num_bin=None,no_mono_feature=None):
        """
        有序数值变量分箱组合
        :param bin_feature:list, 参与分箱的变量
        :param max_num_bin: int 最大分箱数
        :param no_mono_feature list 不参与单调检验的变量
        :return: bin_dict:dict, var_iv:dict
        """
        t0 = time.process_time()
        bin_dict,var_iv = {},{}
        df = copy.deepcopy(self._df)
        bin_feature = IC.check_list(bin_feature)
        df[bin_feature] = df[bin_feature].astype("float") #防止误将分类变量当做连续变量处理,若不是, 调用该函数将报错

        if max_num_bin == None:
            max_num_bin = self._max_num_bin
        else:
            max_num_bin = IC.check_int(max_num_bin)
        if no_mono_feature == None:
            no_mono_feature = []
        else:
            no_mono_feature = IC.check_list(no_mono_feature)

        # 开始分箱
        for col in bin_feature:
            if isinstance(self._special_values,dict):
                special_values = self._special_values[col]
            else:
                special_values = self._special_values
            # 对于唯一值的变量进行跳过
            unique_values = [v for v in df[col].unique() if v not in special_values]
            if len(unique_values) ==1:
                warnings.warn("There are {} columns have only one unique values,{}, which are skipping from this bing.".format(col,unique_values))
                continue

            if col not in no_mono_feature:
                cutOffPoints = woebin(dt=df[[col,self._target]],y=self._target,x=col,breaks_list=self._breaks_list,special_values=special_values,
                                      min_perc_fine_bin=self._min_per_fine_bin,min_perc_coarse_bin=self._min_per_coarse_bin,stop_limit=self._stop_limit,
                                      max_num_bin=max_num_bin,method=self._method)[col]["breaks"].tolist()

                cutOffPoints = [float(i) for i in set(cutOffPoints) if str(i) not in ['inf','-inf']]  # 切分点
                cutOffPoints = sorted([i for i in cutOffPoints if i not in special_values])

                if not cutOffPoints: # 切分点为空
                    warnings.warn("There are zero cutOffPoint of {} columns from this bing, which select all unique values insert cutOffPoint".format(col))
                    cutOffPoints = sorted([i for i in df[col].unique() if i not in special_values])

                # 单调检验合并方案结果
                # mono_cutOffPoints:dict
                mono_cutOffPoints = monotonous_bin(df=df[[col,self._target]],col=col,cutOffPoints=cutOffPoints,
                                                   target=self._target,special_values=special_values)
            else:
                mono_cutOffPoints = {}

            # 最终方案
            bin_dict[col] = woebin(dt=df[[col,self._target]],y=self._target,x=col,breaks_list=mono_cutOffPoints,special_values=special_values,
                                      min_perc_fine_bin=self._min_per_fine_bin,min_perc_coarse_bin=self._min_per_coarse_bin,stop_limit=self._stop_limit,
                                      max_num_bin=max_num_bin,method=self._method)[col]
            # 保存IV
            var_iv[col] = bin_dict[col]["total_iv"].unique()[0]

        print("处理完{}个有序数值变量,耗时:{}秒".format(len(bin_feature), (time.process_time() - t0) * 100 / 60))
        return bin_dict, var_iv

    def self_bin(self,var,special_values=None,breaks_list=None):
        """
        指定断点
        :param var: str, 自定义分箱单变量
        :param special_values: list of dict
        :param breaks_list: list of dict
        :return: bin_dict:dict, var_iv:dict

    breaks_list = {
      'age.in.years': [26, 35, 37, "Inf%,%missing"],
      'housing': ["own", "for free%,%rent"]
      }
    special_values = {
      'credit.amount': [2600, 9960, "6850%,%missing"],
      'purpose': ["education", "others%,%missing"]
      }
        """
        var = IC.check_str(var)
        bin_dict, var_iv = dict(),dict()
        if special_values == None:
            special_values = self._special_values
        else:
            special_values = IC.check_special_values(special_values)
        if breaks_list == None:
            breaks_list = self._breaks_list
        else:
            breaks_list = IC.check_list_of_dict(breaks_list)

        bin_dict[var] = woebin(dt=self._df[[var,self._target]], y=self._target, x=var,
                               breaks_list=breaks_list, special_values=special_values)[var]
        # 保存IV
        var_iv[var] = bin_dict[var]["total_iv"].unique()[0]

        return bin_dict,var_iv

def woe_trans(df,bin_dict,trans_feature,target="target"):
    """
    woe 转换
    :param df: origin data 包含 目标变量
    :param bin_dict: dict
    :param trans_feature:list 转换变量
    :param target: str 目标变量
    :return: df_woe,new_feature
    """
    df = IC.check_df(df)
    bin_dict = IC.check_dict_of_df(bin_dict)
    trans_feature = IC.check_list(trans_feature)
    if not set(trans_feature).issubset(set(list(bin_dict.keys()))):
        print("bin_dict.key",bin_dict.keys(),'\n')
        print("trans_feature:",trans_feature)
        warnings.warn("trans_feature not in bin_dict.keys, Please double check feature set")
        raise SystemExit(0)

    dt = df[trans_feature+[target]]
    df_woe = woebin_ply(dt=dt,bins=bin_dict)
    new_feature = [i+"_woe" for i in trans_feature]

    if not set(new_feature).issubset(set(df_woe.columns.difference([target]).tolist())):
        warnings.warn("new feature not in df_woe.columns, Please double check feature set")
        raise SystemExit(0)

    return df_woe,new_feature

class WoeBinPlot(object):
    """
    分箱作图
    """
    def __init__(self,bin_dict):
        """
        :param bin_dict:DataFrame of bin_dict, 分箱数据字典
        """
        bin_dict = IC.check_dict_of_df(bin_dict)
        self._bins = bin_dict

    def woe_plot(self,features:list=None,show_iv:bool=False,save:bool = False,path:str='./'):
        """
        woe全部变量作图
        :param var: str, 作图变量
        :param title: str 图标题
        :param show_iv: bool, 是否展示iv
        :param save: bool 是否保存
        :return: dict a dict of matplotlib figure objests
        ----------
         Examples
        ----------
        plotlist = woe_plot(bins)

        # save binning plot
        for key,i in plotlist.items():
            plt.show(i)
            plt.savefig(str(key)+'.png')
        """
        features = IC.check_list(features)
        if not set(features).issubset(set(list(self._bins.keys()))):
            print("bin_dict.key", self._bins.keys(), '\n')
            print("features:", features)
            warnings.warn("features not in bin_dict.keys, Please double check feature set")
            raise SystemExit(0)

        plotList = woebin_plot(bins=self._bins,x=features,show_iv=show_iv)
        if save:
            # save binning plot
            for key,i in plotList.items():
                plt.savefig(path+str(key)+'_bin.png')
                plt.close()
        return plotList

PltF = PlotFeatureEn()
class SelectFeature(object):
    """特征选择"""

    def __init__(self,df_woe):
        df_woe = IC.check_df(df_woe)
        self._df_woe = df_woe

    def baseOn_iv(self,var_iv:dict,threshold:float=0.02,xlabel=None,figsize:tuple=(15,7), is_save:bool=False,path="./"):
        """
        选择IV高于阈值的变量, 一般说来,信息值0.02以下表示与目标变量相关性非常弱。
        0.02-0.1很弱；0.1-0.3一般；0.3-0.5强；0.5-1很强,1以上异常,单独关注
        :param var_iv: dict 特征信息值字典
        :param threshold: float,iv阈值
        :param path:文件存储地址
        :return:dict
        """
        var_iv = IC.check_dict_of_float(var_iv)
        high_IV = {k: v for k, v in var_iv.items() if v >= threshold} # 根据IV选择变量
        high_IV = {k:v for (k,v) in sorted(high_IV.items(),key=lambda x:x[1],reverse=True)} # 排序
        high_IV_df = pd.Series(high_IV)
        if is_save: #保存IV图片及IV表
            PltF.draw_IV(IV_dict=high_IV, path=path, xlabel=xlabel, figsize=figsize, is_save=is_save)
            high_IV_df.to_excel(path+"high_iv_df.xlsx",index=False)
        return high_IV

    def baseOn_importance(self,features:list,target:str='target',n_estimators:int = 100,is_save:bool=False,
                           figsize: tuple = (15, 7),path="./"):
        """
        基于随机森林特征权重
        --------------------
        parameter:
                 features: list 评估变量集
                 n_estimators: 随机森林树量
                 target: str
                 is_save:bool 作图
                 figsize: tuple 图片大小
                 path: str 地址
        return:
              feature_importance: dict and importance draw
        """
        X = self._df_woe[features]
        y= self._df_woe[target]
        features = IC.check_list(features)

        rf = RandomForestClassifier(n_estimators=n_estimators, random_state=42)  # 构建分类随机森林分类器
        rf.fit(X.values, y.values)  # 对自变量和因变量进行拟合

        # feature importances dict
        importance = rf.feature_importances_
        feature_importance = dict()
        for k, v in zip(features, importance):
            feature_importance[k] = v
        # 可视化
        if is_save:
            PltF.draw_importance(importance=importance,features=features,figsize=figsize,path=path)
        feature_importance = {k:v for k,v in sorted(feature_importance.items(),key=lambda x:x[1],reverse=True)}
        return feature_importance


    def baseOn_collinear(self,features:dict,cor_threshold:float=0.7,
                        is_save:bool=False,figsize:tuple=(12,12),path:str ='./'):

        # 基于两两相关性共线性检验
        # 1,将候选变量按照IV进行降序排列
        # 2，计算第i和第i+1的变量的线性相关系数
        # 3，对于系数超过阈值的两个变量，剔除IV较低的一个
        features = IC.check_dict_of_float(features)

        deleted_feature = [] # 删除的变量
        for col1 in features.keys():
            if col1 in deleted_feature:
                continue
            for col2 in features.keys():
                if col1 == col2 or col2 in deleted_feature:
                    continue
                cor_v = np.corrcoef(self._df_woe[col1], self._df_woe[col2])[0, 1]
                if abs(cor_v) >= cor_threshold:
                    if features[col1] >= features[col2]:
                        deleted_feature.append(col2)
                        print("相关性检验,删除变量: ",col2)
                    else:
                        deleted_feature.append(col1)
                        print("相关性检验,删除变量: ", col1)

        last_feature = [i for i in features.keys() if i not in deleted_feature]
        # 多变量分析：VIF
        X = np.matrix(self._df_woe[last_feature])
        VIF_list = [variance_inflation_factor(X, i) for i in range(X.shape[1])]
        max_VIF = max(VIF_list)
        print("最大方差膨胀因子:{}".format(max_VIF))
        feature_VIF = {k:v for k,v in zip(last_feature,VIF_list)}
        #相关性可视化
        if is_save:
            PltF.draw_corr(df=self._df_woe,figsize=figsize,path=path)
        return feature_VIF

    def baseOn_vif(self,features:dict,max_vif:int=10):
        """
        基于方差膨胀因子共线性检验
        ---------------------------------------
        parameter:
                 df: df
                 features:dict
                 max_ivf : float or int, vif threshold
        return:
               keep_feature_vif:dict

        """
        features = IC.check_dict_of_float(features)
        data = self._df_woe
        features_sort = list({k:v for k,v in sorted(features.items(), key=lambda x: x[1], reverse=True)}.keys())  # 排序
        select_feature = [features_sort[0]]
        for col,i in zip(features_sort[1:],range(1,len(features_sort))):
            temp_feature = select_feature + [features_sort[i]]
            X = np.matrix(data[temp_feature])
            vif_list = [variance_inflation_factor(X, i) for i in range(X.shape[1])]
            if max(vif_list) < max_vif:
                select_feature.append(col)
            else:
                print("共线性VIF: {},删除变量: {}".format(max(vif_list),col))

        vif_last = [variance_inflation_factor(data[select_feature].values, i) for i in range(len(select_feature))]
        feature_VIF = {k:v for k,v in zip(select_feature,vif_last)}
        return feature_VIF

    def baseOn_steplr(self,features:dict,target:str="target",C:float=1,class_weight="balanced",norm:str="AIC"):
        """
        基于逐步回归特征选择,评估准则: 赤池信息准则AIC,区分度KS,受试者特征AUC
        :param features: dict 特征重要度或信息值
        :param target: str 标签
        :param C: float 正则值
        :param norm option, AIC或者KS
        :param class_weight: 类别权重
        :return: select_feature
        """
        features = IC.check_dict_of_float(features)
        data = self._df_woe
        features_sort = list({k: v for (k, v) in sorted(features.items(), key=lambda x: x[1], reverse=True)}.keys())  # 排序
        X, y = data[features_sort], data[target].values

        # 当前AIC和AUC,KS
        lr = LogisticRegression(C=C, penalty='l1', class_weight=class_weight, solver='liblinear')
        lr.fit(X.values, y)
        y_prob = lr.predict_proba(X.values)[:, 1]
        AIC = calculate_AIC(X=X, y_true=y, y_prob=y_prob)
        KS_AUC = model_norm(y_true=y, y_prob=y_prob)
        print("current AIC is {},KS is {}, AUC is {}".format(AIC, KS_AUC["KS"], KS_AUC['AUC']))

        ###################################################################
        # 向前逐步回归
        print('向前逐步回归..................................')
        if norm in ['AUC','KS']:
            best = -99
        else:
            best = 9999999999 # 默认AIC评估指标

        forward_feature = list()
        for col in features_sort:
            X_new = X[forward_feature+[col]].values
            lr = LogisticRegression(C=C, penalty='l1', class_weight=class_weight, solver='liblinear')
            lr.fit(X_new,y)
            y_prob = lr.predict_proba(X_new)[:,1]

            if norm == 'AUC':
                AUC_new = model_norm(y_true=y, y_prob=y_prob)["AUC"]
                if AUC_new > best:
                    best = AUC_new
                    forward_feature.append(col)

            if norm == 'KS':
                KS_new = model_norm(y_true=y, y_prob=y_prob)["KS"]
                if KS_new > best:
                    best = KS_new
                    forward_feature.append(col)
            else:
                AIC_new = calculate_AIC(X=X_new,y_true=y,y_prob=y_prob)
                # print('AIC:',norm_new)
                if AIC_new < best:
                    best = AIC_new
                    forward_feature.append(col)
        print("forward step best {} is {}".format(norm,best))
        print('forward step deleted cols:',[i for i in features_sort if i not in forward_feature])

        ###################################################################
        # 向后逐步回归
        print('向后逐步回归..................................')
        deleted_feature = []
        for col in features_sort[::-1]:
            select_feature = [i for i in features_sort if i not in deleted_feature]
            X_new = X[select_feature].values
            lr = LogisticRegression(C=C, penalty='l1', class_weight=class_weight, solver='liblinear')
            lr.fit(X_new,y)
            y_prob = lr.predict_proba(X_new)[:,1]

            if norm == 'AUC':
                AUC_new = model_norm(y_true=y, y_prob=y_prob)["AUC"]
                if AUC_new >= KS_AUC['AUC']:
                    KS_AUC['AUC'] = AUC_new
                    deleted_feature.append(col)
            if norm == 'KS':
                KS_new = model_norm(y_true=y, y_prob=y_prob)["KS"]
                if KS_new >= KS_AUC['KS']:
                    KS_AUC['KS'] = KS_new
                    deleted_feature.append(col)
            else:
                AIC_new = calculate_AIC(X=X_new, y_true=y, y_prob=y_prob)
                if AIC_new <= AIC:
                    AIC = AIC_new
                    deleted_feature.append(col)
        ############################################
        if norm == 'AUC':
            print('backward step AUC:',KS_AUC['AUC'])
        elif norm == 'KS':
            print('backward step KS:', KS_AUC['KS'])
        else:
            print('backward step AIC:', AIC)
        ######################################################
        print("backward step deleted cols are {}".format(deleted_feature))
        last_feature = {k:v for k,v in features.items() if k in forward_feature and k not in deleted_feature}
        return last_feature


    def baseOn_l1(self,features:list,target:str="target",C=1,class_weight="balanced",drop_plus:bool=False):
        """
        基于L1正则选择特征
        :param features: list 模型特征
        :param target: str 标签
        :param C: float 正则力度 >0
        :param class_weight: str, dict 标签权重, 例如,{1:0.8,0:0.2}
        :param drop_plus bool 是否删除正数
        :return:feature_coe 变量系数
        """
        features = IC.check_list(features)
        X = self._df_woe[features]
        y = self._df_woe[target]

        lr = LogisticRegression(C=C, penalty='l1', class_weight=class_weight,solver='liblinear')
        lr.fit(X,y)
        # 模型系数
        paramsEst = pd.Series(lr.coef_.tolist()[0], index=features)
        feature_coe = paramsEst.to_dict()
        # 变量选择
        if drop_plus:
            select_feature = {k:v for k,v in feature_coe.items() if v < 0}
        else:
            select_feature = {k:v for k,v in feature_coe.items() if v > 0}
        return select_feature