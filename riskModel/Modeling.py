# -*- coding: utf-8 -*-
# @Time    : 2018-01-21 11:22
# @Author  : HuangSir
# @FileName: Modeling.py
# @Software: PyCharm

"""
超参调优,模型训练,模型评估,标准评分卡转换,模型预测(概率预测,评分预测)
"""
import re
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression,LogisticRegressionCV
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
plt.rcParams['backend'] = 'Agg'
# from sklearn.model_selection import cross_val_score # 交叉验证
# from bayes_opt import BayesianOptimization # 贝叶斯调参
from .utils.tools import model_norm,Prob2Score
from .utils.PltFunction import PlotModel
from scorecardpy import scorecard_ply
from .utils import InputCondition as IC

class SearchParam(object):
    """
    超参C值调优
    方法:网格搜索
    """
    def __init__(self,X,y):
        """
        :param X: like-array
        :param y: like-array
        """
        X = IC.check_array(X)
        y = IC.check_array(y)
        self.X = X
        self.y = y

    def grid_search(self,param_grid:dict,cv:int=5,class_weight="balanced",scoring:str="roc_auc"):
        """
        网格搜索
        :param param_grid:dict, 检索参数
        :param cv:int 交叉验证
        :param class_weight: 类别权重
        :param scoring:str, 评估指标
        :return:float best_param
        """
        param_grid = IC.check_list_of_dict(param_grid)
        cv = IC.check_int(cv)

        lr = LogisticRegression(class_weight=class_weight)
        gr = GridSearchCV(estimator=lr,param_grid=param_grid,scoring=scoring,cv=cv)
        gr.fit(self.X,self.y)
        best_param = gr.best_params_
        print("grid search best score is ",gr.best_score_)
        return best_param

    # def bayes_search(self,param_grid:dict,cv:int=5,n_iter:int=20,
    #                  class_weight="balanced", scoring="roc_auc"):
    #     """
    #     基于贝叶斯调参
    #     :param param_grid:dict
    #     :param cv: int
    #     :param n_iter:int
    #     :return:bast_param
    #     """
    #     param_grid = IC.check_list_of_dict(param_grid)
    #     C = param_grid["C"]
    #     L = param_grid["penalty"]
    #     X,y = self.X,self.y
    #     best_param = {}
    #     for l in L:
    #         def target(C):
    #             """目标函数"""
    #             lr = LogisticRegression(penalty=l,C=C,class_weight=class_weight,random_state=42)
    #             cls = cross_val_score(estimator=lr,X=X,y=y,scoring=scoring,cv=cv).mean()
    #             return cls
    #
    #         bs = BayesianOptimization(f=target,pbounds={"C":C},random_state=42,verbose=2)
    #         bs.maximize(n_iter=n_iter)
    #         best_param[l] = bs.max
    #     return best_param

class TrainLr(object):
    """模型实例化"""
    def __init__(self,df_woe,features:list,target="target",penalty="l1",class_weight="balanced"):
        """
        :param features:list
        :param X:like-array
        :param y: like-array
        :param penalty: str, l1, l2
        :param class_weight: dict
        """
        df_woe = IC.check_df(df_woe)
        features = IC.check_list(features)
        self.features = features
        self.X = df_woe[features].values
        self.y = df_woe[target].values
        self.penalty = penalty
        self.class_weight = class_weight
        if penalty == "l1":
            self.solver = "liblinear"
        else:
            self.solver = "lbfgs"

    def lr(self,C,filename='./'):
        """
        :param C: float 正则参数
        :return: object lr_model
        """
        C = IC.check_float(C)
        lr_model = LogisticRegression(penalty=self.penalty,C=C,solver=self.solver,class_weight=self.class_weight)
        lr_model.fit(X=self.X,y=self.y)
        # 评估
        y_prob = lr_model.predict_proba(self.X)[:,1]
        norm = model_norm(y_true=self.y, y_prob=y_prob)
        print("training model result:",norm)
        # 作图
        pltm = PlotModel(y_true=self.y,y_prob=y_prob)
        pltm.plot_roc_curve(filename=filename)
        pltm.plot_ks_curve(filename=filename)
        # 模型系数
        paramsEst = pd.Series(lr_model.coef_.tolist()[0], index=self.features)
        paramsEst["intercept"] = lr_model.intercept_.tolist()[0]
        print("model params:", paramsEst)
        return lr_model

    def lr_cv(self,Cs=1,cv=5,scoring="roc_auc",filename='./',solver='liblinear'):
        """
        交叉验证训练
        :param Cs: float, list
        :param cv: int
        :param scoring: str 评估指标
        :return: object
        """
        Cs = IC.check_float_list(Cs)
        cv = IC.check_int(cv)

        if self.penalty == 'l1':
            solver = 'liblinear'
        lr_model = LogisticRegressionCV(Cs=Cs,cv=cv,penalty=self.penalty,scoring=scoring,solver=solver)
        lr_model.fit(self.X,self.y)
        # 评估
        y_prob = lr_model.predict_proba(self.X)[:,1]
        norm = model_norm(y_true=self.y,y_prob=y_prob)
        print("training model result:",norm)
        # 作图
        pltm = PlotModel(y_true=self.y,y_prob=y_prob)
        pltm.plot_roc_curve(filename=filename)
        pltm.plot_ks_curve(filename=filename)
        # 模型系数
        paramsEst = pd.Series(lr_model.coef_.tolist()[0], index=self.features)
        paramsEst["intercept"] = lr_model.intercept_.tolist()[0]
        print("model params:", paramsEst)
        return lr_model

class ScoreCard(object):
    """
    评分卡模型
    """
    def __init__(self,lr,bin_dict,model_feature,score0=600,pdo=50):
        """
        :param lr:object, 经过实例化后的lr
        :param bin_dict:dict of df 分箱字典
        :param model_feature: list 模型变量
        :param score0:int 初始分
        :param pdo: 倍率
        """
        self.model = lr
        self.bins = IC.check_dict_of_df(bin_dict)
        self.model_feature = IC.check_list(model_feature)
        self.score0 = IC.check_int(score0)
        self.pdo= IC.check_int(pdo)

    def score_card(self,return_df:bool=True):
        '''
        Creating a Scorecard
        ------
        `scorecard` creates a scorecard based on the results from `bins`
        and LogisticRegression of sklearn.linear_model

        Returns
        ------
        DataFrame
            scorecard df
        '''
        # coefficients
        A = self.score0
        B = self.pdo / np.log(2)
        # bins # if (is.list(bins)) rbindlist(bins)
        if isinstance(self.bins, dict):
            bins_df = pd.concat(self.bins, ignore_index=True)
        else:
            bins_df = None

        xs = [re.sub('_woe$', '', i) for i in self.model_feature]
        # coefficients
        coef_df = pd.Series(self.model.coef_[0], index=np.array(xs)).loc[lambda x: x != 0]  # .reset_index(drop=True)

        # scorecard
        basepoints = A - B * self.model.intercept_[0]
        card = {}
        card['baseScore'] = pd.DataFrame({'variable': "basepoints", 'bin': "基础分", 'points': round(basepoints, 2)},
                                          index=np.arange(1))
        for i in coef_df.index:
            card[i] = bins_df.loc[bins_df['variable'] == i, ['variable', 'bin', 'woe']] \
                .assign(points=lambda x: round(-B * x['woe'] * coef_df[i], 2))[["variable", "bin", "points"]]

        # 转换为df
        df = pd.DataFrame()
        for col in card.keys():
            col_df = card[col]
            df = pd.concat([df,col_df])
        df.set_index(["variable"], inplace=True)

        if return_df:
            return df
        else:
            return card

    def pred_score(self,df_woe,only_total_score=True):
        df_woe = IC.check_df(df_woe)
        df_score = df_woe
        y_prob = self.model.predict_proba(df_woe[self.model_feature].values)[:,1]
        score = Prob2Score(prob=y_prob,basePoint=self.score0,PDO=self.pdo)
        df_score["score"] = score
        if only_total_score:
            return df_score["score"]
        else:
            return df_score

    def score_ply(self,df,only_total_score=True, print_step=0):
        df_score = scorecard_ply(dt=df,card=self.score_card(return_df=False),only_total_score=only_total_score,print_step=print_step)
        return df_score