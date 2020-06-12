# -*- coding: utf-8 -*-
# @Time    : 2018-01-21 11:22
# @Author  : HuangSir
# @FileName: Modeling.py
# @Software: PyCharm

"""
模型训练,模型评估,标准评分卡转换,模型预测(概率预测,评分预测)
"""
import re
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from .utils.tools import model_norm,Prob2Score
from .utils.PltFunction import PlotModel
from scorecardpy import scorecard_ply

class Lr(LogisticRegression):
    """模型实例化"""
    def __init__(self,C,**kwargs):
        super().__init__(**kwargs)
        self.C = C
        self.penalty='l1'
        self.class_weight='balanced'
        self.solver='liblinear'

    def fiting(self,X:pd.DataFrame,y:pd.Series,filename='./'):
        """
        lr cv
        """
        features = X.columns.tolist()

        if self.penalty == 'l1':
            self.solver = 'liblinear'

        # 拟合
        self.fit(X,y)
        # 评估
        y_prob = self.predict_proba(X)[:,1]
        norm = model_norm(y_true=y,y_prob=y_prob)
        print(f'training model result:\n{norm}')

        # 作图
        pltm = PlotModel(y_true=y.values,y_prob=y_prob)
        pltm.plot_roc_curve(filename=filename)
        pltm.plot_ks_curve(filename=filename)
        # 模型系数
        paramsEst = pd.Series(self.coef_.tolist()[0], index=features)
        paramsEst['intercept'] = self.intercept_.tolist()[0]
        print(f'model params:\n{paramsEst}')

class ScoreCard(object):
    """
    评分卡模型
    """
    def __init__(self,lr,bins:dict,ml_cols:list,score0:int=600,pdo:int=50):

        self.model = lr
        self.bins = bins
        self.model_feature = ml_cols
        self.score0 = score0
        self.pdo= pdo

    def score_card(self,re_df:bool=True):
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
        bins_df = pd.concat(self.bins, ignore_index=True)

        xs = [re.sub('_woe$', '', i) for i in self.model_feature]
        # coefficients
        coef_df = pd.Series(self.model.coef_[0], index=np.array(xs)).loc[lambda x: x != 0]  # .reset_index(drop=True)

        # scorecard
        basepoints = A - B * self.model.intercept_[0]
        card = {}
        card['baseScore'] = pd.DataFrame({'variable': 'basepoints', 'bin': '基础分', 'points': round(basepoints, 2)},index=[0])
        for i in coef_df.index:
            card[i] = bins_df.loc[bins_df['variable'] == i, ['variable', 'bin', 'woe']] \
                .assign(points=lambda x: round(-B * x['woe'] * coef_df[i], 2))[['variable', 'bin', 'points']]

        # 转换为df
        df = pd.DataFrame()
        for col in card.keys():
            col_df = card[col]
            df = pd.concat([df,col_df])
        df.set_index(['variable'], inplace=True)

        if re_df:
            return df
        else:
            return card

    def pred_score(self,df_woe:pd.DataFrame,only_total_score=True):
        y_prob = self.model.predict_proba(df_woe[self.model_feature].values)[:,1]
        score = Prob2Score(prob=y_prob,basePoint=self.score0,PDO=self.pdo)
        df_woe['score'] = score
        if only_total_score:
            return df_woe['score']
        else:
            return df_woe

    def apply_score(self,df,**kwargs):
        df_score = scorecard_ply(dt=df,card=self.score_card(re_df=False),**kwargs)
        return df_score