# -*- coding: utf-8 -*-

__version__ = '1.0.2'

# 类
from .ETL import DbRead  # 从数据库中读取数据
from .EDA import BasicStat,Plot # 数据探索分析
from .Preprocessing import Preprocess,DataSample # 数据清洗, 数据抽样
from .FeatureEngineering import FeatureBin,WoeBinPlot,SelectFeature # 特征分箱,特征分箱可视化,特征选择
from .Modeling import SearchParam,TrainLr,ScoreCard # 超参调优,模型训练,评分卡构建
from .utils.PltFunction import PlotModel # 模型作图,ks,roc曲线
from .data.germancredit import Germancredit # 导入数据

# 方法
from .FeatureEngineering import woe_trans # woe映射
from .utils.tools import model_norm,Prob2Score # 模型指标,概率转换为分数
from .RiskStragety import caculate_ks,stragety_score,score_psi,var_psi # ks计算,决策报告,评分psi,变量psi

