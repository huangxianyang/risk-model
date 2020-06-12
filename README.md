## *riskModel* 
#### 风险评分模型
## *Install*
#### pip install riskModel
=======
### *pip install riskModel*
***
## *主要功能*
- 数据探索分析 (`数据概述BasicStat`)
- 数据清洗  Preprocess(`异常值检测--异常值处理--众数处理--缺失处理`)
- 数据采样  DataSample(`随机平衡采样, Borderline-SMOTE过采样`)
- 特征离散化,特征工程 cat_bin,num_bin(`无序类别变量分箱组合, 有序数值变量分箱组合,支持单调性检验合并`)
- 特征选择 SelectFeature(`基于IV信息值, 基于共线性, 基于VIF方差膨胀因子, 基于L1正则化`)
- 模型训练 lr(`lr`)
- 评分卡模型 ScoreCard(`标准评分卡转换,模型预测[概率预测,评分预测]`)
- 风险决策 (`风险策略,稳定性评估PSI,评分决策表,KS`) 
***
```python
# -*- coding: utf-8 -*-
"""
评分卡示例:
0.数据探索分析,1.数据预处理,2.特征分箱,3.特征选择,4.模型训练,5.评分卡构建,6.模型评估,7.风险决策
"""

import scorecardpy as sc

import joblib
import riskModel as rs
from  sklearn.model_selection import train_test_split

path = './result/'
# 导入数据
germanCredit = rs.Germancredit()
df = germanCredit.get_data()
germanCredit.get_describe()
df.sample(5)
# 数值化
for col in germanCredit.sub_col:
    df[col] = df[col].apply(lambda x:int(str(x)[1:]))
# 替换
df = df.replace(germanCredit.rep_dict)
df['target'] = df['target'].replace({1:0,2:1})

all_cols = germanCredit.all_feature
num_cols = germanCredit.num_col
cat_cols = germanCredit.cat_col

#######################################
# 0.数据探索分析
#######################################
bs = rs.BasicStat(df=df)
html_report = bs.get_report()
# 保存
with open(path+'dt_report.html','w',encoding='utf8') as f:
    f.write(html_report)

df_report = bs.get_describe()
df_report.to_excel(path+'df_report.xlsx')

#######################################
# 1.数据清洗
#######################################
pr = rs.Preprocess()
str_s = pr.find_str(df=df,cols=num_cols)
print('字符:',str_s)
sign_s = pr.find_sign(df=df,cols=all_cols)
print('特殊符号:',sign_s)
# 离群值
outliers = pr.outlier_drop(df=df,cols=all_cols,low_percent=0.001,up_percent=0.999,cat_percent=0.01)
print('离群值:',outliers)
# 离群值处理
for k,v in outliers.items():
    if type(v) == dict:
        df[k] = df[k].apply(lambda x:v.get('low') if x < v.get('low') else v.get('up') if x > v.get('up') else x)
    elif type(v) == list:
        df[k] = df[k].replace(v,df[k].mode()[0])

# 缺失统计
missm_result = pr.miss_mode_cnt(df=df,cols=all_cols)
print(missm_result)
# 连续变量删除缺失80%+,众数率90%+
# 分类变量删除缺失率90%,众数率95%+
num_cols = [i for i in num_cols if missm_result['miss'][i] < 0.8 and missm_result['mode'][i] < 0.8]
cat_cols = [i for i in cat_cols if missm_result['miss'][i] < 0.8 and missm_result['mode'][i] < 0.95]
print(f'drop num cols :{[i for i in all_cols if i not in num_cols+cat_cols]}')
all_cols = num_cols + cat_cols
# 缺失填充
df = pr.fill_nan(df=df,cols=all_cols,values=-99999,min_miss_rate=0.05)

# 字符变量数值化
df,factorize_dict = pr.factor_map(df=df,cols=cat_cols)
# 过采样
int_cols = cat_cols + [i for i in num_cols if i in germanCredit.int_col]
print('int_cols:',int_cols)

ds = rs.DataSample(df=df,cols=all_cols,target='target',sampling_strategy='minority')
df_res = ds.over_sample(int_cols=int_cols)
# 分类变量还原
to_factorize = {k: {i:j for j,i in v.items()} for k,v in factorize_dict.items()}
df_res = df_res.replace(to_factorize)
df = df.replace(to_factorize)
########################################
# 2.特征工程
########################################
# 样本划分
# woe bin
train,valid = train_test_split(df_res,test_size=0.3,random_state=0)
cat_bin,cat_iv = rs.cat_bin(df=train,cols=cat_cols,target='target',specials=['-99999'],
                            bin_num_limit=5,count_distr_limit=0.05,method='chimerge')
num_bin,num_iv = rs.num_bin(df=train,cols=num_cols,target='target',specials=[-99999],
                            bin_num_limit=8,count_distr_limit=0.05,sc_method='chimerge',
                            non_mono_cols=['age_in_years'],init_bins=15,init_min_samples=0.05,init_method='chi')

bins = {**cat_bin,**num_bin}
ivs = {**cat_iv,**num_iv}
# woe转换
train_woe = sc.woebin_ply(dt=train,bins=bins)
valid_woe = sc.woebin_ply(dt=valid,bins=bins)
df_woe = sc.woebin_ply(dt=df,bins=bins)

#########################################
# 3.特征选择
#########################################
sf = rs.SelectFeature()
high_iv = sf.baseOn_iv(ivd=ivs,thred=0.05,is_draw=False)
low_vif = sf.baseOn_collinear(df=train_woe,high_iv=high_iv,thred=0.7,is_draw=False)
ml_cols, best_C = sf.baseOn_l1(X=train_woe[low_vif.keys()],y=train_woe['target'],Kfold=5,drop_plus=False)

#########################################
# 4.模型训练
#########################################

lr = rs.Lr(C=best_C)
lr.fiting(X=train_woe[ml_cols.keys()],y=train_woe['target'],filename=path)
joblib.dump(lr,path+'lr.pkl')

########################################
# 5.评分卡构建
########################################

card = rs.ScoreCard(lr=lr,bins=bins,ml_cols=[i.replace('_woe','') for i in ml_cols.keys()],
                    score0=600,pdo=50)
card_df = card.score_card()
joblib.dump(card,path+'card.pkl')
########################################
# 6.模型评估,真实样本评估
#######################################
valid['prob'] = lr.predict_proba(valid_woe[ml_cols.keys()])[:,1]
df['prob'] = lr.predict_proba(df_woe[ml_cols.keys()])[:,1]

print(f'valid evaluate:{rs.model_norm(valid["target"],valid["prob"])}')
print(f'test evaluate:{rs.model_norm(df["target"],df["prob"])}')

valid['score'] = card.apply_score(valid)
df['score'] = card.apply_score(df)

########################################
# 7.风险决策
#######################################
cut_off = rs.stragety_score(df=df,step=50,score='score',label='target')
cut_off.to_excel(path+'cut_off.xlsx')

```

