# -*- coding: utf-8 -*-

#############################################
# Author: huangsir
# Mail: hxysir@163.com
# Created Time:  2019-10-22 18:00
#############################################

from setuptools import setup,find_packages

setup(
    name = "riskModel",
    version = "1.0.2",
    author="huangsir",
    author_email="hxysir@163.com",
    url = "https://github.com/huangxianyang/risk-model",
    license = "MIT License",
    description = "build credit score model",
    long_description = "build rsik score model, including etl, eda, preprocessing, featureengineering, trainmodel, riskstragety eg.",
    platforms = "any",
    keywords = ["pip", "EDA","Preprocessing", "FeatureEngineering", "Modeling","RiskStragety"],

    packages = find_packages(),# 所有包含__init__.py文件的目录
    package_data={'riskModel':['data/*.csv','data/*.txt']},
    # include_package_data = True,
    install_requires = ['pymongo>=3.7.2','PyMySQL>=0.9.3','missingno>=0.4.1','pandas-profiling>=1.4.1','numpy>=1.15.0',
                        'matplotlib>=3.0.2','pandas>=0.23.4','scikit-learn>=0.21.3','imbalanced-learn==0.5.0',
                        'statsmodels>=0.9.0','scorecardpy>=0.1.7','seaborn>=0.9.0','bayesian-optimization>=1.0.1','scipy>=1.3.1'
                        ]
)