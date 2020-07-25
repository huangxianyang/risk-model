# -*- coding: utf-8 -*-

#############################################
# Author: huangsir
# Mail: hxysir@163.com
# Created Time:  2019-10-22 18:00
#############################################

from setuptools import setup,find_packages

setup(
    name = "riskModel",
    version = "1.0.6",
    author="huangsir",
    author_email="hxysir@163.com",
    url = "https://github.com/huangxianyang/risk-model",
    license = "MIT License",
    description = "build credit score model",
    long_description = "build rsik score model, including eda, preprocessing, featureengineering, trainmodel, riskstragety eg.",
    platforms = "any",
    keywords = ["pip", "EDA","Preprocessing", "FeatureEngineering", "Modeling","RiskStragety"],

    packages = find_packages(),# 所有包含__init__.py文件的目录
    package_data={'riskModel':['data/*.csv','data/*.txt']},
    # include_package_data = True,
    install_requires = ['pandas-profiling>=2.4.0','numpy>=1.18.1',
                        'matplotlib>=3.1.3','pandas>=1.0.3','scikit-learn>=0.23.1','imbalanced-learn==0.5.0',
                        'statsmodels>=0.11.1','scorecardpy>=0.1.9.2','seaborn>=0.10.1','scipy>=1.4.1','toad>=0.0.60'
                        ]
)