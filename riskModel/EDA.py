# -*- coding: utf-8 -*-
# @Time    : 2018-01-21 11:22
# @Author  : HuangSir
# @FileName: EDA.py
# @Software: PyCharm

"""
Exploratory analysis
"""

import pandas as pd
import pandas_profiling

from toad import detect, quality

class BasicStat:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def get_report(self):
        '''
        get data report of html
        '''
        # warnings.warn('there are costing much time')
        profile = pandas_profiling.ProfileReport(self.df)
        report = profile.to_html()
        print('finish')
        return report

    def get_describe(self):
        '''
        get data describe of DataFrame
        '''
        report = detect(self.df)
        return report

    def get_quality(self, target='target', iv_only=False, **kwargs):
        report = quality(self.df, target, iv_only, **kwargs)
        return report
