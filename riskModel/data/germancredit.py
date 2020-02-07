# -*- coding: utf-8 -*-

import pandas as pd
import pkg_resources
import numpy as  np

class Germancredit(object):
    # 所有变量
    def __init__(self):
        self.all_feature = ['status_of_existing_checking_account', 'duration_in_month', 'credit_history', 'purpose', 'credit_amount',
                       'savings_account_and_bonds',
                       'present_employment_since', 'installment_rate_in_percentage_of_disposable_income',
                       'personal_status_and_sex', 'other_debtors_or_guarantors',
                       'present_residence_since', 'property', 'age_in_years', 'other_installment_plans', 'housing',
                       'number_of_existing_credits_at_this_bank', 'job',
                       'number_of_people_being_liable_to_provide_maintenance_for', 'telephone', 'foreign_worker']
        # 类别变量
        self.cat_col = ['purpose', 'personal_status_and_sex', 'other_debtors_or_guarantors', 'property', 'other_installment_plans',
                   'housing', 'telephone', 'foreign_worker']
        # 数值变量
        self.num_col = [i for i in self.all_feature if i not in self.cat_col]
        # 整型变量
        self.int_col = self.cat_col + ['status_of_existing_checking_account', 'credit_history', 'savings_account_and_bonds', 'job']
        # 需要处理的数值变量
        self.sub_col = ['status_of_existing_checking_account', 'credit_history', 'savings_account_and_bonds',
                   'present_employment_since', 'job']
        # 需要替换的变量
        self.rep_dict = {
            'status_of_existing_checking_account': {4: 0},
            'savings_account_and_bonds': {5: np.nan},
            'purpose': {'A124': np.nan}
        }

    def get_data(self):
        DATA_FILE = pkg_resources.resource_filename('riskModel', 'data/germancredit.csv')
        data = pd.read_csv(DATA_FILE,encoding='utf-8', sep=' ', header=None,names= self.all_feature + ["target"])
        return data

    def get_describe(self):
        DATA_FILE = pkg_resources.resource_filename('riskModel', 'data/german.txt')
        with open(DATA_FILE,'r+') as f:
            print(f.read())
