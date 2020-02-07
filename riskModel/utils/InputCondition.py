# -*- coding: utf-8 -*-
"""
入参合法性检验
"""

import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype


# check str
def check_str(params):
    if params is not None:
        if not isinstance(params, str):
            raise Exception("Incorrect inputs; parameter should be a str.")
    return params


# check bool
def check_bool(params):
    if params is not None:
        if not isinstance(params, bool):
            raise Exception("Incorrect inputs; parameter should be a bool.")
    return params


# check float
def check_float(params):
    if params is not None:
        if not isinstance(params, float):
            raise Exception("Incorrect inputs; parameter should be a float.")
    return params


# check int
def check_int(params):
    if params is not None:
        if not isinstance(params, int):
            raise Exception("Incorrect inputs; parameter should be an int.")
    return params


# check tuple
def check_tuple(params):
    if params is not None:
        if not isinstance(params, tuple):
            raise Exception("Incorrect inputs; parameter should be a tuple.")
    return params


# check list
def check_list(params):
    if params is not None:
        if not isinstance(params, list):
            raise Exception("Incorrect inputs; parameter should be a list.")
    return params


# check dict
def check_dict(params):
    if params is not None:
        if not isinstance(params, dict):
            raise Exception("Incorrect inputs; parameter should be a dict.")
    return params


# check df
def check_df(params):
    if params is not None:
        if not isinstance(params, pd.DataFrame):
            raise Exception("Incorrect inputs; parameter should be a DataFrame.")
    return params

# check like-list
def check_array(params):
    if params is not None:
        if not isinstance(params,np.ndarray):
            raise Exception("Incorrect inputs; parameter should be a ndarray.")
    return params

# check str or list
def check_str_list(params):
    if params is not None:
        # is string
        if isinstance(params, str):
            param_list = eval(params)
        # is not list
        if not isinstance(params, list):
            raise Exception("Incorrect inputs; parameter should be a list or str.")
    return params


# check list-like of float 默认 None,[0.25,0.5,0.75]
def check_float_list(params):
    new_list = []
    if params is not None:
        # is not list
        if not isinstance(params, list):
            raise Exception("Incorrect inputs; parameter should be a list-like of float.")
        else:
            for each in params:
                if not isinstance(each, float) and not isinstance(each,int):
                    new_list.append(each)
            if new_list:
                raise Exception("Incorrect inputs; parameter should be a list-like of float.")
    return params


# check list-like of dtypes or None
def check_dtypes_list(params):
    if params is not None:
        if params=='all':
            return params
        # is not list
        if not isinstance(params, list):
            raise Exception("Incorrect inputs; parameter should be a list-like of dtypes.")
        else:
            for each in params:
                if not is_numeric_dtype(each):
                    raise Exception("Incorrect inputs; parameter should be a list-like of dtypes.")
    return params


# check special_values: list, dict
def check_special_values(special_values):
    if special_values is not None:
        if not isinstance(special_values, list) and not isinstance(special_values, dict):
            raise Exception("Incorrect inputs; special_values should be a list or dict.")
    return special_values


# check list of dict
def check_list_of_dict(params):
    new_list = []
    if params is not None:
        if not isinstance(params, dict):
            raise Exception("Incorrect inputs; special_values should be a list of dict.")
        else:
            for each in params.values():
                if not isinstance(each, list) and not isinstance(each,tuple):
                    new_list.append(each)
            if new_list:
                raise Exception("Incorrect inputs; parameter should be a list(tuple) of dict.")
    return params


# check dict of df
def check_dict_of_df(params):
    new_list = []
    if params is not None:
        if not isinstance(params, dict):
            raise Exception("Incorrect inputs; special_values should be a dict of df.")
        else:
            for each in params.values():
                if not isinstance(each, pd.DataFrame):
                    new_list.append(each)
            if new_list:
                raise Exception("Incorrect inputs; parameter should be a dict of df.")
    return params

# check dict of df
def check_dict_of_float(params):
    new_list = []
    if params is not None:
        if not isinstance(params, dict):
            raise Exception("Incorrect inputs; special_values should be a dict of float.")
        else:
            for each in params.values():
                if not isinstance(each, float):
                    new_list.append(each)
            if new_list:
                raise Exception("Incorrect inputs; parameter should be a dict of float.")
    return params