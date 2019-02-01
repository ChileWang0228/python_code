#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
运行环境：Anaconda python 3.7.2
@Created on 2019-1-18 09:47
@Author:ChileWang
@algorithm：
"""
import pandas as pd
from pandas.io.stata import StataReader, StataWriter
import os
data_2010_dir = '/home/chilewang/Desktop/Data/2010/'

# stata_data = StataReader(file_name, convert_categoricals=False)
# print(list(stata_data.value_labels().keys()))
# print(type(list(stata_data.value_labels().keys())))
# print(type(pd.DataFrame(stata_data.read())))
# fmtlist = stata_data.fmtlist
# print(fmtlist)
# variable_labels = stata_data.variable_labels()
# print(variable_labels.keys())


def get_data_file_name(dir):
    """
    获取数据目录文件名
    :param dir:
    :return:
    """
    file_name_list = os.listdir(dir)
    print(file_name_list)
    return file_name_list


def read_stata_file(dir, file_name):
    """
    :param dir: stata文件存放目录
    :param file_name:
    :return:返回DataFrame格式和特征表
    """
    stata_data = StataReader(dir + file_name, convert_categoricals=False)
    columns_list = list(stata_data.value_labels().keys())  # 列
    print(file_name)
    print(len(columns_list))
    print(columns_list[0:10])
    print('---------------')
    return pd.DataFrame(stata_data.read()), columns_list


def process_stata_file(dir, file_name_list):
    """
    将目标目录中的多个stata表合成一张表，并返回DataFrame
    :return:生成ｃｓｖ文件
    """
    feature_id = ['fid', 'pid']  # 基于家庭和个人id合并表格
    and_columns_set = set(read_stata_file(dir, file_name_list[0])[1]) | (set(read_stata_file(dir, file_name_list[1])[1]))
    result = pd.merge(read_stata_file(dir, file_name_list[0])[0],
                      read_stata_file(dir, file_name_list[1])[0], on=feature_id, how='left')
    for i in range(2, len(file_name_list)):
        and_columns_set = and_columns_set | set(read_stata_file(dir, file_name_list[i])[1])
        # result = pd.merge(result, read_stata_file(dir, file_name_list[i])[0], on=feature_id, how='left')
    print(len(and_columns_set))
    # print(result.head(5))


if __name__ == '__main__':
    file_name_list = get_data_file_name(data_2010_dir)
    process_stata_file(data_2010_dir, file_name_list)

