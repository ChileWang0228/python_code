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
file_name = '/home/chilewang/Desktop/2010/CFPS_2010_adult_dta/2010adult_072016.dta'

stata_data = StataReader(file_name, convert_categoricals=False)
print(list(stata_data.value_labels().keys()))
print(type(list(stata_data.value_labels().keys())))
print(type(pd.DataFrame(stata_data.read())))
fmtlist = stata_data.fmtlist
print(fmtlist)
variable_labels = stata_data.variable_labels()
print(variable_labels.keys())
