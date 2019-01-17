#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Operating system: Ubuntu18.04
@编译环境：python3.7
@Created on 2019-01-09 23:46
@Author:ChileWang
"""

from django.db import models

# Create your models here.


class Django(models.Model):  # 类名代表数据库表名
    name = models.CharField(max_length=20)
    # 代表数据表中的字段数据类型则由CharField（相当于varchar）
    # DateField（相当于datetime）， max_length 参数限定长度。


class Contact(models.Model):
    name = models.CharField(max_length=20)
    age = models.CharField(default=0, max_length=10)
    email = models.EmailField()

    def __unicode__(self):
        return self.name


class Tag(models.Model):
    contact = models.ForeignKey(Contact, on_delete=models.CASCADE)  # 作为表contact的外键
    """
    在django2.0后，定义外键和一对一关系的时候需要加on_delete选项，此参数为了避免两个表里的数据不一致问题，
    不然会报错：
    TypeError: __init__() missing 1 required positional argument: 'on_delete'
    """
    name = models.CharField(max_length=50)

    def __unicode__(self):
        return self.name

