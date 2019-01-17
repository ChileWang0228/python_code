#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Operating system: Ubuntu18.04
@编译环境：python3.7
@Created on 2019-01-09 23:46
@Author:ChileWang
@algorithm：test database
"""

from django.http import HttpResponse
from TestModel.models import Django


# 数据库操作
def test_db(requset):
    test_1 = Django(name='chile')  # 插入名字
    test_1.save()
    return HttpResponse("<p>插入数据成功</p>")


def get_data(request):
    response = ''
    # 通过objects获取所有行
    data_list = Django.objects.all()

    # 条件过滤,相当于where
    response1 = Django.objects.filter(id=1)

    # 获取单个对象
    response2 = Django.objects.get(id=1)

    # 限制返回数据，相当于sql中的offset 0 limit 2
    Django.objects.order_by('name')[0: 2]

    # 数据排序
    Django.objects.order_by()

    # 上面方法连锁使用
    Django.objects.filter(name="chile").order_by("id")

    # 输出所有数据
    for var in data_list:
        response += var.name + ' '

    return HttpResponse("<p>" + response + "</p>")


def modified_data(request):
    # 修改其中的字段， 再save，相当于sql的update
    data = Django.objects.get(id=1)
    data.name = 'ChileWang'
    data.save()
    # 另外一种方式
    # Test.objects.filter(id=1).update(name='Google')

    # 修改所有的列
    # Test.objects.all().update(name='Google')
    return HttpResponse("<p>修改成功</p>")


def delete_data(request):
    Django.objects.filter(name='ChileWang').delete()

    # 删除所有数据
    # Django.objects.all().delete()

    # 删除id=1的数据
    # test1 = Django.objects.get(id=1)
    # test1.delete()
    return HttpResponse("<p>删除成功</p>")
