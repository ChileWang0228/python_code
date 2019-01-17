#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Operating system: Ubuntu18.04
@编译环境：python3.7
@Created on 2019-01-09 23:46
@Author:ChileWang
"""
from django.http import HttpResponse
from django.shortcuts import render_to_response


def search_form(request):
    # 表单
    return render_to_response('search.html')


def search(request):
    # 接受请求数据
    request.encoding = 'utf-8'
    if 'q' in request.GET:  # ‘q’是表单的名字
        message = '你搜索的内容:' + request.GET['q']
    else:
        message = '你提交了空表单'
    return HttpResponse(message)
