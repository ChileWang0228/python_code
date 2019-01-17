#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Operating system: Ubuntu18.04
@编译环境：python3.7
@Created on 2019-01-09 23:46
@Author:ChileWang
"""
from django.shortcuts import render


def search_post(request):
    # 接受post请求数据
    ctx = {}
    if request.POST:
        ctx['rlt'] = request.POST['q']
    return render(request, 'post.html', ctx)
