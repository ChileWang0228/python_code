#!/bin/sh
python manage.py makemigrations TestModel # 让Django知道我们的模型有一些变更
python manage.py migrate TestModel  # 创建表结构



