#!/usr/bin/env bash
python manage.py migrate  # 创建(迁移)mysql数据库
python manage.py makemigrations polls  # 运行该命令让Django包含polls应用,激活创建的模型
python manage.py migrate polls # 创建表结构