from django.db import models

# Create your models here.
# 创建数据库表格


class Question(models.Model):  # 数据库表名:问题
    question_text = models.CharField(max_length=200)  # 字段：问题描述
    pub_date = models.DateTimeField('date published')  # 字段：问题发布日期


class Choice(models.Model):  # 数据库吧表名:选项
    question = models.ForeignKey(Question, on_delete=models.CASCADE)  # 问题(外键)
    choice_text = models.CharField(max_length=200)  # 选项内容
    votes = models.IntegerField(default=0)  # 投票




