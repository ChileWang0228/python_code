from django.contrib import admin
from .models import Question
# Register your models here.
admin.site.register(Question)  # 将Question注册到管理页面
