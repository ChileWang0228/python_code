from django.contrib import admin
from TestModel.models import Django, Contact, Tag
# Register your models here.
admin.site.register([Django, Contact, Tag])
