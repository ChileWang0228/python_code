"""HelloWorld URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/2.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
# from django.contrib import admin
# from django.urls import path
from . import test_db
from django.conf.urls import url
from . import view, search, search_post
from django.contrib import admin
urlpatterns = [
    url('admin/', admin.site.urls),
    url(r'^hello$', view.hello),
    url(r'^test_db$', test_db.test_db),
    url(r'^test_db1$', test_db.get_data),
    url(r'^test_db2$', test_db.modified_data),
    url(r'^test_db3$', test_db.delete_data),
    url(r'^search$', search.search),  # search是search.html的action属性
    url(r'^search_form$', search.search_form),
    url(r'^search-post$', search_post.search_post)  # search-post是post.html的action属性
]
