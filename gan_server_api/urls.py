# -*- coding: utf-8 -*-
'''
# @Time    : 5/23/18 11:43 AM
# @Author  : luojie
# @File    : urls.py
# @Desc    : 
'''
from django.conf.urls import url
from gan_server_api import views

urlpatterns = [
    url(r'^facades_B2A/$', views.facades_B2A),
    url(r'^shoes_B2A/$', views.shoes_B2A),
    url(r'^handbags_B2A/$', views.handbags_A2B),
    url(r'^cityscapes_A2B/$', views.cityscapes_A2B),
    url(r'^maps_A2B/$', views.maps_A2B),
    url(r'^cityscapes_B2A/$', views.cityscapes_B2A),
    url(r'^maps_B2A/$', views.maps_B2A),
]
