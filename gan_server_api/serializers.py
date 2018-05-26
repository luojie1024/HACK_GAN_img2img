# -*- coding: utf-8 -*-
'''
# @Time    : 5/23/18 11:14 AM
# @Author  : luojie
# @File    : serializers.py.py
# @Desc    : 
'''

from django.contrib.auth.models import User, Group
from rest_framework import serializers


class UserSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = User
        fields = ('url', 'username', 'email', 'groups')


class GroupSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = Group
        fields = ('url', 'name')
