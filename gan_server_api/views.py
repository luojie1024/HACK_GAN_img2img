from django.shortcuts import render

# Create your views here.
from django.contrib.auth.models import User, Group
from rest_framework import viewsets
from gan_server_api.serializers import UserSerializer, GroupSerializer
from django.http import HttpResponse
from rest_framework.renderers import JSONRenderer
from rest_framework.parsers import JSONParser
import base64
from GAN.Predictor import Predictor


class UserViewSet(viewsets.ModelViewSet):
    """
    API endpoint that allows users to be viewed or edited.
    """
    queryset = User.objects.all().order_by('-date_joined')
    serializer_class = UserSerializer


class GroupViewSet(viewsets.ModelViewSet):
    """
    API endpoint that allows groups to be viewed or edited.
    """
    queryset = Group.objects.all()
    serializer_class = GroupSerializer


class JSONResponse(HttpResponse):
    """
    用于返回JSON数据.
    """

    def __init__(self, data, **kwargs):
        content = JSONRenderer().render(data)
        kwargs['content_type'] = 'application/json'
        super(JSONResponse, self).__init__(content, **kwargs)


# facades_B2A
# @csrf_exempt
def facades_B2A(request):
    if request.method == 'POST':
        datas = JSONParser().parse(request)
        base64data = datas['imageBase64']
        # 写图片
        imgdata = base64.b64decode(base64data)
        file_write(imgdata, 'Facades')
        # 預測者
        prediction_facades = Predictor('Facades')
        # 预测
        orgin_result = prediction_facades.Preediction(base64data)

        result = base64.b64encode(orgin_result)

        return JSONResponse({'imageBase64': result})
    else:
        return JSONResponse({'imageBase64': 'data_error'})


def handbags_A2B(request):
    if request.method == 'POST':
        datas = JSONParser().parse(request)
        base64data = datas['imageBase64']
        # 写图片
        imgdata = base64.b64decode(base64data)
        file_write(imgdata, 'Handbags')
        # 預測者
        prediction_facades = Predictor('Handbags')
        # 预测
        orgin_result = prediction_facades.Preediction(base64data)

        result = base64.b64encode(orgin_result)

        return JSONResponse({'imageBase64': result})

    else:
        return JSONResponse({'imageBase64': 'data_error'})


def shoes_B2A(request):
    if request.method == 'POST':
        datas = JSONParser().parse(request)
        base64data = datas['imageBase64']
        # 写图片
        imgdata = base64.b64decode(base64data)
        file_write(imgdata, 'Shoes')
        # 預測者
        prediction_facades = Predictor('Shoes')
        # 预测
        orgin_result = prediction_facades.Preediction(base64data)

        result = base64.b64encode(orgin_result)
        return JSONResponse({'imageBase64': result})
    else:
        return JSONResponse({'imageBase64': 'data_error'})


def maps_A2B(request):
    if request.method == 'POST':
        datas = JSONParser().parse(request)
        base64data = datas['imageBase64']
        # 写图片
        imgdata = base64.b64decode(base64data)
        file_write(imgdata, 'MapsA2B')
        # 預測者
        prediction_facades = Predictor('MapsA2B')
        # 预测
        orgin_result = prediction_facades.Preediction(base64data)

        result = base64.b64encode(orgin_result)
        return JSONResponse({'imageBase64': result})
    else:
        return JSONResponse({'imageBase64': 'data_error'})


def maps_B2A(request):
    if request.method == 'POST':
        datas = JSONParser().parse(request)
        base64data = datas['imageBase64']
        # 写图片
        imgdata = base64.b64decode(base64data)
        file_write(imgdata, 'MapsB2A')
        # 預測者
        prediction_facades = Predictor('MapsB2A')
        # 预测
        orgin_result = prediction_facades.Preediction(base64data)

        result = base64.b64encode(orgin_result)
        return JSONResponse({'imageBase64': result})
    else:
        return JSONResponse({'imageBase64': 'data_error'})


def cityscapes_A2B(request):
    if request.method == 'POST':
        datas = JSONParser().parse(request)
        base64data = datas['imageBase64']
        # 写图片
        imgdata = base64.b64decode(base64data)
        file_write(imgdata, 'CitysA2B')
        # 預測者
        prediction_facades = Predictor('CitysA2B')
        # 预测
        orgin_result = prediction_facades.Preediction(base64data)

        result = base64.b64encode(orgin_result)
        return JSONResponse({'imageBase64': result})
    else:
        return JSONResponse({'imageBase64': 'data_error'})


def cityscapes_B2A(request):
    if request.method == 'POST':
        datas = JSONParser().parse(request)
        base64data = datas['imageBase64']
        # 写图片
        imgdata = base64.b64decode(base64data)
        file_write(imgdata, 'CitysB2A')
        # 預測者
        prediction_facades = Predictor('CitysB2A')
        # 预测
        orgin_result = prediction_facades.Preediction(base64data)

        result = base64.b64encode(orgin_result)
        return JSONResponse({'imageBase64': result})
    else:
        return JSONResponse({'imageBase64': 'data_error'})


def file_write(data, image_name):
    file = open('cache/' + image_name + '.png', 'wb')
    file.write(data)
    file.close()
