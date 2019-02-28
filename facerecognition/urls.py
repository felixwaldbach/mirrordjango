from django.urls import path

from . import api

urlpatterns = [
    path('recognizeimage', api.handle_recognize_image, name='recognizeimage'),
    path('storetrain', api.handle_store, name='storetrain')
]
