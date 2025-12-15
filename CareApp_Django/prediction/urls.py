from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('about/', views.about, name='about'),
    path('predict/', views.predict_form, name='predict'),
    path('analysis/', views.analysis, name='analysis'),
    path('get_subcounties/<str:county>/', views.get_subcounties, name='get_subcounties'),
    path('chat/', views.chat, name='chat'),
    path('result/', views.result, name='result'),
]

