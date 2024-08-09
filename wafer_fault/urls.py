from django.urls import path
from . import views

app_name = 'wafer_fault'

urlpatterns = [
    path('', views.home, name='home'),

]
