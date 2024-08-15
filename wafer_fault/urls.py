from django.urls import path
from . import views
from app_tracking.logger import App_Logger
from app_tracking.exception import AppException

app_name = 'wafer_fault'

urlpatterns = [
    path('', views.home, name='home'),

]
