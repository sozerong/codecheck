from django.urls import path
from .views import DataPreprocessingAPIView

urlpatterns = [
    path("preprocess/", DataPreprocessingAPIView.as_view(), name="data_preprocessing"),
]
