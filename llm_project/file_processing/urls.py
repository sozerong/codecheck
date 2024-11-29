from django.urls import path
from .views import FileProcessingAPIView

urlpatterns = [
    path("process/", FileProcessingAPIView.as_view(), name="file_processing"),
]
