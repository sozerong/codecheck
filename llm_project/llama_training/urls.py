from django.urls import path
from .views import Llama3TrainingAPIView

urlpatterns = [
    path("train/", Llama3TrainingAPIView.as_view(), name="llama3_training"),
]
