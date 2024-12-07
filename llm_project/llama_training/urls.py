from django.urls import path
from .views import StarCoderTrainingAPIView

urlpatterns = [
    path("train/", StarCoderTrainingAPIView.as_view(), name="llama3_training"),
]
