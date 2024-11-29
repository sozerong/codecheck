from django.urls import path
from .views import AIEvaluationAPIView

urlpatterns = [
    path("evaluate/", AIEvaluationAPIView.as_view(), name="ai_evaluation"),
]
