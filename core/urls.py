from django.contrib import admin
from django.urls import path

from .views import sentiment_analysis_view

urlpatterns = [
    path('', sentiment_analysis_view, name='sentiment_analysis'),
    path("admin/", admin.site.urls),
]
