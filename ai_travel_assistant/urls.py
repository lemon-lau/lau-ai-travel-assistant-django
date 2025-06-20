"""
URL configuration for ai_travel_assistant project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path

from .views import (
    health_check_view,
    scrape_view,
    chat_view,
    history_view
)

urlpatterns = [
    path("", health_check_view, name="health_check"),
    path('admin/', admin.site.urls),
    path("scrape/", scrape_view, name="scrape"),
    path("chat/", chat_view, name="chat"),
    path("history/", history_view, name="history"),
]
