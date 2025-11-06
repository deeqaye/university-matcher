from django.contrib import admin
from django.urls import path, include
from . import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('universities/', include('apps.universities.urls')),
    path('gemini/', include('apps.gemini.urls')),
    path('', views.home, name='home'),
]