"""
URL configuration for backend project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/6.0/topics/http/urls/
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
from django.conf import settings
from django.conf.urls.static import static
from prediction import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.home, name='home'),
    path('register/', views.register_view, name='register'),
    path('login/', views.login_view, name='login'),
    path('logout/', views.logout_view, name='logout'),
    path('dashboard/', views.dashboard, name='dashboard'),
    path('patients/', views.patients_list, name='patients_list'),
    path('patients/add/', views.add_patient, name='add_patient'),
    path('patients/<int:patient_id>/', views.patient_detail, name='patient_detail'),
    path("patients/<int:patient_id>/edit/", views.edit_patient, name="edit_patient"),
    path("patients/<int:patient_id>/delete/", views.delete_patient, name="delete_patient"),
    path('patients/<int:patient_id>/add-note/', views.add_medical_note, name='add_medical_note'),
    path('predictions/<int:note_id>/', views.prediction_result, name='prediction_result'),
    path('profile/', views.profile_view, name='profile'),
    path('notes/<int:note_id>/predict/', views.prediction_result, name='prediction_result'),

]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
