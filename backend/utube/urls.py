from django.urls import path
from . import views
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('login/', views.LoginView, name='login'),
    path('upload_video/', views.upload_video, name='upload_video'),
    path('get_videos/', views.get_all_videos, name='get_videos'),
     path('speech_to_text/', views.speech_to_text, name="speech_to_text"),
        path('generate_description/', views.generate_description, name='generate_description'),

]
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)