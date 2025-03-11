from django.urls import path
from . import views
from django.conf import settings
from django.conf.urls.static import static
from .views import like_video, dislike_video, video_like_count,add_comment,summarize_video,log_video_watch,user_watch_history,suggested_videos,LoginView


urlpatterns = [
    path('login/', LoginView, name='login'),  # ✅ Ensure this is correctly defined
    path('upload_video/', views.upload_video, name='upload_video'),
    path('get_videos/', views.get_all_videos, name='get_videos'),
    path('speech_to_text/', views.speech_to_text, name="speech_to_text"),
    path('generate_description/', views.generate_description, name='generate_description'),
    path("like/", like_video, name="like_video"),
    path("dislike/", dislike_video, name="dislike_video"),
    path("video/<str:video_title>/likes/", video_like_count, name="video_like_count"),
    path('comment/', add_comment, name='add_comment'),
    path('summarize/', summarize_video, name='summarize_video'),
    path('video/watch/', log_video_watch, name='log_video_watch'),
    path("video/history/<str:enrollment>/", user_watch_history, name="user_watch_history"),
    path("video/suggested/", suggested_videos, name="suggested-videos"),




]
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)