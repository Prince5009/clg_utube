import os
import json
from django.shortcuts import get_object_or_404
import whisper
from transformers import pipeline
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.core.exceptions import ValidationError
from django.core.files.storage import default_storage
from django.db import connection
from .models import User, Video
from rest_framework.decorators import api_view
from .models import Video, VideoLike
from .models import Comment
from scenedetect import VideoManager, SceneManager, ContentDetector
import os
from django.core.files.storage import default_storage

from scenedetect import VideoManager, SceneManager, ContentDetector

from rest_framework.decorators import api_view
from rest_framework.response import Response
from .models import ShortClip
from fastapi import FastAPI, UploadFile, File
from django.core.files.base import ContentFile
import os
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .models import VideoViewLog, Video, User  # ✅ Correct 'User' instead of 'Users'
from .models import VideoLike, VideoViewLog
from django.db.models import Count
from tempfile import NamedTemporaryFile
import threading
from django.http import JsonResponse
from django.db.models import Count
from .models import Video, VideoViewLog, VideoLike  # ✅ Use correct model names

COMMON_HASHTAGS = "#YouTube #TrendingNow #ViralVideo #SubscribeNow #WatchTillEnd #Creators #DailyVlogs #Reaction #Unboxing #HowTo #Tutorial #Motivation #Fun #Live #Challenge #BehindTheScenes #ShortFilm #Podcast #DIY #Review"

app = FastAPI()
# User login view
@csrf_exempt
def LoginView(request):
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            enrollment = data.get("enrollment")
            password = data.get("password")

            print(f"Received Enrollment: {enrollment}")
            print(f"Received Password: {password}")

            # ✅ Use Django's database connection to check user credentials
            with connection.cursor() as cursor:
                cursor.execute("SELECT password FROM users WHERE enrollment = %s", [enrollment])
                user = cursor.fetchone()

            # ✅ Check if user exists and verify password
            if user:
                stored_password = user[0]  # Extract password from query result
                if password == stored_password:  # Direct comparison for plain-text passwords
                    return JsonResponse({"message": "Login successful", "status": "success"})
                else:
                    return JsonResponse({"message": "Invalid credentials", "status": "error"})
            else:
                return JsonResponse({"message": "User not found", "status": "error"})

        except json.JSONDecodeError:
            return JsonResponse({"message": "Invalid JSON format", "status": "error"}, status=400)

    return JsonResponse({"message": "Method Not Allowed", "status": "error"}, status=405)

# Video upload view
@csrf_exempt
def upload_video(request):
    if request.method == 'POST' and request.FILES.get('video_file'):
        video_file = request.FILES['video_file']
        video_name = request.POST.get('video_name', video_file.name)
        description = request.POST.get('description', '')  # Get the description from the request

        try:
            # Validate the video file format
            if not video_file.name.endswith(('.mp4', '.avi', '.mov', '.mkv')):
                raise ValidationError("Invalid video format. Please upload a valid video file.")

            # Save the video file
            file_path = default_storage.save(os.path.join('videos', video_file.name), video_file)
            video_url = default_storage.url(file_path)  # Get the URL of the uploaded file

            # Save video data in the database
            video = Video(video_name=video_name, description=description, video_file=file_path)
            video.save()

            return JsonResponse({
                "message": "Video uploaded successfully",
                "status": "success",
                "video_url": video_url
            })
        except ValidationError as e:
            return JsonResponse({"message": str(e), "status": "error"}, status=400)
        except Exception as e:
            return JsonResponse({"message": str(e), "status": "error"}, status=500)

    return JsonResponse({"message": "Method Not Allowed", "status": "error"}, status=405)





# Fetch all videos
def get_all_videos(request):
    try:
        videos = Video.objects.all()  # Retrieve all video records
        video_data = []

        # Loop through all videos and collect the video details
        for video in videos:
            video_url = default_storage.url(video.video_file)  # Generate the URL of the video
            video_data.append({
                "video_id": video.id,
                "video_name": video.video_name,
                "video_url": video_url
            })

        # Return the list of all videos as a JSON response
        return JsonResponse({
            "message": "Videos fetched successfully",
            "status": "success",
            "videos": video_data
        })

    except Exception as e:
        return JsonResponse({"message": str(e), "status": "error"}, status=500)

    try:
        video = Video.objects.get(id=video_id)
        video_url = default_storage.url(video.video_file)  # Generate the URL of the video
        
        return JsonResponse({
            "message": "Video fetched successfully",
            "status": "success",
            "video_url": video_url,
            "video_name": video.video_name
        })

    except Video.DoesNotExist:
        return JsonResponse({"message": "Video not found", "status": "error"}, status=404)

    try:
        video = Video.objects.get(id=video_id)
        video_url = default_storage.url(video.video_file)  # Generate the URL of the video
        
        return JsonResponse({
            "message": "Video fetched successfully",
            "status": "success",
            "video_url": video_url,
            "video_name": video.video_name
        })

    except Video.DoesNotExist:
        return JsonResponse({"message": "Video not found", "status": "error"}, status=404)




import speech_recognition as sr

def recognize_speech():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("🎤 Say something...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)

    try:
        query = recognizer.recognize_google(audio)
        return query
    except sr.UnknownValueError:
        return "❌ Could not understand audio."
    except sr.RequestError:
        return "❌ Could not request results, check your internet."

def speech_to_text(request):
    result = recognize_speech()
    return JsonResponse({"recognized_text": result})



def extract_audio(video_path, audio_path):
    """Extracts audio from a video using FFmpeg."""
    command = f"ffmpeg -i \"{video_path}\" -q:a 0 -map a \"{audio_path}\" -y"
    os.system(command)

def transcribe_audio(audio_path, language):
    """Transcribes the given audio file using Whisper."""
    model = whisper.load_model("medium")  # Use "medium" or "large" for better accuracy
    result = model.transcribe(audio_path, language=language)
    return result["text"]

def summarize_text(text):
    """Summarizes the transcribed text using a pre-trained NLP model."""
    summarizer = pipeline("summarization")
    summary = summarizer(text, max_length=100, min_length=30, do_sample=False)
    return summary[0]['summary_text']

@csrf_exempt
@api_view(['POST'])

def generate_description(request):
    """API endpoint to handle video transcription and summarization."""
    if 'video' not in request.FILES:
        return JsonResponse({'error': 'No video file provided'}, status=400)

    video_file = request.FILES['video']

    # Use a temporary file for video
    with NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
        for chunk in video_file.chunks():
            temp_video.write(chunk)
        video_path = temp_video.name

    # Use a temporary file for extracted audio
    with NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
        audio_path = temp_audio.name

    try:
        # Extract audio from video
        extract_audio(video_path, audio_path)

        # Transcription and summarization using threading
        results = {}

        def transcribe():
            results["transcription"] = transcribe_audio(audio_path, "en")  # Change "en" to "hi" for Hindi
        
        def summarize():
            results["summary"] = summarize_text(results.get("transcription", ""))

        transcription_thread = threading.Thread(target=transcribe)
        summarization_thread = threading.Thread(target=summarize)

        transcription_thread.start()
        transcription_thread.join()  # Wait for transcription to finish before summarization

        summarization_thread.start()
        summarization_thread.join()

        # Format output like YouTube
        formatted_description = f"""
📢 **Video Transcription:**  
{results.get("transcription", "")}

📝 **Summary:**  
{results.get("summary", "")}

🔗 **Follow us on social media:**  
👉 YouTube: [Subscribe Now](https://youtube.com/@YourChannel)  
👉 Instagram: [Follow Us](https://instagram.com/YourProfile)  
👉 LinkedIn: [Connect Here](https://linkedin.com/in/YourProfile)  

🔖 **Hashtags:**  
{COMMON_HASHTAGS}
"""

        return JsonResponse({
            'transcription': results.get("transcription", "").encode('utf-8').decode('utf-8'),
            'summary': results.get("summary", "").encode('utf-8').decode('utf-8'),
            'social_media_links': {
                'YouTube': 'https://youtube.com/@YourChannel',
                'Instagram': 'https://instagram.com/YourProfile',
                'LinkedIn': 'https://linkedin.com/in/YourProfile'
            },
            'hashtags': COMMON_HASHTAGS,
            'formatted_description': formatted_description
        }, json_dumps_params={'ensure_ascii': False})

    except Exception as e:
        return JsonResponse({'error': str(e)}, status=400)

    finally:
        # Cleanup temp files
        if os.path.exists(video_path):
            os.remove(video_path)
        if os.path.exists(audio_path):
            os.remove(audio_path)


#like 
@csrf_exempt
def like_video(request):
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            print("Received Data:", data)  # Debugging: Print received data

            if 'enrollment' not in data or 'video_id' not in data:
                return JsonResponse({"error": "Missing required fields"}, status=400)
            
            user = get_object_or_404(User, enrollment=data['enrollment'])  
            video = get_object_or_404(Video, id=data['video_id'])

            # Ensure status is valid
            status = data.get("status", "like")  # Default to 'like' if not provided
            if status not in ["like", "dislike"]:
                return JsonResponse({"error": "Invalid status value"}, status=400)

            # Check if user has already liked/disliked
            like_entry, created = VideoLike.objects.get_or_create(user=user, video=video, defaults={"status": status})

            if not created and like_entry.status == status:
                like_entry.delete()  # Unlike the video
                return JsonResponse({"message": f"{status.capitalize()} removed", "like_status": None})

            # Save the new like/dislike status
            print(f"DEBUG: Before saving - status: {status}, like_entry.status: {like_entry.status}")
            like_entry.status = status
            like_entry.save()
            
            return JsonResponse({"message": f"Video {status}d", "like_status": status})
        
        except json.JSONDecodeError:
            return JsonResponse({"error": "Invalid JSON format"}, status=400)
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            print("Received Data:", data)  # Debugging: Print received data

            if 'enrollment' not in data or 'video_id' not in data:
                return JsonResponse({"error": "Missing required fields"}, status=400)
            
            user = get_object_or_404(User, enrollment=data['enrollment'])  
            video = get_object_or_404(Video, id=data['video_id'])

            # Ensure status is valid
            status = data.get("status", "like")  # Default to 'like' if not provided
            if status not in ["like", "dislike"]:
                return JsonResponse({"error": "Invalid status value"}, status=400)

            # Check if user has already liked/disliked
            like_entry, created = VideoLike.objects.get_or_create(user=user, video=video)

            if like_entry.status == status:
                like_entry.delete()  # Unlike the video
                return JsonResponse({"message": f"{status.capitalize()} removed", "like_status": None})

            # Save the new like/dislike status
            like_entry.status = status
            like_entry.save()
            return JsonResponse({"message": f"Video {status}d", "like_status": status})
        
        except json.JSONDecodeError:
            return JsonResponse({"error": "Invalid JSON format"}, status=400)
@csrf_exempt
def dislike_video(request):
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            print("Received Data:", data)
            
            if 'enrollment' not in data:
                return JsonResponse({"error": "Missing 'enrollment' key in request"}, status=400)
            
            user = get_object_or_404(User, enrollment=data['enrollment'])
            video = get_object_or_404(Video, title=data['video_title'])

            # Check if user has already liked/disliked
            like_entry, created = VideoLike.objects.get_or_create(user=user, video=video)
            
            if like_entry.status == "dislike":
                like_entry.delete()  # Remove dislike
                return JsonResponse({"message": "Dislike removed", "like_status": None})

            like_entry.status = "dislike"
            like_entry.save()
            return JsonResponse({"message": "Video disliked", "like_status": "dislike"})
        
        except json.JSONDecodeError:
            return JsonResponse({"error": "Invalid JSON format"}, status=400)

    return JsonResponse({"error": "Invalid request method"}, status=405)
def video_like_count(request, video_title):
    video = get_object_or_404(Video, title=video_title)
    like_count = VideoLike.objects.filter(video=video, status="like").count()
    dislike_count = VideoLike.objects.filter(video=video, status="dislike").count()
    
    return JsonResponse({"video": video_title, "likes": like_count, "dislikes": dislike_count})




@csrf_exempt
def add_comment(request):
    if request.method == "POST":
        try:
            data = json.loads(request.body.decode("utf-8"))
            print("Received Data:", data)  # Debugging output
        except json.JSONDecodeError:
            return JsonResponse({"error": "Invalid JSON format"}, status=400)

        enrollment = data.get("enrollment")
        video_id = data.get("video_id")
        comment_text = data.get("comment")

        if not enrollment:
            return JsonResponse({"error": "Enrollment number is required"}, status=400)

        # ✅ Fetch the User instance using the enrollment number
        try:
            user = User.objects.get(enrollment=enrollment)
        except User.DoesNotExist:
            return JsonResponse({"error": "User not found"}, status=404)

        # ✅ Fetch the Video instance
        video = get_object_or_404(Video, id=video_id)

        # ✅ Create and save the Comment instance with the correct user reference
        comment = Comment.objects.create(
            user=user,  # Correctly storing the User instance
            video=video,
            comment=comment_text
        )

        return JsonResponse({"message": "Comment added successfully!", "comment_id": comment.id}, status=201)




# # Function to detect scenes using SceneDetect
# def detect_scenes(video_path):
#     video_manager = VideoManager([video_path])
#     scene_manager = SceneManager()
#     scene_manager.add_detector(ContentDetector(threshold=30))  # Adjust threshold

#     video_manager.set_downscale_factor()
#     video_manager.start()
#     scene_manager.detect_scenes(frame_source=video_manager)

#     scenes = scene_manager.get_scene_list()
#     return [(start.get_seconds(), end.get_seconds()) for start, end in scenes]

# # Function to detect high audio peaks
# def detect_audio_peaks(audio_path, min_silence_len=500, silence_thresh=-30):
#     audio = AudioSegment.from_file(audio_path)
#     nonsilent_chunks = detect_nonsilent(audio, min_silence_len=min_silence_len, silence_thresh=silence_thresh)
#     return [(chunk[0] / 1000, chunk[1] / 1000) for chunk in nonsilent_chunks]

# # Function to create short clips
# def create_short_videos(input_video, output_folder, timestamps, num_clips=10):
#     video = VideoFileClip(input_video)
#     selected_clips = timestamps[:num_clips]
#     clip_paths = []

#     for i, (start_time, end_time) in enumerate(selected_clips):
#         output_path = os.path.join(output_folder, f"short_clip_{i+1}.mp4")
#         short_clip = video.subclip(start_time, end_time)
#         short_clip.write_videofile(output_path, codec="libx264", fps=video.fps, audio_codec="aac")
#         clip_paths.append(output_path)

#     return clip_paths

# @api_view(["POST"])
# def generate_short_videos(request):
#     video_path = request.data.get("video_path")
    
#     if not video_path or not os.path.exists(video_path):
#         return Response({"error": "Invalid video path"}, status=400)

#     output_folder = os.path.dirname(video_path)

#     # Detect scenes & audio peaks
#     scene_timestamps = detect_scenes(video_path)
#     audio_timestamps = detect_audio_peaks(video_path)
#     all_timestamps = sorted(scene_timestamps + audio_timestamps, key=lambda x: x[0])

#     # Generate short clips
#     generated_clips = create_short_videos(video_path, output_folder, all_timestamps)

#     # Save clips to DB
#     short_videos = []
#     for clip in generated_clips:
#         short_video = ShortClip.objects.create(short_clip=clip)
#         short_videos.append({"id": short_video.id, "short_clip_url": short_video.short_clip.url})

#     return Response(
#         {
#             "message": "Short videos generated successfully!",
#             "short_videos": short_videos,
#         },
#         status=201,
#     )


#video sumaary 
whisper_model = whisper.load_model("medium")
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

import os

def extract_audio(video_path, audio_path):
    """Extracts audio from a video using FFmpeg."""
    command = f'ffmpeg -i "{video_path}" -q:a 0 -map a "{audio_path}" -y'
    os.system(command)

    # ✅ Check if file was created
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Error: {audio_path} was not created.")


def transcribe_audio(audio_path, language="en"):
    """Transcribes the given audio file using Whisper."""
    result = whisper_model.transcribe(audio_path, language=language)
    return result["text"]

def summarize_text(text):
    """Summarizes the transcribed text."""
    input_length = len(text.split())
    max_length = min(500, int(input_length * 0.4))
    min_length = max(150, int(input_length * 0.2))

    summary = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
    return summary[0]['summary_text']
import logging
from django.http import JsonResponse

logger = logging.getLogger(__name__)

@csrf_exempt
def summarize_video(request):
    if request.method != "POST":
        return JsonResponse({'error': 'Invalid request method'}, status=405)

    if 'video' not in request.FILES:
        return JsonResponse({'error': 'No video file provided'}, status=400)

    video_file = request.FILES['video']
    video_dir = "videos"
    audio_dir = "audio"
    os.makedirs(video_dir, exist_ok=True)
    os.makedirs(audio_dir, exist_ok=True)

    video_path = os.path.join(video_dir, video_file.name)
    audio_filename = f"{os.path.splitext(video_file.name)[0]}.wav"
    audio_path = os.path.join(audio_dir, audio_filename)

    try:
        with open(video_path, 'wb') as f:
            for chunk in video_file.chunks():
                f.write(chunk)

        extract_audio(video_path, audio_path)

        hindi_transcription = transcribe_audio(audio_path, "hi")
        english_transcription = transcribe_audio(audio_path, "en")

        hindi_summary = summarize_text(hindi_transcription)
        english_summary = summarize_text(english_transcription)

        response = JsonResponse({
            'hindi_transcription': hindi_transcription,
            
            'english_summary': english_summary
        }, json_dumps_params={'ensure_ascii': False})
        
    except Exception as e:
        logging.error(f"Error processing video: {str(e)}")
        return JsonResponse({'error': str(e)}, status=500)

    finally:
        if os.path.exists(video_path):
            os.remove(video_path)
        if os.path.exists(audio_path):
            os.remove(audio_path)

    return response



@api_view(['POST'])
@csrf_exempt
def log_video_watch(request):
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            user_enrollment = data.get("enrollment")
            video_id = data.get("video_id")

            # Validate user
            try:
                user = User.objects.get(enrollment=user_enrollment)  # ✅ Fix: Use 'User' instead of 'Users'
            except User.DoesNotExist:
                return JsonResponse({"error": "User not found"}, status=404)

            # Validate video
            try:
                video = Video.objects.get(id=video_id)
            except Video.DoesNotExist:
                return JsonResponse({"error": "Video not found"}, status=404)

            # Log the video view
            VideoViewLog.objects.create(user=user, video=video)

            return JsonResponse({"message": "Video view logged successfully"}, status=201)

        except json.JSONDecodeError:
            return JsonResponse({"error": "Invalid JSON format"}, status=400)
    
    return JsonResponse({"error": "Only POST requests are allowed"}, status=405)


@csrf_exempt
def user_watch_history(request, enrollment):
    if request.method == "GET":
        # Fetch watch history for the given enrollment ID
        watch_history = VideoViewLog.objects.filter(user_id=enrollment).select_related('video').order_by('-watch_at')

        # Format the response
        history = [
    {
        "video_name": log.video.video_name,  # ✅ Use the correct field name
        "watched_at": log.watch_at,
    }
    for log in VideoViewLog.objects.filter(user__enrollment=enrollment)
]

        return JsonResponse({"history": history}, safe=False)

    return JsonResponse({"error": "Invalid request method"}, status=400)




def suggested_videos(request):
    if request.method == "GET":
        # 🔹 Get the most watched videos (top 5)
        most_watched = (
            VideoViewLog.objects.values("video")
            .annotate(watch_count=Count("video"))
            .order_by("-watch_count")[:5]
        )
        most_watched_ids = [entry["video"] for entry in most_watched]

        # 🔹 Get the most liked videos (top 5)
        most_liked = (
            VideoLike.objects.filter(status="like")
            .values("video")
            .annotate(like_count=Count("video"))
            .order_by("-like_count")[:5]
        )
        most_liked_ids = [entry["video"] for entry in most_liked]

        # 🔹 Combine both lists and remove duplicates
        unique_video_ids = set(most_watched_ids + most_liked_ids)
        videos = Video.objects.filter(id__in=unique_video_ids)

        # 🔹 Format response
        response_data = {
            "suggested_videos": [
                {
                    "video_id": video.id,
                    "video_name": video.video_name,
                    "watch_count": next((item["watch_count"] for item in most_watched if item["video"] == video.id), 0),
                    "like_count": next((item["like_count"] for item in most_liked if item["video"] == video.id), 0),
                }
                for video in videos
            ]
        }

        return JsonResponse(response_data, safe=False)

    return JsonResponse({"error": "Invalid request method"}, status=400)
