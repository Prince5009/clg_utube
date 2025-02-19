import os
import json
import whisper
from transformers import pipeline

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.core.exceptions import ValidationError
from django.core.files.storage import default_storage
from django.db import connection
from .models import Video
from rest_framework.decorators import api_view

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

        try:
            # Validate the video file (check if it's a valid video format)
            if not video_file.name.endswith(('.mp4', '.avi', '.mov', '.mkv')):
                raise ValidationError("Invalid video format. Please upload a valid video file.")

            # Save the video file
            file_path = default_storage.save(os.path.join('videos', video_file.name), video_file)
            video_url = default_storage.url(file_path)  # Get the URL of the uploaded file

            # Save video data in the database
            video = Video(video_name=video_name, video_file=file_path)
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

@api_view(['POST'])
def generate_description(request):
    """API endpoint to handle video transcription and summarization."""
    if 'video' not in request.FILES:
        return JsonResponse({'error': 'No video file provided'}, status=400)

    video_file = request.FILES['video']
    
    # Define paths
    video_dir = "videos"
    audio_dir = "audio"
    
    # Ensure the directories exist
    os.makedirs(video_dir, exist_ok=True)
    os.makedirs(audio_dir, exist_ok=True)

    video_path = os.path.join(video_dir, video_file.name)
    
    # Ensure the audio file has a valid extension (e.g., .wav)
    audio_filename = f"{os.path.splitext(video_file.name)[0]}.wav"
    audio_path = os.path.join(audio_dir, audio_filename)

    # Save video file temporarily
    with open(video_path, 'wb') as f:
        for chunk in video_file.chunks():
            f.write(chunk)

    try:
        # Extract audio from the video
        extract_audio(video_path, audio_path)
        
        # Transcribe audio in Hindi
        hindi_transcription = transcribe_audio(audio_path, "hi")

        # Summarize the transcription
        hindi_summary = summarize_text(hindi_transcription)

        return JsonResponse({
            'hindi_transcription': hindi_transcription,
            'hindi_summary': hindi_summary
        })
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=400)
    finally:
        # Clean up temporary files
        if os.path.exists(video_path):
            os.remove(video_path)
        if os.path.exists(audio_path):
            os.remove(audio_path)
