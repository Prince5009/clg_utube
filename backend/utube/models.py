from django.db import models

class User(models.Model):
    enrollment = models.CharField(max_length=45, unique=True, primary_key=True, db_index=True)  
    password = models.CharField(max_length=255)

    class Meta:
        db_table = 'users'

    def __str__(self):
        return f"User: {self.enrollment}"


class Video(models.Model):
    video_name = models.CharField(max_length=255)  # ✅ Check the actual field name
    description = models.TextField(blank=True, null=True)  # Added description field
    video_file = models.FileField(upload_to='videos/')
    upload_date = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = 'videos'

    def __str__(self):
        return self.video_name


class VideoViewLog(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)  # Corrected Users → User
    video = models.ForeignKey(Video, on_delete=models.CASCADE)
    watch_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.user.enrollment} watched {self.video.video_name}"


class LikeDislike(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, to_field="enrollment")
    video = models.ForeignKey(Video, on_delete=models.CASCADE)
    status = models.CharField(max_length=10, choices=[('like', 'Like'), ('dislike', 'Dislike')])
    timestamp = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = 'like_dislike'
        unique_together = ('user', 'video')

    def __str__(self):
        return f"{self.user.enrollment} - {self.video.video_name} - {self.status}"


class VideoLike(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, to_field="enrollment", db_column="user_id")  
    video = models.ForeignKey(Video, on_delete=models.CASCADE)
    status = models.CharField(max_length=10, choices=[('like', 'Like'), ('dislike', 'Dislike')])
    timestamp = models.DateTimeField(auto_now=True)

    class Meta:
        unique_together = ('user', 'video')
        db_table = 'video_likes'

    def __str__(self):
        return f"{self.user.enrollment} - {self.video.video_name} - {self.status}"


class Comment(models.Model):
    user = models.ForeignKey(User, to_field='enrollment', db_column='enrollment', on_delete=models.CASCADE)
    video = models.ForeignKey(Video, on_delete=models.CASCADE)
    comment = models.CharField(max_length=45)
    comment_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "user_comment"

    def __str__(self):
        return f"Comment by {self.user.enrollment} on Video {self.video.id}"


class ShortClip(models.Model):
    short_clip = models.FileField(upload_to="videos/shots/")
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Short Clip - {self.short_clip.name}"
    

class VideoViewLog(models.Model):
    user = models.ForeignKey(
        User, on_delete=models.CASCADE, to_field="enrollment", db_column="enrollment"
    )  # References 'enrollment' instead of 'id'
    video = models.ForeignKey(Video, on_delete=models.CASCADE)
    watch_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "utube_videoviewlog"  # Ensure table name is correct

    def __str__(self):
        return f"{self.user.enrollment} watched {self.video.title if hasattr(self.video, 'title') else 'Unknown Video'}"
 

