# utube/models.py

from django.db import models

class User(models.Model):
    enrollment = models.CharField(max_length=255, unique=True)
    password = models.CharField(max_length=255)

    class Meta:
        db_table = 'users'
        managed = True  # Set to False if the table already exists in your database.


class Video(models.Model):
    video_name = models.CharField(max_length=255)
    video_file = models.FileField(upload_to='videos/')
    upload_date = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = 'videos'  # Table name in the database
        managed = True  # Django will manage this table

    def __str__(self):
        return self.video_name
