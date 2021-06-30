from django.db import models
import uuid

# Create your models here.

class users(models.Model):
    username=models.TextField(max_length=100)
    email=models.EmailField(unique=True,max_length=200)
    password=models.TextField(max_length=1000)
    photo=models.ImageField(upload_to ='data/',default='abc.png')
    datetime = models.DateTimeField(auto_now=True)
    location=models.TextField(max_length=200)
    notifications=models.IntegerField(default=0)
    security=models.IntegerField(default=1)


class WebcamPictures(models.Model):
    email = models.EmailField(max_length=200)
    photo = models.ImageField(upload_to ='webcam/')

    def delete(self, *args, **kwargs):
        self.photo.delete()
        super().delete(*args,**kwargs)

class temp(models.Model):
    username = models.TextField(max_length=100)
    email = models.EmailField(max_length=200)
    password = models.TextField(max_length=1000)
    photo = models.ImageField(upload_to='temp/')
    webcampic=models.ImageField(upload_to='tempwebcam/')
    datetime = models.DateTimeField(auto_now=True)
    location = models.TextField(max_length=200)
    notifications = models.IntegerField(default=0)
    security = models.IntegerField(default=1)
    def delete(self, *args, **kwargs):
        self.photo.delete()
        self.webcampic.delete()
        super().delete(*args,**kwargs)
class demoposts(models.Model):
    id = models.CharField(primary_key=True, default=uuid.uuid4, editable=False, max_length=36)
    userid = models.IntegerField(default=0)
    title = models.TextField()
    photo = models.ImageField(upload_to='demo_posts_images/')
    webcampic=models.ImageField(upload_to='posts_webcam_images/')
    news_date = models.DateTimeField()
    created_date = models.DateTimeField(auto_now_add=True)
    location = models.TextField()
    description = models.TextField()
class posts(models.Model):
    id = models.CharField(primary_key=True, default=uuid.uuid4, editable=False, max_length=36)
    userid = models.IntegerField(default=0)
    title = models.TextField()
    photo = models.ImageField(upload_to='posts_images/')
    news_date = models.DateTimeField()
    created_date = models.DateTimeField(auto_now_add=True)
    location = models.TextField()
    description = models.TextField()
class notifications(models.Model):
    id = models.CharField(primary_key=True, default=uuid.uuid4, editable=False, max_length=36)
    userid = models.IntegerField(default=0)
    webcamimage = models.ImageField(upload_to='posts_webcam_images/')
    location = models.TextField()
    status=models.IntegerField(default=0)
    created_date = models.DateTimeField(auto_now_add=True)

class newscomments(models.Model):
    created_date = models.DateTimeField(auto_now_add=True)
    news_id=models.TextField()
    comment=models.TextField()
    userid = models.IntegerField(default=0)

class likes(models.Model):
    created_date = models.DateTimeField(auto_now_add=True)
    news_id = models.TextField()
    userid = models.IntegerField(default=0)
class supportnews(models.Model):
    created_date = models.DateTimeField(auto_now_add=True)
    news_id = models.TextField()
    userid = models.IntegerField(default=0)

















