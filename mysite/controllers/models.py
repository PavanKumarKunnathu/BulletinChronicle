from django.db import models
import uuid
# Create your models here.
class news(models.Model):
    id = models.CharField(primary_key=True, default=uuid.uuid4, editable=False, max_length=36)
    news_type=models.IntegerField()
    title=models.TextField()
    image=models.ImageField(upload_to='news/')
    news_date=models.DateTimeField()
    created_date=models.DateTimeField(auto_now_add=True)
    location=models.TextField()
    description = models.TextField()
    summary=models.TextField()





