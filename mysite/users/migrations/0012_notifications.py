# Generated by Django 3.2.3 on 2021-06-13 17:46

from django.db import migrations, models
import uuid


class Migration(migrations.Migration):

    dependencies = [
        ('users', '0011_posts'),
    ]

    operations = [
        migrations.CreateModel(
            name='notifications',
            fields=[
                ('id', models.CharField(default=uuid.uuid4, editable=False, max_length=36, primary_key=True, serialize=False)),
                ('userid', models.IntegerField(default=0)),
                ('webcamimage', models.ImageField(upload_to='posts_webcam_images/')),
                ('location', models.TextField()),
            ],
        ),
    ]