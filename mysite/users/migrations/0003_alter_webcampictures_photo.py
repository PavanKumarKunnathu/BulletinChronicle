# Generated by Django 3.2.3 on 2021-06-10 03:46

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('users', '0002_webcampictures'),
    ]

    operations = [
        migrations.AlterField(
            model_name='webcampictures',
            name='photo',
            field=models.FileField(upload_to='webcam/'),
        ),
    ]
