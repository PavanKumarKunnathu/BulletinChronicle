# Generated by Django 3.2.3 on 2021-06-10 09:23

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('users', '0003_alter_webcampictures_photo'),
    ]

    operations = [
        migrations.AlterField(
            model_name='webcampictures',
            name='photo',
            field=models.ImageField(upload_to='webcam/'),
        ),
    ]
