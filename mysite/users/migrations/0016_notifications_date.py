# Generated by Django 3.2.3 on 2021-06-14 11:13

from django.db import migrations, models
import django.utils.timezone


class Migration(migrations.Migration):

    dependencies = [
        ('users', '0015_notifications_status'),
    ]

    operations = [
        migrations.AddField(
            model_name='notifications',
            name='date',
            field=models.DateTimeField(auto_now_add=True, default=django.utils.timezone.now),
            preserve_default=False,
        ),
    ]
