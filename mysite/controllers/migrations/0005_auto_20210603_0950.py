# Generated by Django 3.2.3 on 2021-06-03 04:20

from django.db import migrations, models
import uuid


class Migration(migrations.Migration):

    dependencies = [
        ('controllers', '0004_auto_20210603_0946'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='news',
            name='id',
        ),
        migrations.AlterField(
            model_name='news',
            name='news_id',
            field=models.CharField(default=uuid.uuid4, editable=False, max_length=36, primary_key=True, serialize=False),
        ),
    ]