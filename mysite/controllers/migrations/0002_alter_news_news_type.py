# Generated by Django 3.2.3 on 2021-06-03 02:38

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('controllers', '0001_initial'),
    ]

    operations = [
        migrations.AlterField(
            model_name='news',
            name='news_type',
            field=models.IntegerField(),
        ),
    ]
