# Generated by Django 3.2.7 on 2021-12-01 15:21

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('mrc', '0017_topic_description'),
    ]

    operations = [
        migrations.AlterField(
            model_name='topic',
            name='context',
            field=models.CharField(max_length=2048),
        ),
    ]
