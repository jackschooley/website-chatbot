# Generated by Django 3.2.7 on 2021-11-07 15:52

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('mrc', '0014_alter_mrcresults_topic'),
    ]

    operations = [
        migrations.AddField(
            model_name='topic',
            name='context',
            field=models.CharField(default="I'm not a reply guy obv", max_length=1024),
            preserve_default=False,
        ),
    ]
