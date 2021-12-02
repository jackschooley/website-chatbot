# Generated by Django 3.2.7 on 2021-12-01 17:55

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('mrc', '0018_alter_topic_context'),
    ]

    operations = [
        migrations.CreateModel(
            name='MRCResult',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('question_text', models.CharField(max_length=128)),
                ('date_time', models.DateTimeField()),
                ('answer_text', models.CharField(max_length=1024)),
            ],
        ),
        migrations.AlterField(
            model_name='topic',
            name='description',
            field=models.CharField(max_length=128),
        ),
        migrations.AlterField(
            model_name='topic',
            name='topic',
            field=models.CharField(max_length=32, unique=True),
        ),
        migrations.DeleteModel(
            name='MRCResults',
        ),
        migrations.AddField(
            model_name='mrcresult',
            name='topic',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='mrc.topic', to_field='topic'),
        ),
    ]