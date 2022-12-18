# Generated by Django 4.1.3 on 2022-12-18 03:10

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('trainer', '0001_initial'),
    ]

    operations = [
        migrations.RenameModel(
            old_name='TrainingModel',
            new_name='ModelTrainingParams',
        ),
        migrations.AddField(
            model_name='category',
            name='files',
            field=models.FileField(null=True, upload_to=''),
        ),
        migrations.AddField(
            model_name='category',
            name='num_files',
            field=models.IntegerField(default=0),
        ),
    ]
