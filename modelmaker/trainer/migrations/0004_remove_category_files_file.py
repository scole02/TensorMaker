# Generated by Django 4.1.3 on 2022-12-22 23:46

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('trainer', '0003_rename_num_categories_modeltrainingparams_number_of_categories'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='category',
            name='files',
        ),
        migrations.CreateModel(
            name='File',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('file', models.FileField(blank=True, help_text='Allowed size is 24.0 MBs', upload_to='PN_files/%Y/%m/%d/', verbose_name='Files')),
                ('category', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='trainer.category')),
            ],
        ),
    ]