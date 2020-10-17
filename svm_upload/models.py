# import os
# import sys
# from django.db import models
# from django.contrib.auth.models import AbstractUser
# from django.utils import timezone
# from django.utils.translation import gettext_lazy as _
#
# def upload_to(instance, filename):
#     now = timezone.now()
#     base, extension = os.path.splitext(filename.lower())
#     milliseconds = now.microsecond // 1000
#     return f"users/{instance.pk}/{now:%Y%m%d%H%M%S}{milliseconds}{extension}"
#
# # Create your models here.
# class File_upload(models.Model):
#     uploads = models.FileField(blank = True )
from django.db import models


class MyFile(models.Model):
    file = models.FileField(blank=False, null=False)
    description = models.CharField(blank = True, max_length=255)
    uploaded_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return '{} ({})'.format(self.description, self.file)
