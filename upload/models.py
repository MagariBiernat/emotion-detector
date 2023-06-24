from django.db import models

# Create your models here.

class EmotionModel(models.Model):
	emotion = models.CharField(max_length=20)

	def __str__(self):
		return self.emotion