from django.db import models

class GeminiModel(models.Model):
    name = models.CharField(max_length=100)
    description = models.TextField()
    image_url = models.URLField()
    specifications = models.JSONField()

    def __str__(self):
        return self.name

class GeminiInquiry(models.Model):
    user_name = models.CharField(max_length=100)
    user_email = models.EmailField()
    model = models.ForeignKey(GeminiModel, on_delete=models.CASCADE)
    inquiry_date = models.DateTimeField(auto_now_add=True)
    message = models.TextField()

    def __str__(self):
        return f"Inquiry by {self.user_name} for {self.model.name}"