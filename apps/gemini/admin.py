from django.contrib import admin
from .models import GeminiInquiry

class GeminiInquiryAdmin(admin.ModelAdmin):
    list_display = ('user_name', 'user_email', 'inquiry_date', 'model')  # Ensure field names match the model
    ordering = ('inquiry_date',)  # Ensure ordering field matches the model

admin.site.register(GeminiInquiry, GeminiInquiryAdmin)