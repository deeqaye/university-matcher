from django.contrib import admin
from .models import University

@admin.register(University)
class UniversityAdmin(admin.ModelAdmin):
    list_display = ('name', 'country', 'gpa_min', 'sat_min', 'international_cost_max')
    search_fields = ('name', 'country')
    list_filter = ('country',)