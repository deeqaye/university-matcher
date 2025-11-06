from django.db import models

class University(models.Model):
    name = models.CharField(max_length=255)
    country = models.CharField(max_length=100)
    language = models.CharField(max_length=100)
    gpa_min = models.FloatField()
    sat_min = models.IntegerField()
    international_cost_max = models.FloatField()

    def __str__(self):
        return self.name

class StudentInput(models.Model):
    user = models.CharField(max_length=255)
    countries = models.JSONField()
    languages = models.JSONField()
    budget_max = models.FloatField()
    gpa = models.FloatField()
    sat = models.IntegerField()

    def __str__(self):
        return self.user