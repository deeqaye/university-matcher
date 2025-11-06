
from django.shortcuts import render

def home(request):
    context = {'welcome_message': 'Welcome to the University Matcher!'}
    return render(request, 'home.html', context)