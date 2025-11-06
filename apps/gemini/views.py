from django.shortcuts import render, redirect
from django.http import HttpResponse, JsonResponse
from django.conf import settings
from django.views.decorators.csrf import csrf_exempt
from .models import GeminiModel, GeminiInquiry
import requests
from bs4 import BeautifulSoup
import json

# Only import and configure genai if API key is set
try:
    import google.generativeai as genai
    if hasattr(settings, 'GEMINI_API_KEY') and settings.GEMINI_API_KEY and settings.GEMINI_API_KEY != 'your-gemini-api-key-here':
        api_key = settings.GEMINI_API_KEY
        # Log which API key is being used (masked for security)
        masked_key = f"{api_key[:10]}...{api_key[-4:]}" if len(api_key) > 14 else "***"
        print(f"INFO: Configuring Gemini API with key: {masked_key}")
        genai.configure(api_key=api_key)
        GEMINI_ENABLED = True
    else:
        print("WARNING: GEMINI_API_KEY not configured or set to placeholder value")
        GEMINI_ENABLED = False
except ImportError:
    print("WARNING: google.generativeai not installed. Install with: pip install google-generativeai")
    GEMINI_ENABLED = False

def index(request):
    university = request.GET.get('university', '')
    country = request.GET.get('country', '')
    autoload = request.GET.get('autoload', 'false') == 'true'
    
    context = {
        'gemini_enabled': GEMINI_ENABLED,
        'university': university,
        'country': country,
        'autoload': autoload,
    }
    return render(request, 'gemini/chatbot.html', context)

@csrf_exempt
def chat(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            user_message = data.get('message', '')
            conversation_history = data.get('history', [])
            
            if not GEMINI_ENABLED:
                return JsonResponse({
                    'success': False,
                    'error': 'Gemini API is not configured. Please add your API key in settings.py'
                })
            
            if not user_message:
                return JsonResponse({
                    'success': False,
                    'error': 'No message provided'
                })
            
            # Try different model names in order of preference
            model_names = [
                'gemini-2.5-pro', 'gemini-1.5-pro-latest', 'gemini-1.5-flash-latest'
            ]
            
            model = None
            last_error = None
            
            for model_name in model_names:
                try:
                    model = genai.GenerativeModel(model_name)
                    # Test if model works by doing a simple call
                    test_response = model.generate_content("Hi")
                    break  # Success, use this model
                except Exception as e:
                    last_error = str(e)
                    continue
            
            if model is None:
                return JsonResponse({
                    'success': False,
                    'error': f'Could not initialize Gemini model. Last error: {last_error}. Try updating: pip install --upgrade google-generativeai'
                })
            
            # Build conversation context
            prompt = "You are a helpful university information assistant. Answer questions about universities, admissions, programs, campus life, student services, and student experiences. Provide detailed, accurate, and helpful information.\n\n"
            
            # Add conversation history
            for msg in conversation_history[-10:]:  # Keep last 10 messages for context
                prompt += f"User: {msg['user']}\nAssistant: {msg['bot']}\n\n"
            
            prompt += f"User: {user_message}\nAssistant:"
            
            # Generate response
            response = model.generate_content(prompt)
            bot_response = response.text

            # Save the conversation to the database
            try:
                university_context = data.get('university', None)
                inquiry = GeminiInquiry(
                    user_message=user_message,
                    bot_response=bot_response,
                    university_context=university_context
                )
                inquiry.save()
            except Exception as db_error:
                print(f"Database logging error: {db_error}")
            
            return JsonResponse({
                'success': True,
                'response': bot_response
            })
            
        except Exception as e:
            return JsonResponse({
                'success': False,
                'error': str(e)
            })
    
    return JsonResponse({
        'success': False,
        'error': 'Invalid request method'
    })