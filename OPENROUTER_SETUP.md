# OpenRouter Setup Guide

## Why OpenRouter?

We've switched to **OpenRouter** which provides **FREE unlimited access** to Gemini 2.0 Flash Experimental with:
- ✅ **No quota limits** (unlike Google's free tier with 10-50 requests/minute)
- ✅ **1,048,576 token context window** (massive - can handle very long prompts)
- ✅ **$0/M input and output tokens** (completely free)
- ✅ **Better performance** - Gemini 2.0 Flash is faster than 1.5
- ✅ **Enhanced capabilities** - Better at complex instruction following, coding, multimodal

## Setup Instructions

### Step 1: Get Your OpenRouter API Key

1. Go to **https://openrouter.ai/keys**
2. Sign up or log in (free account)
3. Click **"Create Key"**
4. Copy your API key (starts with `sk-or-...`)

### Step 2: Add API Key to Settings

Open `dias/university-matcher/university_matcher/settings.py` and find this line:

```python
OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY', '')  # Add your OpenRouter API key here
```

**Option A: Environment Variable (Recommended)**
```bash
# Windows PowerShell
$env:OPENROUTER_API_KEY = "your-api-key-here"

# Linux/Mac
export OPENROUTER_API_KEY="your-api-key-here"
```

**Option B: Direct in Settings (Quick)**
```python
OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY', 'sk-or-v1-your-key-here')
```

### Step 3: Restart Django Server

```bash
# Stop the server (Ctrl+C)
# Then restart
python manage.py runserver
```

You should see:
```
================================================================================
INFO: Using OpenRouter with Gemini 2.0 Flash Experimental (FREE)
INFO: API Key: sk-or-v1-...
INFO: Model: google/gemini-2.0-flash-exp:free
INFO: 1,048,576 context window, unlimited tokens
================================================================================
```

## Fallback to Google's API

If no OpenRouter key is configured, the system automatically falls back to Google's official Gemini API (with quotas).

## Benefits

### Google's Free Tier (Old)
- ❌ 10 requests/minute limit (flash)
- ❌ 2 requests/minute limit (pro)
- ❌ Quota errors during heavy use
- ❌ Need to wait 20-30 seconds between requests

### OpenRouter (New)
- ✅ Unlimited requests
- ✅ No waiting
- ✅ No quota errors
- ✅ Better model (Gemini 2.0 vs 1.5)
- ✅ Faster responses
- ✅ Same quality as paid tier

## Troubleshooting

**Problem**: "INFO: OpenRouter not configured, falling back..."
- **Solution**: Check that API key is set correctly in settings.py or environment variable

**Problem**: API call fails with 401
- **Solution**: Your API key is invalid. Get a new one from https://openrouter.ai/keys

**Problem**: API call fails with network error
- **Solution**: Check your internet connection

## Links

- OpenRouter Website: https://openrouter.ai
- Get API Keys: https://openrouter.ai/keys
- Model Info: https://openrouter.ai/google/gemini-2.0-flash-exp:free
- Documentation: https://openrouter.ai/docs

