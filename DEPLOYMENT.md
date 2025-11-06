# Deployment Guide - University Matcher

## 🚀 Quick Deployment Options

---

## Option 1: Railway (Recommended)

### Why Railway?
- ✅ $5 free credit/month (plenty for this app)
- ✅ Automatic deployments from GitHub
- ✅ Very easy setup
- ✅ Great for Django

### Steps:

1. **Create GitHub Repository**
   ```bash
   cd e:\stuff\dome\dias\university-matcher
   git init
   git add .
   git commit -m "Initial commit"
   
   # Create repo on GitHub, then:
   git remote add origin https://github.com/YOUR_USERNAME/university-matcher.git
   git push -u origin main
   ```

2. **Deploy on Railway**
   - Go to [railway.app](https://railway.app)
   - Sign up with GitHub
   - Click "New Project" → "Deploy from GitHub repo"
   - Select your `university-matcher` repository
   - Railway will auto-detect Django and deploy!

3. **Set Environment Variables**
   
   In Railway dashboard, go to Variables and add:
   ```
   SECRET_KEY=your-super-secret-key-here-generate-a-new-one
   DEBUG=False
   ALLOWED_HOSTS=*.railway.app
   GEMINI_API_KEY=AIzaSyA_ULCqmxYyNSt0S5XURKRjhi7-Lu3T79c
   ```

4. **Generate Django Secret Key**
   ```python
   # Run in Python
   from django.core.management.utils import get_random_secret_key
   print(get_random_secret_key())
   ```

5. **Done!**
   - Railway will give you a URL like: `https://university-matcher-production.up.railway.app`
   - Your app is live!

---

## Option 2: Render

### Steps:

1. **Push to GitHub** (same as Railway step 1)

2. **Deploy on Render**
   - Go to [render.com](https://render.com)
   - Sign up with GitHub
   - Click "New +" → "Web Service"
   - Connect your repository
   - Configure:
     ```
     Name: university-matcher
     Environment: Python 3
     Build Command: pip install -r requirements.txt && python manage.py collectstatic --noinput
     Start Command: gunicorn university_matcher.wsgi:application
     ```

3. **Set Environment Variables**
   
   In Render dashboard, add:
   ```
   SECRET_KEY=your-super-secret-key-here
   DEBUG=False
   ALLOWED_HOSTS=*.onrender.com
   GEMINI_API_KEY=AIzaSyA_ULCqmxYyNSt0S5XURKRjhi7-Lu3T79c
   ```

4. **Done!**
   - URL: `https://university-matcher.onrender.com`

---

## Option 3: PythonAnywhere (Completely Free)

### Steps:

1. **Sign Up**
   - Go to [pythonanywhere.com](https://www.pythonanywhere.com)
   - Create free account

2. **Upload Code**
   - Go to "Files" tab
   - Upload your project or clone from GitHub:
   ```bash
   cd ~
   git clone https://github.com/YOUR_USERNAME/university-matcher.git
   ```

3. **Create Virtual Environment**
   ```bash
   mkvirtualenv --python=/usr/bin/python3.10 myenv
   cd ~/university-matcher
   pip install -r requirements.txt
   ```

4. **Configure Web App**
   - Go to "Web" tab
   - Add new web app
   - Choose "Manual configuration"
   - Python 3.10
   - Set:
     - Source code: `/home/YOUR_USERNAME/university-matcher`
     - Working directory: `/home/YOUR_USERNAME/university-matcher`
     - WSGI file: Edit to point to your `wsgi.py`

5. **Update Settings**
   - In settings.py, add your domain to `ALLOWED_HOSTS`
   ```python
   ALLOWED_HOSTS = ['YOUR_USERNAME.pythonanywhere.com']
   ```

6. **Collect Static Files**
   ```bash
   python manage.py collectstatic
   ```

7. **Done!**
   - URL: `https://YOUR_USERNAME.pythonanywhere.com`

---

## 📝 Pre-Deployment Checklist

### Files Already Created ✅
- `requirements.txt` - Python dependencies
- `Procfile` - For Railway/Render
- `runtime.txt` - Python version
- Updated `settings.py` - Production-ready

### Before Deploying:

1. **Update data.csv path** if needed
   - Make sure `data.csv` is in the correct location
   - Or update the path in `views.py`

2. **Test locally**
   ```bash
   python manage.py collectstatic
   gunicorn university_matcher.wsgi
   ```

3. **Generate new SECRET_KEY**
   ```python
   from django.core.management.utils import get_random_secret_key
   print(get_random_secret_key())
   ```

---

## 🔒 Security Notes

1. **Never commit sensitive data**
   - API keys should be environment variables
   - Don't commit `.env` files

2. **Set DEBUG=False in production**

3. **Use strong SECRET_KEY**

4. **Update ALLOWED_HOSTS**

---

## 🐛 Troubleshooting

### Static files not loading
```bash
python manage.py collectstatic --noinput
```

### Database errors
- Default SQLite works fine for small apps
- For production, consider PostgreSQL (Railway/Render offer free tiers)

### Import errors
- Make sure `requirements.txt` includes all dependencies
- Check Python version matches `runtime.txt`

---

## 📊 Cost Comparison

| Platform | Free Tier | Storage | Database | Custom Domain |
|----------|-----------|---------|----------|---------------|
| Railway | $5 credit/mo | 1 GB | Yes (PostgreSQL) | Yes |
| Render | 750 hrs/mo | 1 GB | Yes (PostgreSQL) | Yes |
| PythonAnywhere | Always free | 512 MB | SQLite only | No |

---

## 🎉 After Deployment

Your app will be live at:
- **Railway**: `https://university-matcher-production.up.railway.app`
- **Render**: `https://university-matcher.onrender.com`
- **PythonAnywhere**: `https://YOUR_USERNAME.pythonanywhere.com`

Share the link and start matching students to universities! 🎓

