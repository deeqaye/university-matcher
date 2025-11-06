# University Matcher

## Setup Instructions

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Download University Images (Important!)
Before running the server, download images for all universities:
```bash
python download_university_images.py
```

This will:
- Read all universities from `../data.csv`
- Fetch images from Wikipedia and Unsplash
- Cache them locally in `static/images/universities/`
- Takes ~5-10 minutes for all universities

### 3. Configure Gemini API Key
Set your Gemini API key in `university_matcher/settings.py` or as an environment variable:
```bash
export GEMINI_API_KEY="your-api-key-here"
```

### 4. Run Server
```bash
python manage.py runserver
```

Visit `http://localhost:8000/universities/` to start searching!

---

# University Matcher (Original README)

This project is a Django web application that helps users find universities based on their student variables such as language, GPA, SAT scores, and budget. Additionally, it includes a feature to inquire about the Gemini 2.5-Pro model.

## Features

- User input form for university matching
- Display of matching universities based on user criteria
- Inquiry form for the Gemini 2.5-Pro model
- Image extraction feature from the Gemini inquiry page

## Project Structure

```
university-matcher/
├── manage.py               # Command-line utility for managing the project
├── university_matcher/
│   ├── __init__.py        # Indicates the directory is a Python package
│   ├── asgi.py             # ASGI configuration for asynchronous server communication
│   ├── settings.py         # Project settings and configuration
│   ├── urls.py             # URL routing for the project
│   ├── wsgi.py             # WSGI configuration for web server communication
├── apps/
│   ├── universities/       # App for university matching functionality
│   ├── gemini/             # App for Gemini model inquiries
├── static/                 # Static files (CSS, JS)
├── templates/              # Base HTML templates
└── README.md               # Project documentation
```

## Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd university-matcher
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

3. Run migrations:
   ```
   python manage.py migrate
   ```

4. Start the development server:
   ```
   python manage.py runserver
   ```

## Usage

- Navigate to `http://127.0.0.1:8000/` to access the university matching form.
- Fill in the required fields and submit to see matching universities.
- Use the inquiry form to ask about the Gemini 2.5-Pro model.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or features.

## License

This project is licensed under the MIT License. See the LICENSE file for details.