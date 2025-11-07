from django.shortcuts import render
from django.http import HttpResponse
import pandas as pd
from django.conf import settings
from pathlib import Path
from functools import lru_cache
from . import uni_find_wrapper
import requests
from bs4 import BeautifulSoup
from urllib.parse import quote, urlparse
import time
import re

# Configure LLM API - Use OpenRouter for Gemini 2.0 Flash (free) or fallback to Google's API
import json

# Check for OpenRouter API key first
OPENROUTER_API_KEY = getattr(settings, 'OPENROUTER_API_KEY', '')
USE_OPENROUTER = bool(OPENROUTER_API_KEY)

if USE_OPENROUTER:
    print("="*80)
    print("INFO: Using OpenRouter with Gemini 2.0 Flash Experimental (FREE)")
    print(f"INFO: API Key: {OPENROUTER_API_KEY[:10]}...")
    print("INFO: Model: google/gemini-2.0-flash-exp:free")
    print("INFO: 1,048,576 context window, unlimited tokens")
    print("="*80)
    GEMINI_ENABLED = True
    google_exceptions = None
else:
    # Fallback to Google's official API
    print("INFO: OpenRouter not configured, falling back to Google's official Gemini API")
    try:
        import google.generativeai as genai
        from google.api_core import exceptions as google_exceptions
        if hasattr(settings, 'GEMINI_API_KEY') and settings.GEMINI_API_KEY and settings.GEMINI_API_KEY != 'your-gemini-api-key-here':
            api_key = settings.GEMINI_API_KEY
            masked_key = f"{api_key[:10]}...{api_key[-4:]}" if len(api_key) > 14 else "***"
            print(f"INFO: Configuring Google Gemini API with key: {masked_key}")
            genai.configure(api_key=api_key)
            GEMINI_ENABLED = True
        else:
            print("WARNING: No API key configured")
            GEMINI_ENABLED = False
            google_exceptions = None
    except ImportError:
        print("WARNING: google.generativeai not installed")
        GEMINI_ENABLED = False
        google_exceptions = None

ENABLE_LLM_ENRICHMENT = GEMINI_ENABLED and getattr(settings, 'ENABLE_LLM_ENRICHMENT', True)
MAX_LONG_DESCRIPTIONS = getattr(settings, 'MAX_LONG_DESCRIPTIONS', 2)


@lru_cache(maxsize=1)
def _load_info_csv():
    """Load info.csv once per worker and cache it to reduce memory churn."""
    possible_paths = [
        Path(settings.BASE_DIR) / 'info.csv',
        Path(settings.BASE_DIR).parent / 'info.csv',
        Path(settings.BASE_DIR).parent.parent / 'info.csv',
    ]

    for path in possible_paths:
        if path.exists():
            try:
                info_df = pd.read_csv(path)
                print(f"INFO: Loaded info.csv from {path}")
                return info_df
            except Exception as e:
                print(f"ERROR loading info.csv from {path}: {e}")
                return None

    print("INFO: info.csv not found in known locations; continuing without it.")
    return None


# Helper function to call OpenRouter API with retry logic
def call_openrouter_api(prompt, model="google/gemini-2.0-flash-exp:free", max_retries=3):
    """Call OpenRouter API with the given prompt and retry on rate limits"""
    url = "https://openrouter.ai/api/v1/chat/completions"
    
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://university-matcher.local",
        "X-Title": "University Matcher"
    }
    
    data = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ]
    }
    
    for attempt in range(max_retries):
        try:
            response = requests.post(url, headers=headers, json=data, timeout=120)
            response.raise_for_status()
            result = response.json()
            
            # Extract text from response
            if 'choices' in result and len(result['choices']) > 0:
                return result['choices'][0]['message']['content']
            else:
                raise Exception(f"Unexpected response format: {result}")
                
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:  # Rate limit
                wait_time = (2 ** attempt) * 2  # Exponential backoff: 2s, 4s, 8s
                print(f"Rate limited by OpenRouter, waiting {wait_time}s before retry (attempt {attempt + 1}/{max_retries})...")
                time.sleep(wait_time)
                if attempt == max_retries - 1:
                    raise Exception(f"Rate limit exceeded after {max_retries} retries")
            else:
                raise
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            wait_time = (2 ** attempt)
            print(f"API error: {e}, retrying in {wait_time}s...")
            time.sleep(wait_time)

# Wrapper class to mimic genai.GenerativeModel interface
class OpenRouterModel:
    def __init__(self, model_name):
        self.model_name = model_name
    
    def generate_content(self, prompt):
        class Response:
            def __init__(self, text):
                self.text = text
        
        try:
            text = call_openrouter_api(prompt, self.model_name)
            return Response(text)
        except Exception as e:
            # Return a response object even on error for consistency
            raise e

# Function to get model (works with both OpenRouter and Google's API)
def get_llm_model(model_name):
    """Get an LLM model instance that works with both OpenRouter and Google's API"""
    if USE_OPENROUTER:
        # Always use Gemini 2.0 Flash for OpenRouter (it's free and the best)
        return OpenRouterModel("google/gemini-2.0-flash-exp:free")
    else:
        # Use Google's official API with the provided model name
        return genai.GenerativeModel(model_name)

def index(request):
    # Load unique countries and languages from CSV
    csv_path = Path(settings.BASE_DIR) / 'data.csv'
    try:
        df = pd.read_csv(csv_path)
        countries = sorted(df['country'].unique().tolist())
        # Extract languages (handle comma-separated values)
        all_languages = set()
        for lang in df['language'].dropna():
            for l in str(lang).split(','):
                all_languages.add(l.strip())
        languages = sorted([l for l in all_languages if l])
        
        context = {
            'countries': countries,
            'languages': languages,
            'language_levels': ['A1', 'A2', 'B1', 'B2', 'C1', 'C2']
        }
    except Exception as e:
        print(f"Error loading form data: {e}")
        context = {
            'countries': [],
            'languages': [],
            'language_levels': ['A1', 'A2', 'B1', 'B2', 'C1', 'C2']
        }
    
    return render(request, 'universities/index.html', context)

def get_university_image(university_name, country):
    """Get cached university image or return placeholder"""
    # Create safe filename from university name
    safe_name = re.sub(r'[^\w\s-]', '', university_name.lower())
    safe_name = re.sub(r'[-\s]+', '-', safe_name)
    filename = f"{safe_name}.jpg"
    
    # Check if cached image exists
    cache_path = Path(settings.BASE_DIR) / 'static' / 'images' / 'universities' / filename
    static_url = f"/static/images/universities/{filename}"
    
    if cache_path.exists():
        return static_url
    
    # If not cached, fall back to a themed Unsplash image before using a placeholder
    query = quote(f"{university_name} {country} campus")
    unsplash_url = f"https://source.unsplash.com/800x600/?{query}"
    return unsplash_url


def get_university_info_from_csv(university_name, country):
    """Load info.csv and find matching university data with improved matching"""
    try:
        info_df = _load_info_csv()
        if info_df is None:
            # info.csv is optional - return None to use data.csv only
            return None
        
        # Normalize names for better matching
        def normalize_name(name):
            if pd.isna(name):
                return ""
            name = str(name).lower().strip()
            # Remove common suffixes and normalize
            name = name.replace("(ucl)", "").replace("(uol)", "").strip()
            return name
        
        normalized_search_name = normalize_name(university_name)
        
        # Normalize country for matching
        def normalize_country(c):
            if pd.isna(c):
                return ""
            c = str(c).lower().strip()
            # Handle variations like "UK (United Kingdom)" -> "uk"
            if '(' in c:
                c = c.split('(')[0].strip()
            return c
        
        normalized_country = normalize_country(country)
        
        # Try multiple matching strategies
        matches = pd.DataFrame()
        
        # Strategy 1: Name contains + Country contains (best match)
        matches = info_df[
            (info_df['University Name'].str.lower().str.contains(normalized_search_name, case=False, na=False, regex=False)) &
            (info_df['Country'].str.lower().str.contains(normalized_country, case=False, na=False, regex=False))
        ]
        
        # Strategy 2: If no match, try with country variations
        if matches.empty:
            country_variations = [normalized_country]
            if ' ' in country:
                country_variations.append(country.split()[0].lower())
            if 'uk' in normalized_country or 'united kingdom' in normalized_country:
                country_variations.extend(['uk', 'united kingdom'])
            if 'usa' in normalized_country or 'united states' in normalized_country:
                country_variations.extend(['usa', 'united states', 'us'])
            
            for country_var in country_variations:
                matches = info_df[
                    (info_df['University Name'].str.lower().str.contains(normalized_search_name, case=False, na=False, regex=False)) &
                    (info_df['Country'].str.lower().str.contains(country_var, case=False, na=False, regex=False))
                ]
                if not matches.empty:
                    break
        
        # Strategy 3: Extract key words from university name
        if matches.empty:
            search_words = [w for w in normalized_search_name.split() if len(w) > 3]
            if search_words:
                for word in search_words:
                    temp_matches = info_df[
                        info_df['University Name'].str.lower().str.contains(word, case=False, na=False, regex=False)
                    ]
                    if not temp_matches.empty:
                        matches = temp_matches
                        break
        
        # Strategy 4: Last resort - just name contains
        if matches.empty:
            matches = info_df[
                info_df['University Name'].str.lower().str.contains(normalized_search_name, case=False, na=False, regex=False)
            ]
        
        if matches.empty:
            print(f"INFO: No match found for '{university_name}' in {country}")
            return None
        
        row = matches.iloc[0]
        print(f"INFO: Matched '{university_name}' to CSV entry: '{row['University Name']}'")
        
        # Get column values safely, handling potential name variations
        def safe_get(col_name_variations):
            for col_name in col_name_variations:
                if col_name in info_df.columns:
                    try:
                        val = row[col_name]
                        if pd.isna(val) or val == '' or str(val).strip() == '':
                            continue
                        return str(val).strip()
                    except (KeyError, IndexError):
                        continue
            return 'N/A'
        
        # Get all column names to handle variations
        all_cols = info_df.columns.tolist()
        
        # Debug: print column names
        print(f"DEBUG: CSV columns found: {all_cols[:5]}...")  # Print first 5
        
        # Find actual column names (handling newlines and variations)
        def find_col(patterns):
            for pattern in patterns:
                for col in all_cols:
                    col_normalized = col.lower().replace('\n', ' ').replace('\r', ' ')
                    pattern_normalized = pattern.lower()
                    if pattern_normalized in col_normalized:
                        return col
            return None
        
        # Try to find columns with various patterns
        ielts_col = find_col(['Minimum IELTS', 'IELTS'])
        gpa_col = find_col(['Minimum GPA', 'GPA'])
        sat_col = find_col(['Minimum SAT', 'SAT'])
        cost_intl_col = find_col(['Cost per year', 'interational', 'international'])
        cost_local_col = find_col(['Cost per year', 'local'])
        
        # Try direct column access first, then use find_col
        def get_val(col_variations, find_col_result=None):
            # Try direct column access
            for col_name in col_variations:
                if col_name in all_cols:
                    val = safe_get([col_name])
                    if val != 'N/A':
                        return val
            # Try find_col result
            if find_col_result:
                val = safe_get([find_col_result])
                if val != 'N/A':
                    return val
            # Try all variations
            return safe_get(col_variations)
        
        result = {
            'university_name': safe_get(['University Name']),
            'country': safe_get(['Country']),
            'language_of_teaching': safe_get(['Language of teaching']),
            'ielts_min': get_val(['Minimum IELTS score\n(or average)', 'Minimum IELTS score (or average)', 'Minimum IELTS score'], ielts_col),
            'gpa_min': get_val(['Minimum GPA \n(or average)', 'Minimum GPA (or average)', 'Minimum GPA'], gpa_col),
            'sat_min': get_val(['Minimum SAT score\n(or average)', 'Minimum SAT score (or average)', 'Minimum SAT score'], sat_col),
            'acceptance_rate': safe_get(['Acceptance rate (%)']),
            'cost_international': get_val(['Cost per year \nfor interational students', 'Cost per year for international students'], cost_intl_col),
            'cost_local': get_val(['Cost per year \nfor local students', 'Cost per year for local students'], cost_local_col),
            'national_ranking': safe_get(['National ranking']),
            'global_ranking': safe_get(['Global ranking']),
            'scholarship': safe_get(['Scholarship (yes/no)']),
            'public_private': safe_get(['Public/Private']),
            'student_faculty_ratio': safe_get(['Student to faculty ratio']),
            'website': safe_get(['Link to official website']),
        }
        
        # Debug: print what we found
        print(f"DEBUG: Extracted data for {result['university_name']}: acceptance_rate={result['acceptance_rate']}, global_ranking={result['global_ranking']}")
        
        return result
    except Exception as e:
        import traceback
        print(f"ERROR loading info.csv: {e}")
        traceback.print_exc()
    
    return None

def get_university_city(university_name, country):
    """Get the city where the university is located using LLM"""
    if not ENABLE_LLM_ENRICHMENT:
        name_lower = university_name.lower()
        if 'london' in name_lower or 'ucl' in name_lower:
            return "London, UK"
        if 'cambridge' in name_lower:
            return "Cambridge, UK"
        if 'oxford' in name_lower:
            return "Oxford, UK"
        return country

    if not GEMINI_ENABLED:
        # Try to extract city from CSV first
        csv_info = get_university_info_from_csv(university_name, country)
        if csv_info and csv_info.get('university_name'):
            # Many university names contain city names, try to extract
            name = csv_info.get('university_name', '')
            if 'London' in name:
                return "London, UK"
            elif 'Cambridge' in name:
                return "Cambridge, UK"
            elif 'Oxford' in name:
                return "Oxford, UK"
        return country  # Fallback to country if LLM not available
    
    try:
        # Use flash model first (cheaper, higher quotas)
        model_names = ['gemini-flash-latest', 'gemini-2.5-pro']
        for model_name in model_names:
            try:
                model = get_llm_model(model_name)
                prompt = f"""What is the city where {university_name} in {country} is located? 

Respond with ONLY the city name, followed by a comma and the country name. For example: "London, UK" or "Boston, USA". 
Do not include any other text, explanations, or punctuation beyond the format "City, Country"."""
                response = model.generate_content(prompt)
                text = (response.text or "").strip()
                # Clean up any extra text
                text = text.split('\n')[0].strip()
                if text and len(text) < 100 and ',' in text:  # Simple validation
                    return text
            except Exception as e:
                # Check if it's a quota error - if so, skip to fallback immediately
                if google_exceptions and isinstance(e, google_exceptions.ResourceExhausted):
                    print(f"Quota exceeded for {model_name} when getting city. Using fallback.")
                    break  # Skip to fallback
                print(f"Error with model {model_name}: {e}")
                continue
    except Exception as e:
        # Check if it's a quota error
        if google_exceptions and isinstance(e, google_exceptions.ResourceExhausted):
            print(f"Quota exceeded when getting city for {university_name}. Using fallback.")
        else:
            print(f"Error getting city: {e}")
    
    # Fallback: try to infer from university name
    name_lower = university_name.lower()
    if 'london' in name_lower or 'ucl' in name_lower:
        return "London, UK"
    elif 'cambridge' in name_lower:
        return "Cambridge, UK"
    elif 'oxford' in name_lower:
        return "Oxford, UK"
    
    return f"{country}"  # Final fallback


def build_csv_fallback_description(university_name, country, stats, csv_info=None):
    """Construct a lightweight description using CSV and match data."""
    parts = []

    institution_type = None
    if csv_info:
        raw_type = str(csv_info.get('public_private', '')).strip()
        if raw_type and raw_type.upper() != 'N/A':
            institution_type = raw_type.lower()

    if institution_type:
        parts.append(f"{university_name} is a {institution_type} institution located in {country}.")
    else:
        parts.append(f"{university_name} is located in {country}.")

    if csv_info:
        global_rank = str(csv_info.get('global_ranking', '')).strip()
        if global_rank and global_rank.upper() != 'N/A':
            parts.append(f"It holds a global ranking of {global_rank}.")

        acceptance = str(csv_info.get('acceptance_rate', '')).strip()
        if acceptance and acceptance.upper() != 'N/A':
            parts.append(f"Recent acceptance rates hover around {acceptance}.")

        scholarship = str(csv_info.get('scholarship', '')).strip()
        if scholarship and scholarship.upper() != 'N/A':
            parts.append(f"Scholarships are {scholarship} for qualified applicants.")

        language = str(csv_info.get('language_of_teaching', '')).strip()
        if language and language.upper() != 'N/A':
            parts.append(f"Programs are taught primarily in {language}.")

        international_cost = str(csv_info.get('cost_international', '')).strip()
        if international_cost and international_cost.upper() != 'N/A':
            parts.append(f"Estimated annual cost for international students is {international_cost}.")
    elif stats:
        cost = stats.get('international_cost_max')
        if cost not in (None, 'N/A', 0):
            parts.append(f"Estimated annual cost for international students is ${cost}.")

    if stats:
        gpa = stats.get('GPA_min')
        sat = stats.get('SAT_min')
        ielts = stats.get('IELTS_min')

        admission_bits = []
        if gpa not in (None, 'N/A', 0):
            admission_bits.append(f"GPA {gpa}+")
        if sat not in (None, 'N/A', 0):
            admission_bits.append(f"SAT {sat}+")
        if ielts not in (None, 'N/A', 0):
            admission_bits.append(f"IELTS {ielts}+")

        if admission_bits:
            parts.append("Admissions typically expect " + ', '.join(admission_bits) + ".")

    return " ".join(parts)


def build_csv_short_description(university_name, country, csv_info=None):
    parts = []
    if csv_info:
        institution_type = str(csv_info.get('public_private', '')).strip()
        if institution_type and institution_type.upper() != 'N/A':
            parts.append(f"{university_name} is a {institution_type.lower()} university in {country} with a strong international profile.")
        else:
            parts.append(f"{university_name} is located in {country} and welcomes students from around the world.")
        highlight = str(csv_info.get('global_ranking', '')).strip()
        if highlight and highlight.upper() != 'N/A':
            parts.append(f"It currently holds a global ranking of {highlight}, reflecting its academic reputation.")
        else:
            language = str(csv_info.get('language_of_teaching', '')).strip()
            if language and language.upper() != 'N/A':
                parts.append(f"Teaching is primarily in {language}, making it accessible for international learners.")
    else:
        parts.append(f"{university_name} is located in {country} and offers a broad mix of programmes.")
    parts.append("Campus life blends research activity with a diverse student community, creating plenty of academic and cultural opportunities.")
    return " ".join(parts)


def _format_currency(amount):
    try:
        value = float(amount)
    except (TypeError, ValueError):
        return str(amount)
    return f"${value:,.0f}"


def build_user_input_summary(user_input, user_preferences):
    note = (user_preferences or '').strip()
    if note:
        return note

    parts = []

    selected_countries = [c for c in user_input.get('countries', []) if c]
    if selected_countries:
        parts.append(f"Preferred countries: {', '.join(selected_countries)}")

    languages = user_input.get('languages', [])
    levels = user_input.get('lang_levels', [])
    if languages:
        formatted = []
        for idx, lang in enumerate(languages):
            level = levels[idx] if idx < len(levels) else None
            formatted.append(f"{lang} ({level})" if level else lang)
        parts.append(f"Language readiness: {', '.join(formatted)}")

    user_gpa = user_input.get('gpa')
    if user_gpa:
        parts.append(f"GPA: {user_gpa:.2f}")

    user_sat = user_input.get('sat')
    if user_sat:
        parts.append(f"SAT: {user_sat}")

    user_ielts = user_input.get('ielts')
    if user_ielts:
        parts.append(f"IELTS: {user_ielts:.1f}")

    budget = user_input.get('budget_max')
    if budget:
        parts.append(f"Budget: {_format_currency(budget)} per year")

    public_pref = user_input.get('public_preference')
    if public_pref == 1:
        parts.append("Prefers public institutions")
    elif public_pref == 0:
        parts.append("Prefers private institutions")

    return "; ".join(parts)


def generate_preference_paragraph(university, country, user_input, user_preferences):
    if not ENABLE_LLM_ENRICHMENT or not GEMINI_ENABLED:
        return None

    summary = build_user_input_summary(user_input, user_preferences)
    if not summary:
        return None

    prompt = f"""You are writing a personalised recommendation for a prospective student.

Student profile summary:
{summary}

Task: Write 3-4 sentences that explain why {university} in {country} could be a good match for this student. Base your response ONLY on the student profile above and reasonable inferences about university life (student communities, clubs, networking, city atmosphere).

Important constraints:
- Do NOT mention tuition fees, scholarship availability, rankings, acceptance rates, or other metrics.
- Focus on aligning the student's interests and readiness with possible experiences they could pursue at the university using general knowledge.
- Keep the tone encouraging and tailored to the student's stated goals.
"""

    try:
        model = get_llm_model('gemini-flash-latest')
        response = model.generate_content(prompt)
        text = (response.text or '').strip()
        if text:
            return text
    except Exception as e:
        print(f"Preference paragraph LLM error for {university}: {e}")

    return None


def build_user_fit_summary(uni_record, user_input, user_preferences):
    note = (user_preferences or '').strip()
    if note:
        return note

    parts = []

    selected_countries = [c for c in user_input.get('countries', []) if c]
    uni_country = uni_record.get('country', '')
    if selected_countries:
        country_list = ", ".join(selected_countries)
        if uni_country and uni_country in selected_countries:
            parts.append(f"You asked for universities in {country_list}, and this option keeps you in {uni_country}.")
        elif uni_country:
            parts.append(f"This adds {uni_country} alongside your preferred countries ({country_list}).")

    languages = user_input.get('languages', [])
    levels = user_input.get('lang_levels', [])
    if languages:
        formatted = []
        for idx, lang in enumerate(languages):
            level = levels[idx] if idx < len(levels) else None
            formatted.append(f"{lang} ({level})" if level else lang)
        parts.append(f"You told us you can study in {', '.join(formatted)}.")

    user_gpa = user_input.get('gpa')
    if user_gpa:
        parts.append(f"A GPA of {user_gpa:.2f} showcases strong academics.")

    user_sat = user_input.get('sat')
    if user_sat:
        parts.append(f"Your SAT score of {user_sat} highlights solid test readiness.")

    user_ielts = user_input.get('ielts')
    if user_ielts:
        parts.append(f"An IELTS score of {user_ielts:.1f} confirms you're comfortable learning in English.")

    budget = user_input.get('budget_max')
    if budget:
        parts.append(f"You've planned for roughly {_format_currency(budget)} per year to support your studies.")

    public_pref = user_input.get('public_preference')
    if public_pref == 1:
        parts.append("You mentioned a preference for public institutions.")
    elif public_pref == 0:
        parts.append("You mentioned a preference for private institutions.")

    if not parts:
        parts.append(f"{uni_record.get('university')} aligns with the academic profile you shared.")

    return " ".join(parts)


def get_ai_description(university_name, country, stats, csv_info=None):
    """Get AI-generated description of the university with quota error handling"""
    if not ENABLE_LLM_ENRICHMENT or not GEMINI_ENABLED:
        return build_csv_fallback_description(university_name, country, stats, csv_info)
    
    # Use provided csv_info or load if not provided
    if csv_info is None:
        csv_info = get_university_info_from_csv(university_name, country)
    
    data_points = []
    if csv_info:
        field_map = {
            'Acceptance rate': csv_info.get('acceptance_rate', 'N/A'),
            'Global ranking': csv_info.get('global_ranking', 'N/A'),
            'National ranking': csv_info.get('national_ranking', 'N/A'),
            'Scholarships': csv_info.get('scholarship', 'N/A'),
            'Language of teaching': csv_info.get('language_of_teaching', 'N/A'),
            'Student-faculty ratio': csv_info.get('student_faculty_ratio', 'N/A'),
            'International tuition': csv_info.get('cost_international', stats.get('international_cost_max', 'N/A')),
        }
        for label, value in field_map.items():
            if value and str(value).strip() and str(value).strip().upper() != 'N/A':
                data_points.append(f"- {label}: {value}")
                            else:
        # Fall back to the matching stats so the LLM still gets some structure
        fallback_map = {
            'Minimum GPA': stats.get('GPA_min'),
            'Minimum SAT': stats.get('SAT_min'),
            'Minimum IELTS': stats.get('IELTS_min'),
            'International tuition (estimate)': stats.get('international_cost_max'),
        }
        for label, value in fallback_map.items():
            if value not in (None, 'N/A', 0):
                data_points.append(f"- {label}: {value}")
    
    data_block = "\n".join(data_points) if data_points else "- No structured data available"
    prompt = f"""Write a polished overview of {university_name} in {country} for prospective international students.

Guidelines:
- Produce a single paragraph of roughly 6-7 sentences (about 180-220 words).
- Cover the university's history or founding context, leading academic or research strengths, notable rankings or achievements, student experience/campus life, and international student support (tuition costs, scholarships, language of instruction).
- Blend trusted public knowledge with the structured data points listed below.
- Keep the tone informative and engaging without marketing fluff.

University data:
{data_block}
"""
    
    model_names = ['gemini-flash-latest', 'gemini-2.5-pro']
    max_retries = 2
    
    for model_name in model_names:
        try:
            model = get_llm_model(model_name)
            for attempt in range(max_retries):
                try:
                    response = model.generate_content(prompt)
                    text = (response.text or '').strip()
                    if not text:
                        continue
                    sentence_count = text.count('.') + text.count('!') + text.count('?')
                    word_count = len(text.split())
                    if sentence_count >= 5 and word_count >= 140:
                        return text
                        if attempt < max_retries - 1:
                        prompt = f"The previous answer was too short. Rewrite with 6-7 sentences and include concrete facts about {university_name}.\n\nUniversity data:\n{data_block}"
                        continue
                    return text
                except Exception as inner_error:
                    if attempt < max_retries - 1:
                        time.sleep(2 ** attempt)
                        continue
                    raise inner_error
        except Exception as model_error:
            if google_exceptions and isinstance(model_error, google_exceptions.ResourceExhausted):
                print(f"Quota exceeded for {model_name}, trying next model or fallback...")
                continue
            print(f"Error with model {model_name}: {model_error}")
            import traceback
            traceback.print_exc()
            continue
        
    print(f"WARNING: All LLM models failed for {university_name}. Using fallback description.")
    return build_csv_fallback_description(university_name, country, stats, csv_info)

def get_short_university_description(university_name, country, csv_info=None):
    """Generate a short, complete university description (2-3 sentences)"""
    if not ENABLE_LLM_ENRICHMENT or not GEMINI_ENABLED:
        return build_csv_short_description(university_name, country, csv_info)
    if not GEMINI_ENABLED:
        return f"{university_name} is a university located in {country}."
    
    # Use provided csv_info or load if not provided
    if csv_info is None:
        csv_info = get_university_info_from_csv(university_name, country)
    
    try:
        # Use flash model (faster, cheaper)
        model_names = ['gemini-flash-latest']
        
        for model_name in model_names:
            try:
                model = get_llm_model(model_name)
                
                # Build context from CSV if available
                context = ""
                if csv_info:
                    context = f"\nKey facts: Located in {country}. Global ranking: {csv_info.get('global_ranking', 'N/A')}. Type: {csv_info.get('public_private', 'N/A')}."
                
                prompt = f"""Write a brief, complete description of {university_name} in {country}.{context}

REQUIREMENTS:
- Write EXACTLY 2-3 complete sentences (no more, no less)
- Include: location, type (public/private), and one notable characteristic
- Be concise but complete - no truncation
- Professional tone
- Do NOT include admission requirements, costs, or rankings

Example format:
"The University of Cambridge is a prestigious public research university located in Cambridge, England. Founded in 1209, it is one of the world's oldest universities and a member of the Russell Group. The university is renowned for its academic excellence across sciences, humanities, and arts."

Your 2-3 sentence description:"""
                
                response = model.generate_content(prompt)
                text = (response.text or "").strip()
                
                # Validate it's actually short (2-3 sentences)
                sentence_count = text.count('.') + text.count('!') + text.count('?')
                word_count = len(text.split())
                
                if text and sentence_count <= 4 and word_count <= 100:
                    return text
                elif text and sentence_count > 4:
                    # Too long, take first 3 sentences
                    sentences = text.split('. ')
                    return '. '.join(sentences[:3]) + '.'
                elif text:
                    return text
                    
            except Exception as e:
                # Check if it's a quota error
                if google_exceptions and isinstance(e, google_exceptions.ResourceExhausted):
                    print(f"Quota exceeded for short description. Using fallback.")
                    break
                print(f"Error generating short description: {e}")
                continue
        
        # Fallback: create short description from CSV data
        if csv_info:
            desc = f"{university_name} is a {csv_info.get('public_private', 'prestigious')} university located in {country}. "
            if csv_info.get('global_ranking') != 'N/A':
                desc += f"It is ranked {csv_info.get('global_ranking')} globally. "
            else:
                desc += f"It offers quality education with diverse programs. "
            return desc
        
        return f"{university_name} is a university located in {country}."
        
    except Exception as e:
        print(f"ERROR getting short description: {e}")
        return f"{university_name} is located in {country}."

def select_top_universities_with_llm(enriched_results, user_preferences, user_input):
    """Use LLM to select top-5 universities based on user preferences and other criteria"""
    if not ENABLE_LLM_ENRICHMENT or not GEMINI_ENABLED:
        return enriched_results[:5]
    if not GEMINI_ENABLED or not enriched_results:
        # If LLM not available or no results, return first 5
        return enriched_results[:5]
    
    if len(enriched_results) <= 5:
        # If 5 or fewer results, return all
        return enriched_results
    
    if not user_preferences or not user_preferences.strip():
        # If no preferences provided, return first 5
        return enriched_results[:5]
    
    # NO FILTERING - Send all universities to LLM
    # Randomize order to avoid position bias
    import random
    enriched_results_randomized = enriched_results.copy()
    random.shuffle(enriched_results_randomized)
    
    print(f"\n{'='*80}")
    print(f"INFO: Sending ALL {len(enriched_results_randomized)} matched universities to LLM in RANDOMIZED order")
    print(f"INFO: LLM will research and evaluate EVERY university before selecting top 5")
    print(f"{'='*80}\n")
    
    try:
        # Use OpenRouter's Gemini 2.0 Flash (single model, no fallback needed)
        model_names = ['google/gemini-2.0-flash-exp:free'] if USE_OPENROUTER else ['gemini-flash-latest', 'gemini-2.5-pro']
        
        for model_name in model_names:
            try:
                model = get_llm_model(model_name)
                
                # Build minimal university data for LLM - NO NUMBERING to avoid position bias
                universities_data = []
                # Use randomized list
                for uni in enriched_results_randomized:
                    # Include name, country, and match score - NO numbering
                    match_score = uni.get('match_score', 100)
                    uni_data = f"• {uni['university']} ({uni['country']}) - Match Score: {match_score}"
                    universities_data.append(uni_data)
                
                universities_text = "\n".join(universities_data)
                
                # Debug: show what we're sending to LLM
                print(f"\nDEBUG: Sending {len(universities_data)} universities to LLM in RANDOMIZED order (no numbering to avoid position bias):")
                print(f"First 10 universities in randomized list:")
                for u in universities_data[:10]:
                    print(f"  {u}")
                if len(universities_data) > 10:
                    print(f"  ... and {len(universities_data) - 10} more universities")
                
                # Build user criteria with country preferences
                selected_countries = user_input.get('countries', [])
                
                if selected_countries:
                    countries_list = ", ".join(selected_countries)
                    countries_text = f"\n\n🔴 MANDATORY REQUIREMENT - COUNTRY SELECTION:\nThe student has selected ONLY these countries: {countries_list}\n\nYou MUST select universities ONLY from these countries:\n"
                    for country in selected_countries:
                        countries_text += f"- {country}\n"
                    countries_text += f"\nAny university from a country NOT listed above is AUTOMATICALLY DISQUALIFIED, no matter how prestigious.\nExample: If student selected 'UK', you can ONLY select UK universities. Technical University of Munich (Germany) is DISQUALIFIED even though it's highly ranked.\n"
                else:
                    countries_text = ""
                
                user_criteria = f"{user_preferences}{countries_text}"
                
                # Debug: Show what criteria is being sent
                print(f"\n{'='*80}")
                print("FULL USER CRITERIA BEING SENT TO LLM:")
                print(user_criteria)
                print(f"{'='*80}\n")
                
                prompt = f"""You are a university selection expert. Your task is to select the 5 BEST universities from the list below.

STUDENT'S VISION AND PREFERENCES:
{user_criteria}

UNIVERSITIES TO ANALYZE (in RANDOM ORDER):
{universities_text}

TASK: Select ONLY the 5 best university names from the list above. DO NOT write explanations yet.

⚠️ CRITICAL INSTRUCTIONS:

1. **EVALUATE EVERY UNIVERSITY**: 
   - Use your knowledge of global rankings (QS, THE, ARWU)
   - Consider academic reputation and research strength
   - The list is RANDOMIZED - don't just pick the first 5

2. **COUNTRY FILTERING (HIGHEST PRIORITY)**:
   - ONLY select universities from the student's selected countries
   - If a university is from a different country, DISQUALIFY it immediately
   - Example: If student selected "UK", Oxford ✓ Cambridge ✓ TUM Munich ✗

3. **SELECTION CRITERIA**:
   - Filter by country FIRST
   - Then rank by: global ranking → program fit → student preferences
   - Select the top 5 highest-ranked universities from selected countries

EXAMPLE:
If student selected "UK" and list has: Bologna (Italy), Oxford (UK), TUM (Germany), Cambridge (UK), Imperial (UK)
→ Filter out: Bologna, TUM (wrong countries)
→ Remaining: Oxford, Cambridge, Imperial
→ Select by ranking: Oxford, Cambridge, Imperial (top 3 UK universities)

OUTPUT FORMAT (ONLY NAMES, NO EXPLANATIONS):
1. University Name
2. University Name
3. University Name
4. University Name
5. University Name

DO NOT ADD:
- No explanations
- No country names
- No descriptions
- JUST the university names, one per line"""
                
                response = model.generate_content(prompt)
                text = (response.text or "").strip()
                
                # Debug: print the LLM response
                print(f"\n{'='*80}")
                print(f"LLM ({model_name}) SELECTED UNIVERSITIES:")
                print(f"{'='*80}")
                print(text)
                print(f"{'='*80}\n")
                
                if text:
                    # Parse the simple list response - just extract university names
                    selected_universities = []
                    lines = text.split('\n')
                    
                    for line in lines:
                        line = line.strip()
                        if not line:
                            continue
                        
                        # Remove numbering (1., 2., etc.) and extract university name
                        # Handle formats: "1. Name", "1) Name", "- Name", "Name"
                        uni_name = line
                        
                        # Remove leading numbers, dots, dashes, parentheses
                        uni_name = re.sub(r'^[\d\.\)\-\*•]+\s*', '', uni_name).strip()
                        
                        # Remove any trailing notes in parentheses like "(UK)"
                        uni_name = re.sub(r'\s*\([^)]*\)\s*$', '', uni_name).strip()
                        
                        if uni_name and len(uni_name) > 3:  # Valid university name
                            selected_universities.append(uni_name)
                    
                    print(f"DEBUG: Parsed {len(selected_universities)} university names: {selected_universities}")
                    
                    # If we didn't find any universities with the numbered format, try to extract from text
                    if not selected_universities:
                        # Try to find university names in the text by matching against enriched_results
                        for uni in enriched_results:
                            if uni['university'].lower() in text.lower():
                                # Try to extract explanation near the university name
                                pattern = re.escape(uni['university'])
                                match = re.search(pattern, text, re.IGNORECASE)
                                if match:
                                    # Get text after the university name (up to next university or end)
                                    start_pos = match.end()
                                    # Find next university name or end of text
                                    next_uni_pos = len(text)
                                    for other_uni in enriched_results:
                                        if other_uni['university'] != uni['university']:
                                            other_match = re.search(re.escape(other_uni['university']), text[start_pos:], re.IGNORECASE)
                                            if other_match:
                                                next_uni_pos = min(next_uni_pos, start_pos + other_match.start())
                                    
                                    explanation = text[start_pos:next_uni_pos].strip()[:300]  # Limit to 300 chars
                                    if explanation:
                                        selected_universities.append({'name': uni['university'], 'explanation': explanation})
                                    else:
                                        selected_universities.append({'name': uni['university'], 'explanation': ''})
                                    
                                    if len(selected_universities) >= 5:
                                        break
                    
                    # Match selected university names to enriched results
                    top_5_results = []
                    for selected_name in selected_universities[:5]:
                        # Try to find exact match first (search in randomized list)
                        found = False
                        for uni in enriched_results_randomized:
                            if uni['university'].lower() == selected_name.lower():
                                top_5_results.append(uni)
                                found = True
                                print(f"✓ Matched: '{selected_name}' → '{uni['university']}' ({uni['country']})")
                                break
                        
                        # If no exact match, try partial match
                        if not found:
                            for uni in enriched_results_randomized:
                                if selected_name.lower() in uni['university'].lower() or uni['university'].lower() in selected_name.lower():
                                    if uni not in top_5_results:
                                        top_5_results.append(uni)
                                        found = True
                                        print(f"✓ Partial match: '{selected_name}' → '{uni['university']}' ({uni['country']})")
                                        break
                        
                        if not found:
                            print(f"✗ Could not match: '{selected_name}'")
                    
                    # VALIDATE: Check if LLM followed country preferences
                    selected_countries = user_input.get('countries', [])
                    if selected_countries and top_5_results:
                        # Normalize country names
                        normalized_selected = [c.strip().lower() for c in selected_countries]
                        
                        # Count how many selected universities are from selected countries
                        valid_selections = []
                        invalid_selections = []
                        
                        for uni in top_5_results:
                            uni_country = uni.get('country', '').strip().lower()
                            is_valid = False
                            
                            for selected in normalized_selected:
                                if selected in uni_country or uni_country in selected:
                                    is_valid = True
                                    break
                                # Handle variations
                                if ('uk' in selected or 'united kingdom' in selected) and ('uk' in uni_country or 'united kingdom' in uni_country or 'britain' in uni_country):
                                    is_valid = True
                                    break
                            
                            if is_valid:
                                valid_selections.append(uni)
                            else:
                                invalid_selections.append(uni)
                        
                        print(f"\n{'='*80}")
                        print(f"VALIDATION: LLM Country Selection")
                        print(f"Selected countries: {selected_countries}")
                        print(f"Valid selections (from selected countries): {len(valid_selections)}")
                        print(f"Invalid selections (from other countries): {len(invalid_selections)}")
                        
                        if invalid_selections:
                            print(f"\n❌ LLM IGNORED COUNTRY PREFERENCE - Selected universities from wrong countries:")
                            for uni in invalid_selections:
                                print(f"  - {uni['university']} ({uni['country']}) - NOT in selected countries!")
                        
                        if valid_selections:
                            print(f"\n✅ Valid selections:")
                            for uni in valid_selections:
                                print(f"  - {uni['university']} ({uni['country']})")
                        
                        # If LLM selected wrong countries, show error but DO NOT override
                        # Let the system fail so we can see what LLM is doing wrong
                        if invalid_selections:
                            print(f"\n❌❌❌ CRITICAL ERROR: LLM FAILED TO FOLLOW COUNTRY FILTER ❌❌❌")
                            print(f"This indicates the prompt needs to be improved")
                            print(f"Proceeding with LLM's selections to see results...")
                            print(f"{'='*80}\n")
                        
                        print(f"{'='*80}\n")
                    
                    if len(top_5_results) >= 3:  # If we found at least 3 matches, use them
                        print(f"INFO: LLM selected {len(top_5_results)} universities based on preferences")
                        return top_5_results[:5]
                    else:
                        print(f"WARNING: LLM selection found only {len(top_5_results)} matches, using first 5 instead")
                        return enriched_results[:5]
                
            except Exception as model_error:
                # Check if it's a quota error
                if google_exceptions and isinstance(model_error, google_exceptions.ResourceExhausted):
                    print(f"Quota exceeded for {model_name} when selecting top universities. Using first 5.")
                    if model_name == model_names[0] and len(model_names) > 1:
                        continue  # Try next model
                    break  # Use fallback
                else:
                    print(f"Error with model {model_name} for university selection: {model_error}")
                    if model_name == model_names[0] and len(model_names) > 1:
                        continue  # Try next model
                    break  # Use fallback
        
        # Fallback: return first 5
        print("WARNING: LLM selection failed, using first 5 universities")
        return enriched_results[:5]
        
    except Exception as e:
        print(f"ERROR in LLM university selection: {e}")
        import traceback
        traceback.print_exc()
        return enriched_results[:5]

def results(request):
    if request.method == 'POST':
        try:
            # Get countries from multi-select dropdown
            countries = request.POST.getlist('countries')
            
            # Get languages from checkboxes
            languages = request.POST.getlist('languages')
            
            # Get language levels from hidden input (created by JavaScript)
            lang_levels_str = request.POST.get('lang_levels', '')
            lang_levels = [ll.strip() for ll in lang_levels_str.split(',') if ll.strip()] if lang_levels_str else []
            
            # Fallback: check if old format is used (comma-separated text input)
            if not countries and request.POST.get('countries'):
                countries = [c.strip() for c in request.POST.get('countries').split(',') if c.strip()]
            if not languages and request.POST.get('languages_list'):
                languages = [l.strip() for l in request.POST.get('languages_list').split(',') if l.strip()]
            
            print(f"DEBUG: Countries: {countries}")
            print(f"DEBUG: Languages: {languages}")
            print(f"DEBUG: Language levels: {lang_levels}")
            
            # Get user preferences
            user_preferences = request.POST.get('preferences', '').strip()
            
            user_input = {
                'countries': countries,
                'languages': languages,
                'lang_levels': lang_levels,
                'ielts': float(request.POST.get('ielts', 0)),
                'gpa': float(request.POST.get('gpa', 0)),
                'sat': int(request.POST.get('sat', 0)),
                'budget_max': float(request.POST.get('budget_max', 0)),
                'public_preference': int(request.POST.get('public_preference', -1)),
            }
            
            csv_path = Path(settings.BASE_DIR) / 'data.csv'
            data = pd.read_csv(csv_path)
            df = uni_find_wrapper.preprocess_university_data(data)
            result = uni_find_wrapper.calculate_match_score(df, user_input)
            
            # Convert to list - include match_score
            results_list = result[['university', 'country', 'language', 'IELTS_min', 'GPA_min', 'SAT_min', 'international_cost_max', 'is_public', 'match_score']].to_dict(orient='records')
            
            print("=" * 80)
            print(f"MATCHING RESULTS: Found {len(results_list)} universities that match criteria")
            print(f"Selected countries: {countries}")
            print("=" * 80)
            print(f"ALL {len(results_list)} MATCHED UNIVERSITIES (No sorting - LLM will prioritize):")
            for idx, uni in enumerate(results_list, 1):
                print(f"  {idx}. {uni['university']} ({uni['country']}) - Match Score: {uni['match_score']}")
            print("=" * 80)
            
            # FIRST: Do quick LLM selection on basic data (before expensive enrichment)
            # Create minimal university list for quick selection - include match scores
            basic_universities = [{'university': uni['university'], 'country': uni['country'], 'match_score': uni['match_score']} for uni in results_list]
            selected_universities_with_explanations = {}
            
            if user_preferences and user_preferences.strip() and ENABLE_LLM_ENRICHMENT and len(results_list) > 5:
                try:
                    print(f"\nSENDING TO LLM: {len(basic_universities)} universities")
                    print(f"User preferences: {user_preferences[:100]}...")
                    
                    # Quick LLM selection on basic data only
                    selected_basic = select_top_universities_with_llm(basic_universities, user_preferences, user_input)
                    
                    print(f"\nLLM SELECTED {len(selected_basic)} UNIVERSITIES:")
                    for idx, u in enumerate(selected_basic, 1):
                        print(f"  {idx}. {u['university']} ({u['country']})")
                    print("=" * 80)
                    
                    # Store explanations for selected universities
                    for u in selected_basic:
                        selected_universities_with_explanations[u['university']] = u.get('preference_explanation', '')
                    # Get names of selected universities
                    selected_names = [u['university'] for u in selected_basic]
                    # Get universities to enrich (only the selected ones)
                    universities_to_enrich = [uni for uni in results_list if uni['university'] in selected_names][:5]
                except Exception as e:
                    print(f"WARNING: Quick LLM selection failed: {e}, using first 5")
                    universities_to_enrich = results_list[:5]
            else:
                universities_to_enrich = results_list[:5]
            
            # Enrich each selected university with AI description, image, and additional data
            enriched_results = []
            try:
                if ENABLE_LLM_ENRICHMENT:
                    long_description_limit = max(0, int(MAX_LONG_DESCRIPTIONS))
                else:
                    long_description_limit = 0
            except (TypeError, ValueError):
                long_description_limit = 0
            long_description_count = 0
            for uni in universities_to_enrich:
                stats = {
                    'GPA_min': uni['GPA_min'],
                    'SAT_min': uni['SAT_min'],
                    'IELTS_min': uni['IELTS_min'],
                    'international_cost_max': uni['international_cost_max'],
                    'is_public': uni.get('is_public')
                }
                
                # Get CSV info for additional data
                csv_info = get_university_info_from_csv(uni['university'], uni['country'])
                
                # Use CSV data if available, otherwise fall back to matching data
                if csv_info:
                    # Use CSV data for more accurate information
                    uni['gpa_avg'] = csv_info.get('gpa_min', uni.get('GPA_min', 'N/A'))
                    uni['ielts_min'] = csv_info.get('ielts_min', uni.get('IELTS_min', 'N/A'))
                    uni['sat_min'] = csv_info.get('sat_min', uni.get('SAT_min', 'N/A'))
                    uni['tuition'] = csv_info.get('cost_international', uni.get('international_cost_max', 'N/A'))
                    uni['tuition_local'] = csv_info.get('cost_local', 'N/A')
                else:
                    # Fallback to matching data
                    uni['gpa_avg'] = uni.get('GPA_min', 'N/A')
                    uni['ielts_min'] = uni.get('IELTS_min', 'N/A')
                    uni['sat_min'] = uni.get('SAT_min', 'N/A')
                    uni['tuition'] = uni.get('international_cost_max', 'N/A')
                    uni['tuition_local'] = 'N/A'
                
                # Add all required fields
                use_long_description = GEMINI_ENABLED and long_description_count < long_description_limit
                if use_long_description:
                    uni['description'] = get_ai_description(uni['university'], uni['country'], stats, csv_info)
                    long_description_count += 1
                else:
                    uni['description'] = build_csv_fallback_description(uni['university'], uni['country'], stats, csv_info)
                uni['short_description'] = get_short_university_description(uni['university'], uni['country'], csv_info)
                uni['image_url'] = get_university_image(uni['university'], uni['country'])
                uni['city'] = get_university_city(uni['university'], uni['country'])
                uni['acceptance_rate'] = csv_info.get('acceptance_rate', 'N/A') if csv_info else 'N/A'
                uni['national_ranking'] = csv_info.get('national_ranking', 'N/A') if csv_info else 'N/A'
                uni['global_ranking'] = csv_info.get('global_ranking', 'N/A') if csv_info else 'N/A'
                uni['languages'] = csv_info.get('language_of_teaching', uni.get('language', 'N/A')) if csv_info else uni.get('language', 'N/A')
                uni['student_faculty_ratio'] = csv_info.get('student_faculty_ratio', 'N/A') if csv_info else 'N/A'
                uni['scholarship'] = csv_info.get('scholarship', 'N/A') if csv_info else 'N/A'
                uni['public_private'] = csv_info.get('public_private', 'N/A') if csv_info else 'N/A'
                uni['website'] = csv_info.get('website', 'N/A') if csv_info else 'N/A'
                
                pref_paragraph = generate_preference_paragraph(uni['university'], uni['country'], user_input, user_preferences)
                if not pref_paragraph:
                    pref_paragraph = build_user_fit_summary(uni, user_input, user_preferences)
                uni['preference_explanation'] = pref_paragraph
                
                enriched_results.append(uni)
            
            # Results are already selected and enriched
            return render(request, 'universities/results.html', {
                'results': enriched_results,
                'gemini_enabled': ENABLE_LLM_ENRICHMENT,
                'total_matches': len(results_list),
                'preferences_used': bool(user_preferences)
            })
        except Exception as e:
            import traceback
            traceback.print_exc()
            return render(request, 'universities/results.html', {'error': str(e)})
    
    return HttpResponse("Invalid request method.", status=405)