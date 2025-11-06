#!/usr/bin/env python
"""
Script to download and cache images for all universities in data.csv
Run this script before starting the server to pre-populate the image cache.

Usage: python download_university_images.py
"""

import pandas as pd
import requests
from pathlib import Path
from urllib.parse import quote, urljoin
import re
import time
from bs4 import BeautifulSoup

# Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_CSV = BASE_DIR.parent / 'data.csv'
CACHE_DIR = BASE_DIR / 'static' / 'images' / 'universities'

def create_safe_filename(university_name):
    """Create a safe filename from university name"""
    safe_name = re.sub(r'[^\w\s-]', '', university_name.lower())
    safe_name = re.sub(r'[-\s]+', '-', safe_name)
    return f"{safe_name}.jpg"

def fetch_wikipedia_image(university_name, country):
    """Try to fetch actual campus photo from Wikipedia (not logos/emblems)"""
    try:
        headers = {
            'User-Agent': 'UniversityMatcherBot/1.0'
        }
        # Try Wikipedia REST API
        for title in [f"{university_name}", f"{university_name} ({country})", f"{university_name}, {country}"]:
            url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{quote(title)}"
            try:
                resp = requests.get(url, timeout=15, headers=headers)
                if resp.status_code == 200:
                    data = resp.json()
                    img = (data.get('originalimage') or {}).get('source') or (data.get('thumbnail') or {}).get('source')
                    if img:
                        # Skip SVG files (usually logos/emblems) and images with "seal", "coat", "emblem", "logo" in URL
                        img_lower = img.lower()
                        if any(word in img_lower for word in ['.svg', 'seal', 'coat', 'emblem', 'logo', 'arms']):
                            print(f"  Skipping emblem/logo: {img[:60]}...")
                            continue
                        print(f"  Found Wikipedia image: {img[:60]}...")
                        return img
            except Exception as inner_e:
                continue  # Try next title variation
    except Exception as e:
        print(f"  Wikipedia error: {str(e)[:100]}")
    return None

def fetch_unsplash_image_direct(university_name, country):
    """Fetch from Unsplash using direct photo API"""
    try:
        # Use Unsplash with better search terms
        # Note: source.unsplash.com returns redirects to actual images
        search_terms = f"{university_name}+campus+building".replace(' ', '+')
        unsplash_url = f"https://source.unsplash.com/800x600/?{search_terms}"
        return unsplash_url
    except Exception as e:
        print(f"  Unsplash error: {str(e)[:100]}")
    return None

def fetch_wikimedia_commons_image(university_name, country):
    """Try to fetch campus photos from Wikimedia Commons"""
    try:
        headers = {'User-Agent': 'UniversityMatcherBot/1.0'}
        # Search Wikimedia Commons for campus photos
        search_url = f"https://commons.wikimedia.org/w/api.php"
        params = {
            'action': 'query',
            'list': 'search',
            'srsearch': f'{university_name} campus building',
            'srnamespace': 6,  # File namespace
            'format': 'json',
            'srlimit': 5
        }
        resp = requests.get(search_url, params=params, timeout=15, headers=headers)
        if resp.status_code == 200:
            data = resp.json()
            results = data.get('query', {}).get('search', [])
            for result in results:
                filename = result.get('title', '').replace('File:', '')
                if filename:
                    # Get file info to get actual image URL
                    file_params = {
                        'action': 'query',
                        'titles': f'File:{filename}',
                        'prop': 'imageinfo',
                        'iiprop': 'url',
                        'format': 'json'
                    }
                    file_resp = requests.get(search_url, params=file_params, timeout=15, headers=headers)
                    if file_resp.status_code == 200:
                        file_data = file_resp.json()
                        pages = file_data.get('query', {}).get('pages', {})
                        for page in pages.values():
                            imageinfo = page.get('imageinfo', [])
                            if imageinfo and len(imageinfo) > 0:
                                img_url = imageinfo[0].get('url')
                                if img_url and not any(word in img_url.lower() for word in ['.svg', 'seal', 'coat', 'emblem', 'logo']):
                                    print(f"  Found Wikimedia image: {img_url[:60]}...")
                                    return img_url
    except Exception as e:
        print(f"  Wikimedia error: {str(e)[:100]}")
    return None

def fetch_from_university_website(university_url):
    """Intelligently scrape campus images from university's official website"""
    if not university_url or university_url == 'N/A' or not university_url.startswith('http'):
        return None
    
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        }
        
        print(f"  Visiting: {university_url[:60]}...")
        resp = requests.get(university_url, timeout=20, headers=headers)
        
        if resp.status_code != 200:
            print(f"  Website returned status {resp.status_code}")
            return None
        
        soup = BeautifulSoup(resp.content, 'html.parser')
        
        # Priority 1: Look for hero/banner images (usually campus shots)
        hero_candidates = []
        for selector in ['section.hero img', 'div.hero img', 'div.banner img', 'header img', 
                        '[class*="hero"] img', '[class*="banner"] img', '[class*="slider"] img',
                        '[class*="carousel"] img', '[id*="hero"] img']:
            hero_images = soup.select(selector)
            hero_candidates.extend(hero_images)
        
        print(f"  Found {len(hero_candidates)} hero/banner images")
        
        for img in hero_candidates:
            src = img.get('src') or img.get('data-src') or img.get('data-lazy')
            if src:
                src = urljoin(university_url, src)
                src_lower = src.lower()
                # Skip logos/icons
                if not any(word in src_lower for word in ['.svg', 'logo', 'icon', 'seal', 'emblem', 'coat']):
                    if any(ext in src_lower for ext in ['.jpg', '.jpeg', '.png', '.webp']):
                        print(f"  ✓ Found hero image: {src[:80]}...")
                        return src
        
        # Priority 2: Look for Open Graph image (og:image) - usually campus photo
        og_image = soup.find('meta', property='og:image')
        if og_image and og_image.get('content'):
            src = urljoin(university_url, og_image['content'])
            src_lower = src.lower()
            if not any(word in src_lower for word in ['.svg', 'logo', 'seal', 'emblem']) and \
               any(ext in src_lower for ext in ['.jpg', '.jpeg', '.png', '.webp']):
                print(f"  ✓ Found OG image: {src[:80]}...")
                return src
        
        # Priority 3: Images with campus keywords in alt/src
        campus_keywords = ['campus', 'building', 'architecture', 'aerial', 'view', 'main building', 'library', 'hall']
        all_images = soup.find_all('img')
        print(f"  Scanning {len(all_images)} total images for campus keywords...")
        
        scored_images = []
        for img in all_images:
            src = img.get('src') or img.get('data-src') or img.get('data-lazy')
            alt = (img.get('alt') or '').lower()
            title = (img.get('title') or '').lower()
            
            if not src:
                continue
            
            src = urljoin(university_url, src)
            src_lower = src.lower()
            
            # Skip bad images
            if any(word in src_lower for word in ['.svg', 'logo', 'icon', 'seal', 'emblem', 'coat', 'avatar', 'flag']):
                continue
            
            # Must be an image format
            if not any(ext in src_lower for ext in ['.jpg', '.jpeg', '.png', '.webp']):
                continue
            
            # Score based on keywords
            score = 0
            for keyword in campus_keywords:
                if keyword in alt:
                    score += 3
                if keyword in title:
                    score += 2
                if keyword in src_lower:
                    score += 1
            
            # Bonus for larger file sizes in URL (full size images)
            if any(size in src_lower for size in ['1920', '1080', '2000', 'large', 'full']):
                score += 2
            
            if score > 0:
                scored_images.append((score, src))
        
        # Return highest scored image
        if scored_images:
            scored_images.sort(reverse=True, key=lambda x: x[0])
            best_image = scored_images[0][1]
            print(f"  ✓ Found best campus image (score {scored_images[0][0]}): {best_image[:80]}...")
            return best_image
        
        # Priority 4: First large image that's not a logo
        print(f"  No campus-specific images, trying first suitable image...")
        for img in all_images[:20]:
            src = img.get('src') or img.get('data-src')
            if src:
                src = urljoin(university_url, src)
                src_lower = src.lower()
                # Must be photo format and not a logo
                if any(ext in src_lower for ext in ['.jpg', '.jpeg', '.png', '.webp']) and \
                   not any(word in src_lower for word in ['.svg', 'logo', 'icon', 'seal', 'emblem', 'coat']):
                    # Prefer larger images
                    if any(size in src_lower for size in ['large', 'full', '1920', '1080', '2000']) or \
                       not any(size in src_lower for size in ['thumb', 'small', '150', '200']):
                        print(f"  Using first suitable: {src[:80]}...")
                        return src
                    
    except Exception as e:
        print(f"  Website error: {str(e)[:100]}")
    return None

def fetch_generic_university_image(country):
    """Fetch a generic university/campus image based on country"""
    try:
        # Use Lorem Picsum for consistent placeholder images
        seed = abs(hash(country)) % 1000
        return f"https://picsum.photos/seed/{seed}/800/600"
    except Exception:
        return None

def download_image(url, save_path):
    """Download image from URL and save to path"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, timeout=20, stream=True, allow_redirects=True, headers=headers)
        if response.status_code == 200:
            # Check if it's actually an image
            content_type = response.headers.get('content-type', '')
            if 'image' in content_type or len(response.content) > 1000:  # At least 1KB
                with open(save_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                # Verify file was created and has content
                if save_path.exists() and save_path.stat().st_size > 1000:
                    return True
                else:
                    # Remove invalid file
                    if save_path.exists():
                        save_path.unlink()
    except Exception as e:
        print(f"  Download error: {str(e)[:100]}")
    return False

def main():
    # Create cache directory
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Cache directory: {CACHE_DIR}")
    
    # Check if data.csv exists
    if not DATA_CSV.exists():
        print(f"ERROR: {DATA_CSV} not found!")
        return
    
    # Read CSV
    print(f"Reading {DATA_CSV}...")
    df = pd.read_csv(DATA_CSV)
    
    if 'university' not in df.columns or 'country' not in df.columns:
        print("ERROR: CSV must have 'university' and 'country' columns")
        return
    
    total = len(df)
    print(f"Found {total} universities to process\n")
    
    success_count = 0
    cached_count = 0
    failed_count = 0
    
    for idx, row in df.iterrows():
        university_name = row['university']
        country = row['country']
        university_url = row.get('link', 'N/A') if 'link' in row else 'N/A'
        
        print(f"[{idx+1}/{total}] {university_name} ({country})")
        
        # Create filename
        filename = create_safe_filename(university_name)
        save_path = CACHE_DIR / filename
        
        # Check if already cached
        if save_path.exists():
            print(f"  ✓ Already cached")
            cached_count += 1
            continue
        
        # Try to fetch image
        image_url = None
        
        # FIRST: Try university's official website
        if university_url and university_url != 'N/A':
            print(f"  Trying official website...")
            image_url = fetch_from_university_website(university_url)
        
        # Try Wikipedia (filtering out logos)
        if not image_url:
            print(f"  Trying Wikipedia...")
            image_url = fetch_wikipedia_image(university_name, country)
        
        # Try Wikimedia Commons for campus photos
        if not image_url:
            print(f"  Trying Wikimedia Commons...")
            image_url = fetch_wikimedia_commons_image(university_name, country)
        
        # Try Unsplash as fallback
        if not image_url:
            print(f"  Trying Unsplash...")
            image_url = fetch_unsplash_image_direct(university_name, country)
        
        # Last resort: generic image
        if not image_url:
            print(f"  Using generic image...")
            image_url = fetch_generic_university_image(country)
        
        # Download and save
        if image_url:
            print(f"  Downloading from: {image_url[:60]}...")
            if download_image(image_url, save_path):
                print(f"  ✓ Successfully cached")
                success_count += 1
            else:
                print(f"  ✗ Download failed")
                failed_count += 1
        else:
            print(f"  ✗ No image source found")
            failed_count += 1
        
        # Small delay to avoid rate limiting
        if idx < total - 1:
            time.sleep(0.5)
    
    print(f"\n{'='*60}")
    print(f"SUMMARY:")
    print(f"  Total universities: {total}")
    print(f"  Already cached: {cached_count}")
    print(f"  Successfully downloaded: {success_count}")
    print(f"  Failed: {failed_count}")
    print(f"{'='*60}")
    print(f"\nImages saved to: {CACHE_DIR}")

if __name__ == '__main__':
    main()

