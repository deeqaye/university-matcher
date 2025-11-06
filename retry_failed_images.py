#!/usr/bin/env python
"""
Retry downloading images for universities that failed
Uses better fallback strategies
"""

import pandas as pd
import requests
from pathlib import Path
from urllib.parse import quote, urljoin
import re
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO

# Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_CSV = BASE_DIR.parent / 'data.csv'
CACHE_DIR = BASE_DIR / 'static' / 'images' / 'universities'

def create_safe_filename(university_name):
    safe_name = re.sub(r'[^\w\s-]', '', university_name.lower())
    safe_name = re.sub(r'[-\s]+', '-', safe_name)
    return f"{safe_name}.jpg"

def create_placeholder_image(university_name, country, save_path):
    """Create a nice colored placeholder image with university name"""
    try:
        # Create image
        img = Image.new('RGB', (800, 600), color=(102, 126, 234))  # Purple background
        draw = ImageDraw.Draw(img)
        
        # Try to use a nice font, fallback to default
        try:
            font_large = ImageFont.truetype("arial.ttf", 48)
            font_small = ImageFont.truetype("arial.ttf", 24)
        except:
            font_large = ImageFont.load_default()
            font_small = ImageFont.load_default()
        
        # Draw university name (wrap if too long)
        words = university_name.split()
        lines = []
        current_line = []
        
        for word in words:
            current_line.append(word)
            test_line = ' '.join(current_line)
            if len(test_line) > 25:  # Max characters per line
                if len(current_line) > 1:
                    current_line.pop()
                    lines.append(' '.join(current_line))
                    current_line = [word]
                else:
                    lines.append(test_line)
                    current_line = []
        
        if current_line:
            lines.append(' '.join(current_line))
        
        # Center text
        y_offset = 250
        for line in lines:
            bbox = draw.textbbox((0, 0), line, font=font_large)
            text_width = bbox[2] - bbox[0]
            x = (800 - text_width) // 2
            draw.text((x, y_offset), line, fill=(255, 255, 255), font=font_large)
            y_offset += 60
        
        # Draw country
        bbox = draw.textbbox((0, 0), country, font=font_small)
        text_width = bbox[2] - bbox[0]
        x = (800 - text_width) // 2
        draw.text((x, y_offset + 20), country, fill=(230, 230, 250), font=font_small)
        
        # Save
        img.save(save_path, 'JPEG', quality=85)
        return True
    except Exception as e:
        print(f"  Error creating placeholder: {e}")
        return False

def download_from_url(url, save_path):
    """Download image from URL"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, timeout=20, stream=True, allow_redirects=True, headers=headers)
        if response.status_code == 200:
            content_type = response.headers.get('content-type', '')
            if 'image' in content_type or len(response.content) > 1000:
                with open(save_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                if save_path.exists() and save_path.stat().st_size > 1000:
                    return True
                elif save_path.exists():
                    save_path.unlink()
        return False
    except Exception as e:
        print(f"  Download error: {str(e)[:80]}")
        return False

def fetch_picsum_deterministic(university_name):
    """Get a deterministic picsum image based on university name"""
    seed = abs(hash(university_name)) % 1000
    return f"https://picsum.photos/seed/uni{seed}/800/600"

def main():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    
    if not DATA_CSV.exists():
        print(f"ERROR: {DATA_CSV} not found!")
        return
    
    df = pd.read_csv(DATA_CSV)
    
    # Find universities without cached images
    missing = []
    for idx, row in df.iterrows():
        university_name = row['university']
        country = row['country']
        filename = create_safe_filename(university_name)
        cache_path = CACHE_DIR / filename
        
        if not cache_path.exists():
            missing.append((university_name, country))
    
    print(f"Found {len(missing)} universities without images\n")
    
    if len(missing) == 0:
        print("All universities have images!")
        return
    
    success = 0
    failed = 0
    
    for idx, (uni_name, country) in enumerate(missing, 1):
        print(f"[{idx}/{len(missing)}] {uni_name} ({country})")
        filename = create_safe_filename(uni_name)
        save_path = CACHE_DIR / filename
        
        downloaded = False
        
        # Try Lorem Picsum first (most reliable)
        print(f"  Trying Lorem Picsum...")
        picsum_url = fetch_picsum_deterministic(uni_name)
        if download_from_url(picsum_url, save_path):
            print(f"  ✓ Downloaded from Picsum")
            success += 1
            downloaded = True
        
        # If picsum fails, create custom placeholder
        if not downloaded:
            print(f"  Creating custom placeholder...")
            if create_placeholder_image(uni_name, country, save_path):
                print(f"  ✓ Created custom placeholder")
                success += 1
                downloaded = True
        
        if not downloaded:
            print(f"  ✗ Still failed")
            failed += 1
    
    print(f"\n{'='*60}")
    print(f"RETRY SUMMARY:")
    print(f"  Missing images: {len(missing)}")
    print(f"  Successfully fixed: {success}")
    print(f"  Still failing: {failed}")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()

