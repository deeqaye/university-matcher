#!/usr/bin/env python
"""
Test script to download images for a few universities to verify it works
"""

import pandas as pd
import requests
from pathlib import Path
from urllib.parse import quote
import re

# Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_CSV = BASE_DIR.parent / 'data.csv'
CACHE_DIR = BASE_DIR / 'static' / 'images' / 'universities'

def create_safe_filename(university_name):
    safe_name = re.sub(r'[^\w\s-]', '', university_name.lower())
    safe_name = re.sub(r'[-\s]+', '-', safe_name)
    return f"{safe_name}.jpg"

def test_download():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    
    # Test universities
    test_universities = [
        ("Harvard University", "USA"),
        ("University of Oxford", "UK"),
        ("University of Vienna", "Austria"),
    ]
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    
    for uni_name, country in test_universities:
        print(f"\n{'='*60}")
        print(f"Testing: {uni_name} ({country})")
        filename = create_safe_filename(uni_name)
        save_path = CACHE_DIR / filename
        
        if save_path.exists():
            save_path.unlink()  # Remove for testing
        
        # Try Wikipedia with filtering
        try:
            url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{quote(uni_name)}"
            print(f"1. Wikipedia URL: {url}")
            resp = requests.get(url, timeout=15, headers=headers)
            print(f"   Status: {resp.status_code}")
            if resp.status_code == 200:
                data = resp.json()
                img = (data.get('originalimage') or {}).get('source') or (data.get('thumbnail') or {}).get('source')
                if img:
                    print(f"   Image URL: {img[:80]}...")
                    # Check if it's a logo/emblem
                    if any(word in img.lower() for word in ['.svg', 'seal', 'coat', 'emblem', 'logo', 'arms']):
                        print(f"   ⚠️ SKIPPED: This is a logo/emblem, not a campus photo")
                    else:
                        # Download
                        img_resp = requests.get(img, timeout=20, stream=True, headers=headers)
                        if img_resp.status_code == 200:
                            with open(save_path, 'wb') as f:
                                for chunk in img_resp.iter_content(8192):
                                    if chunk:
                                        f.write(chunk)
                            if save_path.exists() and save_path.stat().st_size > 1000:
                                print(f"   ✓ SUCCESS: Saved {save_path.stat().st_size} bytes to {filename}")
                            else:
                                print(f"   ✗ FAILED: File too small")
                        else:
                            print(f"   ✗ FAILED: Download status {img_resp.status_code}")
                else:
                    print(f"   ✗ No image found")
        except Exception as e:
            print(f"   ✗ ERROR: {e}")
        
        # Try Wikimedia Commons
        if not save_path.exists():
            print(f"2. Trying Wikimedia Commons...")
            try:
                search_url = "https://commons.wikimedia.org/w/api.php"
                params = {
                    'action': 'query',
                    'list': 'search',
                    'srsearch': f'{uni_name} campus',
                    'srnamespace': 6,
                    'format': 'json',
                    'srlimit': 3
                }
                resp = requests.get(search_url, params=params, timeout=15, headers=headers)
                if resp.status_code == 200:
                    data = resp.json()
                    results = data.get('query', {}).get('search', [])
                    print(f"   Found {len(results)} results")
                    for r in results[:1]:  # Try first result
                        print(f"   File: {r.get('title', 'N/A')[:50]}...")
            except Exception as e:
                print(f"   ✗ ERROR: {e}")
        
        # Try picsum as fallback
        if not save_path.exists():
            print(f"3. Trying Lorem Picsum (generic)...")
            seed = abs(hash(uni_name)) % 1000
            picsum_url = f"https://picsum.photos/seed/{seed}/800/600"
            try:
                resp = requests.get(picsum_url, timeout=15, stream=True, headers=headers, allow_redirects=True)
                if resp.status_code == 200:
                    with open(save_path, 'wb') as f:
                        for chunk in resp.iter_content(8192):
                            if chunk:
                                f.write(chunk)
                    if save_path.exists() and save_path.stat().st_size > 1000:
                        print(f"   ✓ SUCCESS (Picsum): Saved {save_path.stat().st_size} bytes")
                    else:
                        print(f"   ✗ FAILED: File too small")
                else:
                    print(f"   ✗ FAILED: Status {resp.status_code}")
            except Exception as e:
                print(f"   ✗ ERROR: {e}")

if __name__ == '__main__':
    test_download()

