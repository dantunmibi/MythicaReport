#!/usr/bin/env python3
"""
🔮 Download & Manage Copyright-Free Music Library - MYTHICA REPORT
V3.0 - SCRAPER-BASED: More resilient to hotlink protection.
"""

import os
import json
import requests
import hashlib
from datetime import datetime
import time

# This import will be used by the scraper function
try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False

TMP = os.getenv("GITHUB_WORKSPACE", ".") + "/tmp"
MUSIC_DIR = os.path.join(TMP, "music")
os.makedirs(MUSIC_DIR, exist_ok=True)

# ✅ NEW ARCHITECTURE: Using PAGE URLs, not direct .mp3 links.
# This allows us to scrape the real download link, defeating hotlink protection.
MUSIC_LIBRARY = {
    'dark_mystery': {'name': 'Dark Mystery Theme', 'page_url': 'https://pixabay.com/music/ambient-dark-ambient-108851/', 'scenes': ['investigation', 'unsolved'], 'volume_default': 0.18},
    'mysterious_investigation': {'name': 'Mystery Detective', 'page_url': 'https://pixabay.com/music/main-title-night-detective-116999/', 'scenes': ['investigation', 'true_crime'], 'volume_default': 0.20},
    'suspense_build': {'name': 'Dark Tension Builder', 'page_url': 'https://pixabay.com/music/main-title-the-suspense-15617/', 'scenes': ['suspense', 'paranormal'], 'volume_default': 0.17},
    'creeping_tension': {'name': 'Creeping Horror', 'page_url': 'https://pixabay.com/music/horror-scene-creepy-horror-background-94295/', 'scenes': ['suspense', 'cryptids'], 'volume_default': 0.15},
    'anxious_suspense': {'name': 'Anxious Tension', 'page_url': 'https://pixabay.com/music/suspense-horror-ambience-183248/', 'scenes': ['suspense', 'paranormal'], 'volume_default': 0.14},
    'paranormal_ambient': {'name': 'Paranormal Atmosphere', 'page_url': 'https://pixabay.com/music/ambient-mystery-noir-ambient-124013/', 'scenes': ['paranormal', 'supernatural'], 'volume_default': 0.16},
    'dramatic_reveal': {'name': 'Dramatic Discovery', 'page_url': 'https://pixabay.com/music/cinematic-dramatic-inspirational-142005/', 'scenes': ['revelation', 'true_crime'], 'volume_default': 0.24},
    'archaeological_discovery': {'name': 'Ancient Discovery', 'page_url': 'https://pixabay.com/music/cinematic-epic-orchestral-116227/', 'scenes': ['ancient', 'revelation'], 'volume_default': 0.24},
    'hidden_truth': {'name': 'Hidden Secrets', 'page_url': 'https://pixabay.com/music/suspense-dark-secrets-170795/', 'scenes': ['conspiracy', 'unsolved'], 'volume_default': 0.19},
    'creature_encounter': {'name': 'Unknown Entity', 'page_url': 'https://pixabay.com/music/ambient-dark-forest-ambient-184304/', 'scenes': ['cryptids', 'forest'], 'volume_default': 0.18},
}

# --- CACHE AND HASH FUNCTIONS ---
def get_music_cache_path(): return os.path.join(MUSIC_DIR, "music_cache.json")
def load_music_cache():
    cache_path = get_music_cache_path()
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'r') as f: return json.load(f)
        except: return {}
    return {}
def save_music_cache(cache):
    with open(get_music_cache_path(), 'w') as f: json.dump(cache, f, indent=2)
def get_track_hash(track_key): return hashlib.md5(track_key.encode()).hexdigest()[:12]


# In download_music.py

def scrape_pixabay_download_link(page_url: str) -> str | None:
    """
    V2: Uses a headless browser (Selenium) to defeat anti-bot measures and scrape the link.
    """
    if not BS4_AVAILABLE:
        print("      ⚠️ BeautifulSoup4 is not installed. Cannot scrape.")
        return None
    try:
        from selenium import webdriver
        from selenium.webdriver.common.by import By
        from selenium.webdriver.chrome.service import Service as ChromeService
        from webdriver_manager.chrome import ChromeDriverManager

        print("      🚀 Initializing headless browser (Selenium)...")
        options = webdriver.ChromeOptions()
        options.add_argument("--headless")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36")
        
        # Automatically downloads and manages the correct chromedriver
        driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()), options=options)

        print(f"      Navigating to page: {page_url}")
        driver.get(page_url)
        time.sleep(3) # Wait for JavaScript to load the page content

        soup = BeautifulSoup(driver.page_source, 'html.parser')
        driver.quit() # Close the browser as soon as we have the HTML

        download_button = soup.find('a', href=lambda href: href and '/download/audio/' in href)

        if download_button and download_button.get('href'):
            link = download_button['href']
            full_link = 'https://pixabay.com' + link if link.startswith('/') else link
            return full_link.split('?')[0]
        else:
            print("      ⚠️ Could not find download button on page via Selenium.")
            return None

    except ImportError:
        print("      ⚠️ Selenium is not installed. `pip install selenium webdriver-manager`")
        return None
    except Exception as e:
        print(f"      ⚠️ Error during headless browser scraping: {e}")
        return None

# ✅ OVERHAULED DOWNLOAD FUNCTION ---
def download_track(track_key, track_info, force=False):
    track_hash = get_track_hash(track_key)
    filename = f"{track_key}_{track_hash}.mp3"
    filepath = os.path.join(MUSIC_DIR, filename)

    if not force and os.path.exists(filepath) and os.path.getsize(filepath) > 10000:
        print(f"✅ Using cached: {track_info['name']}")
        return filepath

    print(f"📥 Downloading: {track_info['name']}...")
    
    page_url = track_info.get('page_url')
    if not page_url:
        print(f"   ❌ No page_url defined for {track_key}")
        return None
        
    download_link = scrape_pixabay_download_link(page_url)
    
    if not download_link:
        print(f"   ❌ Could not find a valid download link for {track_key}")
        return None

    print(f"      ✅ Found download link: {download_link[:70]}...")
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36', 'Referer': page_url}
        response = requests.get(download_link, headers=headers, timeout=120, stream=True, allow_redirects=True)
        response.raise_for_status()

        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192): f.write(chunk)
        
        if os.path.exists(filepath) and os.path.getsize(filepath) > 50000:
            print(f"      ✅ Download successful ({os.path.getsize(filepath) / 1024:.1f} KB).")
            cache = load_music_cache()
            cache[track_key] = {'local_path': filepath, 'downloaded_at': datetime.now().isoformat(), 'source_page': page_url}
            save_music_cache(cache)
            return filepath
        else:
            if os.path.exists(filepath): os.remove(filepath)
            print("      ⚠️ Download failed or file was empty.")
    except requests.exceptions.RequestException as e:
        print(f"      ⚠️ Download failed: {e}")
    return None

def get_music_for_scene(scene_type, content_type='general'):
    scene_priority = {
        'investigation': ['dark_mystery', 'mysterious_investigation'],
        'paranormal': ['paranormal_ambient', 'anxious_suspense'],
        'suspense': ['suspense_build', 'creeping_tension'],
        'revelation': ['dramatic_reveal', 'archaeological_discovery'],
        'ancient': ['archaeological_discovery', 'dark_mystery'],
        'true_crime': ['mysterious_investigation', 'dramatic_reveal'],
        'conspiracy': ['hidden_truth', 'dark_mystery'],
        'cryptids': ['creature_encounter', 'creeping_tension'],
        'unsolved': ['dark_mystery', 'mysterious_investigation'],
        'general': ['dark_mystery', 'suspense_build']
    }
    priority_tracks = scene_priority.get(scene_type, scene_priority['general'])
    
    for track_key in priority_tracks:
        if track_key in MUSIC_LIBRARY:
            track_info = MUSIC_LIBRARY[track_key]
            local_path = download_track(track_key, track_info)
            if local_path: return track_key, local_path, track_info['volume_default']

    print(f"⚠️ Priority tracks for '{scene_type}' unavailable, trying any track...")
    for track_key, track_info in MUSIC_LIBRARY.items():
        local_path = download_track(track_key, track_info)
        if local_path: return track_key, local_path, track_info['volume_default']
    
    return None, None, 0.18

def download_all_music():
    print("\n" + "="*70 + "\n🔮 DOWNLOADING MYSTERY MUSIC LIBRARY (SCRAPER MODE)\n" + "="*70)
    successful, failed = 0, []
    for i, (track_key, track_info) in enumerate(MUSIC_LIBRARY.items(), 1):
        print(f"\n📀 [{i}/{len(MUSIC_LIBRARY)}] {track_info['name']}")
        if download_track(track_key, track_info, force=True):
            successful += 1
        else:
            failed.append(track_key)
    
    print("\n" + "="*70 + "\n📊 DOWNLOAD SUMMARY\n" + "="*70)
    print(f"✅ Successful: {successful}/{len(MUSIC_LIBRARY)}")
    if failed:
        print(f"❌ Failed: {len(failed)}/{len(MUSIC_LIBRARY)} -> {', '.join(failed)}")
    print(f"\n💾 Music library location: {MUSIC_DIR}")

if __name__ == "__main__":
    import sys
    if '--download-all' in sys.argv:
        download_all_music()
    else:
        print("Downloading essential tracks for Mythica Report...")
        essential = ['dark_mystery', 'suspense_build', 'paranormal_ambient', 'dramatic_reveal']
        for track_key in essential:
            if track_key in MUSIC_LIBRARY:
                download_track(track_key, MUSIC_LIBRARY[track_key])
        print("\n✅ Essential mystery tracks ready! 🔮")