#!/usr/bin/env python3
"""
🔮 Download & Manage Copyright-Free Music Library - MYTHICA REPORT
V5.0 - JSON-LD SCRAPER: The fastest and most reliable method.
"""

import os
import json
import requests
import hashlib
from datetime import datetime

# This import is required for the scraper function
try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False

TMP = os.getenv("GITHUB_WORKSPACE", ".") + "/tmp"
MUSIC_DIR = os.path.join(TMP, "music")
os.makedirs(MUSIC_DIR, exist_ok=True)

# ✅ CORRECTED: Using PAGE URLs for the JSON-LD scraper.
MUSIC_LIBRARY = {
    'dark_mystery': {'name': 'Dark Mystery Theme', 'page_url': 'https://pixabay.com/music/ambient-dark-ambient-108851/', 'scenes': ['investigation', 'unsolved'], 'volume_default': 0.18},
    'mysterious_investigation': {'name': 'Mystery Detective', 'page_url': 'https://pixabay.com/music/main-title-night-detective-116999/', 'scenes': ['investigation', 'true_crime'], 'volume_default': 0.20},
    'suspense_build': {'name': 'Dark Tension Builder', 'page_url': 'https://pixabay.com/music/main-title-the-suspense-15617/', 'scenes': ['suspense', 'paranormal'], 'volume_default': 0.17},
    'creeping_tension': {'name': 'Creeping Horror', 'page_url': 'https://pixabay.com/music/horror-scene-creepy-horror-background-94295/', 'scenes': ['suspense', 'cryptids'], 'volume_default': 0.15},
    'paranormal_ambient': {'name': 'Paranormal Atmosphere', 'page_url': 'https://pixabay.com/music/ambient-mystery-noir-ambient-124013/', 'scenes': ['paranormal', 'supernatural'], 'volume_default': 0.16},
    'dramatic_reveal': {'name': 'Dramatic Discovery', 'page_url': 'https://pixabay.com/music/cinematic-dramatic-inspirational-142005/', 'scenes': ['revelation', 'true_crime'], 'volume_default': 0.24},
}

# --- Standard Cache/Hash functions ---
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

# --- THE NEW, ROBUST DOWNLOAD LOGIC ---

# Create a single, persistent session for all requests
session = requests.Session()
session.headers.update({
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.9',
})

def get_download_url_from_page(page_url: str) -> str | None:
    """Scrapes the page to find the embedded JSON-LD download link."""
    if not BS4_AVAILABLE:
        print("      ⚠️ BeautifulSoup4 is not installed.")
        return None
    try:
        print(f"      Scraping page: {page_url}")
        response = session.get(page_url, timeout=30)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        json_ld_script = soup.find('script', type='application/ld+json')
        if not json_ld_script:
            print("      ⚠️ Could not find JSON-LD script tag.")
            return None
            
        data = json.loads(json_ld_script.string)
        content_url = data.get('contentUrl')
        if content_url:
            return content_url.split('?')[0]
        else:
            print("      ⚠️ 'contentUrl' not found in JSON-LD.")
            return None
    except Exception as e:
        print(f"      ⚠️ Error scraping page: {e}")
        return None

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
        
    download_link = get_download_url_from_page(page_url)
    if not download_link:
        print(f"   ❌ Failed to get download link for {track_key}")
        return None

    print(f"      ✅ Found download link via scraper.")
    try:
        # Use the same session to download the file, which carries over any necessary cookies
        response = session.get(download_link, timeout=120, stream=True, allow_redirects=True)
        response.raise_for_status()

        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        if os.path.exists(filepath) and os.path.getsize(filepath) > 50000:
            print(f"      ✅ Download successful ({os.path.getsize(filepath) / 1024:.1f} KB).")
            cache = load_music_cache()
            cache[track_key] = {'local_path': filepath, 'downloaded_at': datetime.now().isoformat()}
            save_music_cache(cache)
            return filepath
        else:
            if os.path.exists(filepath): os.remove(filepath)
            print("      ⚠️ Downloaded file was empty or too small.")
    except Exception as e:
        print(f"      ⚠️ Download request failed: {e}")
    return None

def get_music_for_scene(scene_type, content_type='general'):
    scene_priority = {
        'investigation': ['dark_mystery', 'mysterious_investigation'],
        'suspense': ['suspense_build', 'creeping_tension'],
        'paranormal': ['paranormal_ambient'],
        'revelation': ['dramatic_reveal'],
        # ... add other mappings as needed
        'general': ['dark_mystery', 'suspense_build']
    }
    priority_tracks = scene_priority.get(scene_type, scene_priority['general'])
    
    for track_key in priority_tracks:
        if track_key in MUSIC_LIBRARY:
            track_info = MUSIC_LIBRARY[track_key]
            local_path = download_track(track_key, track_info)
            if local_path:
                return track_key, local_path, track_info.get('volume_default', 0.18)

    print(f"⚠️ Priority tracks for '{scene_type}' failed, trying any available track...")
    for track_key, track_info in MUSIC_LIBRARY.items():
        local_path = download_track(track_key, track_info)
        if local_path:
            return track_key, local_path, track_info.get('volume_default', 0.18)
    
    return None, None, 0.18

def download_all_music():
    print("\n" + "="*70 + "\n🔮 DOWNLOADING MYSTERY MUSIC LIBRARY (JSON-LD SCRAPER)\n" + "="*70)
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