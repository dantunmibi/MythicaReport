#!/usr/bin/env python3
"""
🔮 Download & Manage Copyright-Free Music Library - MYTHICA REPORT
Automatically downloads and caches mysterious, suspenseful music tracks
"""

import os
import json
import requests
from pathlib import Path
import hashlib
from datetime import datetime

TMP = os.getenv("GITHUB_WORKSPACE", ".") + "/tmp"
MUSIC_DIR = os.path.join(TMP, "music")
os.makedirs(MUSIC_DIR, exist_ok=True)

# 🔮 COPYRIGHT-FREE MYSTERY MUSIC LIBRARY - HYBRID APPROACH
# Primary: Pixabay (CC0) | Backup: Incompetech (CC BY 4.0)
# Each track has a backup URL for reliability

MUSIC_LIBRARY = {
    # 🌑 DARK MYSTERIOUS (Investigation/Discovery)
    'dark_mystery': {
        'name': 'Dark Mystery Theme',
        'url': 'https://cdn.pixabay.com/download/audio/2022/11/22/audio_20cb6ce0c5.mp3?filename=dark-ambient-108851.mp3',
        'backup_url': 'https://incompetech.com/music/royalty-free/mp3-royaltyfree/Killers.mp3',
        'duration': 108,
        'emotion': 'mysterious, dark, investigative',
        'scenes': ['investigation', 'unsolved'],
        'volume_default': 0.08
    },

    'mysterious_investigation': {
        'name': 'Mystery Detective',
        'url': 'https://cdn.pixabay.com/download/audio/2023/06/14/audio_a3f9c0e20e.mp3?filename=night-detective-226857.mp3',
        'backup_url': 'https://incompetech.com/music/royalty-free/mp3-royaltyfree/Cipher.mp3',
        'duration': 134,
        'emotion': 'investigative, moody, mysterious',
        'scenes': ['investigation', 'true_crime'],
        'volume_default': 0.10
    },

    'noir_detective': {
        'name': 'Film Noir Mystery',
        'url': 'https://cdn.pixabay.com/download/audio/2022/03/10/audio_4ac3e30491.mp3?filename=mysterious-dark-122826.mp3',
        'backup_url': 'https://incompetech.com/music/royalty-free/mp3-royaltyfree/Wounded.mp3',
        'duration': 120,
        'emotion': 'noir, detective, shadowy',
        'scenes': ['investigation', 'unsolved'],
        'volume_default': 0.06
    },

    # 👻 SUSPENSEFUL TENSION (Building Suspense)
    'suspense_build': {
        'name': 'Dark Tension Builder',
        'artist': 'Muzaproduction',
        'source': 'Pixabay',
        'tags': ['suspense', 'tension', 'horror', 'dark', 'creeping'],
        # ✅ NEW, WORKING LINKS
        'url': 'https://incompetech.com/music/royalty-free/mp3-royaltyfree/Evening%20of%20Chaos.mp3', # Primary
        'backup_url': 'https://cdn.pixabay.com/download/audio/2023/02/13/audio_7717072455.mp3', # Backup
        'duration': 135,
        'emotion': 'tense, building, suspenseful',
        'scenes': ['suspense', 'paranormal'],
        'volume_default': 0.07
    },

    'creeping_tension': {
        'name': 'Creeping Horror',
        'url': 'https://cdn.pixabay.com/download/audio/2022/05/11/audio_c2f0a30d14.mp3?filename=creepy-horror-background-122826.mp3',
        'backup_url': 'https://incompetech.com/music/royalty-free/mp3-royaltyfree/Darkness%20is%20Coming.mp3',
        'duration': 141,
        'emotion': 'creepy, tense, ominous',
        'scenes': ['suspense', 'cryptids'],
        'volume_default': 0.05
    },

    'anxious_suspense': {
        'name': 'Anxious Tension',
        'url': 'https://cdn.pixabay.com/download/audio/2023/11/07/audio_d8c7ea6e48.mp3?filename=horror-ambience-183248.mp3',
        'backup_url': 'https://incompetech.com/music/royalty-free/mp3-royaltyfree/Unnerving.mp3',
        'duration': 118,
        'emotion': 'anxious, nervous, unsettling',
        'scenes': ['suspense', 'paranormal'],
        'volume_default': 0.04
    },

    # 🌫️ EERIE ATMOSPHERIC (Paranormal/Supernatural)
    'paranormal_ambient': {
        'name': 'Paranormal Atmosphere',
        'url': 'https://cdn.pixabay.com/download/audio/2022/08/02/audio_0a0c0b81ec.mp3?filename=mystery-noir-ambient-124013.mp3',
        'backup_url': 'https://incompetech.com/music/royalty-free/mp3-royaltyfree/Invariance.mp3',
        'duration': 124,
        'emotion': 'eerie, otherworldly, haunting',
        'scenes': ['paranormal', 'supernatural'],
        'volume_default': 0.06
    },

    'ghostly_whispers': {
        'name': 'Ghostly Ambience',
        'url': 'https://cdn.pixabay.com/download/audio/2023/02/13/audio_78a8675158.mp3?filename=dark-mystery-trailer-149604.mp3',
        'backup_url': 'https://incompetech.com/music/royalty-free/mp3-royaltyfree/The%20House%20of%20Leaves.mp3',
        'duration': 149,
        'emotion': 'ghostly, ethereal, spooky',
        'scenes': ['paranormal', 'unsolved'],
        'volume_default': 0.05
    },

    'supernatural_mystery': {
        'name': 'Supernatural Theme',
        'url': 'https://cdn.pixabay.com/download/audio/2022/10/25/audio_7e8c03e5c3.mp3?filename=horror-piano-dramatic-109943.mp3',
        'backup_url': 'https://incompetech.com/music/royalty-free/mp3-royaltyfree/Volatile%20Reaction.mp3',
        'duration': 109,
        'emotion': 'supernatural, mystical, dark',
        'scenes': ['paranormal', 'ancient'],
        'volume_default': 0.07
    },

    # ⚡ DRAMATIC REVELATION (Plot Twists/Reveals)
    'dramatic_reveal': {
        'name': 'Dramatic Discovery',
        'url': 'https://cdn.pixabay.com/download/audio/2022/05/27/audio_1808fbf07a.mp3?filename=dramatic-inspirational-142005.mp3',
        'backup_url': 'https://incompetech.com/music/royalty-free/mp3-royaltyfree/Arcane.mp3',
        'duration': 142,
        'emotion': 'dramatic, revealing, intense',
        'scenes': ['revelation', 'true_crime'],
        'volume_default': 0.10
    },

    'shocking_truth': {
        'name': 'Shocking Discovery',
        'url': 'https://cdn.pixabay.com/download/audio/2022/09/06/audio_0625c1539c.mp3?filename=epic-cinematic-inspiration-141080.mp3',
        'backup_url': 'https://incompetech.com/music/royalty-free/mp3-royaltyfree/Prelude%20and%20Action.mp3',
        'duration': 140,
        'emotion': 'shocking, powerful, dramatic',
        'scenes': ['revelation', 'conspiracy'],
        'volume_default': 0.10
    },

    'plot_twist': {
        'name': 'Plot Twist Theme',
        'url': 'https://cdn.pixabay.com/download/audio/2023/10/25/audio_8b0c7a1c6a.mp3?filename=inspirational-epic-orchestral-186819.mp3',
        'backup_url': 'https://incompetech.com/music/royalty-free/mp3-royaltyfree/Heart%20of%20the%20Beast.mp3',
        'duration': 88,
        'emotion': 'surprising, dramatic, intense',
        'scenes': ['revelation', 'unsolved'],
        'volume_default': 0.10
    },

    # 🏛️ ANCIENT MYSTERY (Historical/Archaeological)
    'ancient_secrets': {
        'name': 'Ancient Mystery',
        'url': 'https://cdn.pixabay.com/download/audio/2022/08/23/audio_2f4b0cfcd1.mp3?filename=cinematic-inspiration-149922.mp3',
        'backup_url': 'https://incompetech.com/music/royalty-free/mp3-royaltyfree/Shadowlands%201%20-%20Horizon.mp3',
        'duration': 95,
        'emotion': 'ancient, mysterious, epic',
        'scenes': ['ancient', 'archaeology'],
        'volume_default': 0.10
    },

    'lost_civilization': {
        'name': 'Lost Worlds',
        'url': 'https://cdn.pixabay.com/download/audio/2022/11/28/audio_eaa2c13f5c.mp3?filename=action-cinematic-134127.mp3',
        'backup_url': 'https://incompetech.com/music/royalty-free/mp3-royaltyfree/Nowhere%20Land.mp3',
        'duration': 134,
        'emotion': 'epic, ancient, mysterious',
        'scenes': ['ancient', 'conspiracy'],
        'volume_default': 0.10
    },

    'archaeological_discovery': {
        'name': 'Ancient Discovery',
        'url': 'https://cdn.pixabay.com/download/audio/2022/10/07/audio_c7b5c2e3d8.mp3?filename=epic-orchestral-116227.mp3',
        'backup_url': 'https://incompetech.com/music/royalty-free/mp3-royaltyfree/Danse%20Macabre%20-%20Big%20Hit.mp3',
        'duration': 116,
        'emotion': 'epic, archaeological, grand',
        'scenes': ['ancient', 'revelation'],
        'volume_default': 0.10
    },

    # 🔥 THRILLER ACTION (Chase/Danger)
    'chase_sequence': {
        'name': 'Thriller Chase',
        'url': 'https://cdn.pixabay.com/download/audio/2023/02/28/audio_69d61ea5e8.mp3?filename=intense-epic-cinematic-trailer-171048.mp3',
        'backup_url': 'https://incompetech.com/music/royalty-free/mp3-royaltyfree/The%20Complex.mp3',
        'duration': 125,
        'emotion': 'intense, urgent, dangerous',
        'scenes': ['action', 'true_crime'],
        'volume_default': 0.10
    },

    'dangerous_encounter': {
        'name': 'Danger Theme',
        'url': 'https://cdn.pixabay.com/download/audio/2022/03/15/audio_bbc8d4c9c7.mp3?filename=inspiring-cinematic-ambient-116199.mp3',
        'backup_url': 'https://incompetech.com/music/royalty-free/mp3-royaltyfree/Ishikari%20Lore.mp3',
        'duration': 158,
        'emotion': 'dangerous, thrilling, intense',
        'scenes': ['action', 'cryptids'],
        'volume_default': 0.10
    },

    'urgent_investigation': {
        'name': 'Urgent Mystery',
        'url': 'https://cdn.pixabay.com/download/audio/2022/05/13/audio_c6f5d1cd4c.mp3?filename=cinematic-time-lapse-115672.mp3',
        'backup_url': 'https://incompetech.com/music/royalty-free/mp3-royaltyfree/Deadly%20Roulette.mp3',
        'duration': 131,
        'emotion': 'urgent, fast-paced, tense',
        'scenes': ['action', 'investigation'],
        'volume_default': 0.10
    },

    # 🕯️ CONSPIRACY MOOD (Cover-ups/Secrets)
    'conspiracy_theme': {
        'name': 'Conspiracy Atmosphere',
        'url': 'https://cdn.pixabay.com/download/audio/2022/03/23/audio_c8c3c1c3ed.mp3?filename=powerful-beat-121791.mp3',
        'backup_url': 'https://incompetech.com/music/royalty-free/mp3-royaltyfree/Stages%20of%20Grief.mp3',
        'duration': 120,
        'emotion': 'secretive, dark, conspiratorial',
        'scenes': ['conspiracy', 'cover_up'],
        'volume_default': 0.10
    },

    'hidden_truth': {
        'name': 'Hidden Secrets',
        'url': 'https://cdn.pixabay.com/download/audio/2022/03/18/audio_ca3c8d3ab3.mp3?filename=epic-drums-216819.mp3',
        'backup_url': 'https://incompetech.com/music/royalty-free/mp3-royaltyfree/mining%20by%20moonlight.mp3',
        'duration': 99,
        'emotion': 'mysterious, hidden, dark',
        'scenes': ['conspiracy', 'unsolved'],
        'volume_default': 0.10
    },

    # 🌲 CRYPTID ENCOUNTERS (Unknown Creatures)
    'creature_encounter': {
        'name': 'Unknown Entity',
        'url': 'https://cdn.pixabay.com/download/audio/2023/11/28/audio_ef8b3e8c6d.mp3?filename=dark-forest-ambient-184304.mp3',
        'backup_url': 'https://incompetech.com/music/royalty-free/mp3-royaltyfree/Cryptic%20Sorrow.mp3',
        'duration': 142,
        'emotion': 'ominous, creature, wilderness',
        'scenes': ['cryptids', 'forest'],
        'volume_default': 0.10
    },

    'forest_mystery': {
        'name': 'Deep Woods Mystery',
        'url': 'https://cdn.pixabay.com/download/audio/2023/07/25/audio_c4f3847f24.mp3?filename=dark-ambient-mystery-231397.mp3',
        'backup_url': 'https://incompetech.com/music/royalty-free/mp3-royaltyfree/Stormfront.mp3',
        'duration': 127,
        'emotion': 'forest, mysterious, wild',
        'scenes': ['cryptids', 'unexplained'],
        'volume_default': 0.10
    }
}


def get_music_cache_path():
    """Get path to music cache file"""
    return os.path.join(MUSIC_DIR, "music_cache.json")


def load_music_cache():
    """Load cached music metadata"""
    cache_path = get_music_cache_path()
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'r') as f:
                return json.load(f)
        except:
            return {}
    return {}


def save_music_cache(cache):
    """Save music cache metadata"""
    cache_path = get_music_cache_path()
    with open(cache_path, 'w') as f:
        json.dump(cache, f, indent=2)


def get_track_hash(track_key):
    """Get hash for track identification"""
    return hashlib.md5(track_key.encode()).hexdigest()[:12]


def download_track(track_key, track_info, force=False):
    """
    Download a music track, trying multiple URLs with a robust User-Agent.
    Returns: local file path or None
    """
    track_hash = get_track_hash(track_key)
    filename = f"{track_key}_{track_hash}.mp3"
    filepath = os.path.join(MUSIC_DIR, filename)

    cache = load_music_cache()
    if not force and track_key in cache:
        cached_path = cache[track_key].get('local_path')
        if cached_path and os.path.exists(cached_path) and os.path.getsize(cached_path) > 10000:
            print(f"✅ Using cached: {track_info['name']}")
            return cached_path

    print(f"📥 Downloading: {track_info['name']}...")
    
    # ✅ FLEXIBLE URL GATHERING
    urls_to_try = [
        ('Primary', track_info.get('url')),
        ('Backup', track_info.get('backup_url')),
        ('Tertiary', track_info.get('tertiary_url')) # Will be ignored if not present
    ]
    
    # ✅ IMPROVED BROWSER-LIKE HEADER
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36'
    }

    for source_name, url in urls_to_try:
        # Skip if the URL key (e.g., 'tertiary_url') doesn't exist for a track
        if not url:
            continue
            
        try:
            print(f"   🔄 Trying {source_name} source...")
            response = requests.get(url, headers=headers, timeout=120, stream=True, allow_redirects=True)
            response.raise_for_status() # Raise an exception for HTTP error codes

            # Basic content type check
            content_type = response.headers.get('Content-Type', '').lower()
            if 'audio' not in content_type and 'octet-stream' not in content_type:
                print(f"      ⚠️ Unexpected content type: {content_type}, skipping.")
                continue

            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            if os.path.exists(filepath) and os.path.getsize(filepath) > 50000:
                print(f"      ✅ Downloaded {os.path.getsize(filepath) / 1024:.1f} KB from {source_name}")
                                # Update cache
                # Start with a copy of all info from the main library
                cache_entry = track_info.copy()
                # Update with download-specific details
                cache_entry.update({
                    'local_path': filepath,
                    'url_used': url,
                    'source_used': source_name,
                    'downloaded_at': datetime.now().isoformat(),
                    'file_size_kb': round(os.path.getsize(filepath) / 1024, 2)
                })
                cache[track_key] = cache_entry
                save_music_cache(cache)
                return filepath
            else:
                if os.path.exists(filepath): os.remove(filepath)
                print(f"      ⚠️ Download was empty or too small.")

        except requests.exceptions.RequestException as e:
            print(f"      ⚠️ Network error for {source_name}: {e}")

    print(f"   ❌ All sources failed for {track_key}")
    return None


def get_music_for_scene(scene_type, content_type='general'):
    """
    Get best music track for a scene type
    Returns: (track_key, local_path, volume)
    """
    
    # Priority order for scene types
    scene_priority = {
        'investigation': ['dark_mystery', 'mysterious_investigation', 'noir_detective'],
        'paranormal': ['paranormal_ambient', 'ghostly_whispers', 'supernatural_mystery'],
        'suspense': ['suspense_build', 'creeping_tension', 'anxious_suspense'],
        'revelation': ['dramatic_reveal', 'shocking_truth', 'plot_twist'],
        'ancient': ['ancient_secrets', 'lost_civilization', 'archaeological_discovery'],
        'true_crime': ['mysterious_investigation', 'chase_sequence', 'dramatic_reveal'],
        'conspiracy': ['conspiracy_theme', 'hidden_truth', 'dark_mystery'],
        'cryptids': ['creature_encounter', 'forest_mystery', 'creeping_tension'],
        'unsolved': ['dark_mystery', 'mysterious_investigation', 'ghostly_whispers'],
        'general': ['dark_mystery', 'suspense_build', 'dramatic_reveal']
    }
    
    # Content type overrides
    content_overrides = {
        'intro': ['dark_mystery', 'mysterious_investigation', 'suspense_build'],
        'outro': ['dramatic_reveal', 'plot_twist', 'shocking_truth'],
        'background': ['paranormal_ambient', 'dark_mystery', 'ancient_secrets'],
        'intense': ['chase_sequence', 'dangerous_encounter', 'urgent_investigation']
    }
    
    # Get priority list
    if content_type in content_overrides:
        priority_tracks = content_overrides[content_type]
    else:
        priority_tracks = scene_priority.get(scene_type, scene_priority['general'])
    
    # Try to download tracks in priority order
    for track_key in priority_tracks:
        if track_key in MUSIC_LIBRARY:
            track_info = MUSIC_LIBRARY[track_key]
            local_path = download_track(track_key, track_info)
            
            if local_path:
                return track_key, local_path, track_info['volume_default']
    
    # Fallback: try any available track
    print(f"⚠️ Priority tracks unavailable, trying fallbacks...")
    for track_key, track_info in MUSIC_LIBRARY.items():
        local_path = download_track(track_key, track_info)
        if local_path:
            return track_key, local_path, track_info['volume_default']
    
    return None, None, 0.18


def download_all_music():
    """Download all music tracks with source tracking"""
    
    print("\n" + "="*70)
    print("🔮 DOWNLOADING MYSTERY MUSIC LIBRARY - MYTHICA REPORT")
    print("="*70)
    print("Primary: Pixabay (CC0) | Backup: Incompetech (CC BY 4.0)")
    print()
    
    total = len(MUSIC_LIBRARY)
    successful = 0
    failed = []
    source_stats = {'Primary (Pixabay)': 0, 'Backup (Incompetech)': 0}
    
    for track_key, track_info in MUSIC_LIBRARY.items():
        print(f"\n📀 [{successful + len(failed) + 1}/{total}] {track_info['name']}")
        
        local_path = download_track(track_key, track_info)
        
        if local_path:
            successful += 1
            print(f"   ✅ Ready: {local_path}")
            
            # Track which source was used
            cache = load_music_cache()
            if track_key in cache:
                source = cache[track_key].get('source', 'Unknown')
                source_stats[source] = source_stats.get(source, 0) + 1
        else:
            failed.append(track_key)
            print(f"   ❌ Failed")
    
    print("\n" + "="*70)
    print("📊 DOWNLOAD SUMMARY")
    print("="*70)
    print(f"✅ Successful: {successful}/{total}")
    print(f"❌ Failed: {len(failed)}/{total}")
    
    if source_stats:
        print(f"\n📍 Sources Used:")
        for source, count in source_stats.items():
            percentage = (count / successful * 100) if successful > 0 else 0
            print(f"   {source}: {count} tracks ({percentage:.1f}%)")
    
    if failed:
        print(f"\n❌ Failed tracks: {', '.join(failed)}")
    
    print(f"\n💾 Music library location: {MUSIC_DIR}")
    print(f"🎵 Reliability: Both Pixabay and Incompetech sources available")
    
    return successful, failed


def test_music_urls():
    """Quick test to verify music URLs are accessible"""
    
    print("\n🧪 Testing mystery music URL accessibility...")
    
    working = []
    broken = []
    
    # Test first 5 tracks
    test_tracks = list(MUSIC_LIBRARY.items())[:5]
    
    for track_key, track_info in test_tracks:
        print(f"\n🔍 Testing {track_info['name']}...")
        
        for source, url_key in [('Primary', 'url'), ('Backup', 'backup_url')]:
            if url_key not in track_info:
                continue
                
            url = track_info[url_key]
            try:
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                    'Referer': 'https://pixabay.com/' if 'pixabay' in url else 'https://incompetech.com/'
                }
                
                response = requests.head(url, timeout=10, allow_redirects=True, headers=headers)
                if response.status_code == 200:
                    print(f"   ✅ {source} URL working")
                    working.append(f"{track_key} ({source})")
                else:
                    print(f"   ⚠️ {source} returned {response.status_code}")
                    broken.append(f"{track_key} ({source})")
            except Exception as e:
                print(f"   ❌ {source} failed: {str(e)[:50]}")
                broken.append(f"{track_key} ({source})")
    
    print(f"\n📊 Quick Test Results:")
    print(f"   Working: {len(working)}")
    print(f"   Broken: {len(broken)}")
    
    if len(working) > 0:
        print(f"\n✅ URLs are functional! Safe to download library.")
    else:
        print(f"\n⚠️ URL issues detected. Check network or URLs.")
    
    return len(working), len(broken)


def cleanup_old_music(keep_days=30):
    """Clean up music files older than specified days"""
    
    print(f"\n🧹 Cleaning up music older than {keep_days} days...")
    
    cache = load_music_cache()
    removed = 0
    
    from datetime import timedelta
    cutoff = datetime.now() - timedelta(days=keep_days)
    
    for track_key, track_data in list(cache.items()):
        downloaded_at = track_data.get('downloaded_at')
        
        if downloaded_at:
            try:
                download_date = datetime.fromisoformat(downloaded_at.replace('Z', '+00:00'))
                
                if download_date < cutoff:
                    local_path = track_data.get('local_path')
                    if local_path and os.path.exists(local_path):
                        os.remove(local_path)
                        print(f"   🗑️ Removed: {track_key}")
                        removed += 1
                    
                    del cache[track_key]
            except:
                pass
    
    if removed > 0:
        save_music_cache(cache)
        print(f"✅ Removed {removed} old tracks")
    else:
        print(f"✅ No old tracks to remove")


def print_music_library():
    """Print available music library"""
    
    print("\n" + "="*70)
    print("🔮 MYTHICA REPORT MUSIC LIBRARY")
    print("="*70)
    
    by_category = {}
    for track_key, track_info in MUSIC_LIBRARY.items():
        scenes = ', '.join(track_info['scenes'])
        if scenes not in by_category:
            by_category[scenes] = []
        by_category[scenes].append(track_info)
    
    for scenes, tracks in sorted(by_category.items()):
        print(f"\n📂 {scenes.upper().replace('_', ' ')}")
        for track in tracks:
            print(f"   🎵 {track['name']}")
            print(f"      Emotion: {track['emotion']}")
            print(f"      Duration: {track['duration']}s")
            print(f"      Default volume: {track['volume_default']*100:.0f}%")


def validate_all_urls():
    """Comprehensive URL validation for all tracks"""
    
    print("\n" + "="*70)
    print("🔍 VALIDATING ALL MUSIC URLS")
    print("="*70)
    
    results = {
        'working': [],
        'broken': [],
        'total': len(MUSIC_LIBRARY) * 2  # Primary + backup
    }
    
    for track_key, track_info in MUSIC_LIBRARY.items():
        print(f"\n🎵 {track_info['name']}")
        
        for source, url_key in [('Primary (Pixabay)', 'url'), ('Backup (Incompetech)', 'backup_url')]:
            if url_key not in track_info:
                continue
            
            url = track_info[url_key]
            try:
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                    'Referer': 'https://pixabay.com/' if 'pixabay' in url else 'https://incompetech.com/'
                }
                
                response = requests.head(url, timeout=15, allow_redirects=True, headers=headers)
                
                if response.status_code == 200:
                    print(f"   ✅ {source}: OK")
                    results['working'].append(f"{track_key} ({source})")
                else:
                    print(f"   ⚠️ {source}: HTTP {response.status_code}")
                    results['broken'].append(f"{track_key} ({source})")
                    
            except Exception as e:
                print(f"   ❌ {source}: {str(e)[:40]}")
                results['broken'].append(f"{track_key} ({source})")
    
    print("\n" + "="*70)
    print("📊 VALIDATION RESULTS")
    print("="*70)
    print(f"✅ Working URLs: {len(results['working'])}/{results['total']}")
    print(f"❌ Broken URLs: {len(results['broken'])}/{results['total']}")
    print(f"📈 Success Rate: {len(results['working'])/results['total']*100:.1f}%")
    
    if results['broken']:
        print(f"\n⚠️ Issues found in:")
        for item in results['broken']:
            print(f"   - {item}")
    
    return results


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--download-all':
        download_all_music()
        
    elif len(sys.argv) > 1 and sys.argv[1] == '--cleanup':
        cleanup_old_music()
        
    elif len(sys.argv) > 1 and sys.argv[1] == '--list':
        print_music_library()
        
    elif len(sys.argv) > 1 and sys.argv[1] == '--test':
        test_music_urls()
        
    elif len(sys.argv) > 1 and sys.argv[1] == '--validate':
        validate_all_urls()
        
    else:
        print("🔮 Mythica Report - Mystery Music Download System")
        print("\nUsage:")
        print("  python download_music.py --download-all  # Download entire library")
        print("  python download_music.py --test          # Quick URL test (5 tracks)")
        print("  python download_music.py --validate      # Validate ALL URLs")
        print("  python download_music.py --cleanup       # Remove old tracks")
        print("  python download_music.py --list          # List library")
        print("\n💡 Uses Pixabay (primary) + Incompetech (backup) for reliability")
        print("\nDownloading essential mystery tracks...")
        
        # Download one track from each category
        essential = [
            'dark_mystery',           # Investigation
            'suspense_build',         # Suspense
            'paranormal_ambient',     # Paranormal
            'dramatic_reveal',        # Revelation
            'ancient_secrets',        # Ancient
            'creature_encounter'      # Cryptids
        ]
        
        for track_key in essential:
            if track_key in MUSIC_LIBRARY:
                download_track(track_key, MUSIC_LIBRARY[track_key])
        
        print("\n✅ Essential mystery tracks ready for Mythica Report! 🔮")