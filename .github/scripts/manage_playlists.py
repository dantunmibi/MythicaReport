#!/usr/bin/env python3
"""
🔮 Manage Mystery Playlists - MYTHICA REPORT v6.0
Organizes videos into dark mystery content pillars:
- Dark History: Forgotten Tragedies (20%)
- Medical Mysteries: Conditions That Baffle Doctors (15%)
- Dark Experiments: Secret Research Exposed (10%)
- Vanished: Unsolved Disappearances (28%)
- True Crime Files (14%)
- Unexplained Phenomena (13%)
"""

import os
import json
from datetime import datetime
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from google.oauth2.credentials import Credentials
from collections import defaultdict
import re
import difflib
import time

TMP = os.getenv("GITHUB_WORKSPACE", ".") + "/tmp"
PLAYLIST_CONFIG_FILE = os.path.join(TMP, "playlist_config.json")
UPLOAD_LOG = os.path.join(TMP, "upload_history.json")


def get_youtube_client():
    """Authenticate YouTube API"""
    try:
        creds = Credentials(
            None,
            refresh_token=os.getenv("GOOGLE_REFRESH_TOKEN"),
            token_uri="https://oauth2.googleapis.com/token",
            client_id=os.getenv("GOOGLE_CLIENT_ID"),
            client_secret=os.getenv("GOOGLE_CLIENT_SECRET"),
            scopes=["https://www.googleapis.com/auth/youtube"]
        )
        youtube = build("youtube", "v3", credentials=creds, cache_discovery=False)
        print("✅ YouTube API authenticated")
        return youtube
    except Exception as e:
        print(f"❌ Authentication failed: {e}")
        raise


# 🔮 MYSTERY PLAYLIST CONFIGURATION v6.0
PLAYLIST_RULES = {
    "mystery": {
        # 🆕 NEW CATEGORY: Dark History (Mondays)
        "dark_history": {
            "title": "🕰️ Dark History: Forgotten Tragedies",
            "description": "Real historical events with mysterious and disturbing elements. The Radium Girls. Centralia's eternal fire. Events history tried to bury. Dark facts told with mysterious intrigue, not educational lectures.",
            "keywords": [
                "dark history", "forgotten", "tragedy", "historical", "history", "buried",
                "covered up", "hidden", "suppressed", "lost", "erased", "destroyed",
                "radium girls", "glowed", "glowing", "toxic", "poisoned", "radiation",
                "centralia", "coal fire", "burning", "ghost town", "abandoned",
                "disaster", "catastrophe", "event", "incident", "happened", "occurred",
                "victims", "died", "killed", "death toll", "casualties", "perished",
                "mysterious death", "unexplained deaths", "mass death", "epidemic",
                "1900s", "1800s", "century", "decades ago", "years ago", "past",
                "documentary", "investigation", "revealed", "discovered", "uncovered",
                "truth", "facts", "evidence", "records", "files", "documents",
                "government", "company", "corporation", "officials", "authorities",
                "scandal", "corruption", "negligence", "criminal", "liability",
                "workers", "factory", "mine", "industry", "labor", "exploitation",
                "medical", "experiment", "treatment", "procedure", "disease", "illness"
            ]
        },
        
        # 🆕 NEW CATEGORY: Disturbing Medical (Wednesdays)
        "disturbing_medical": {
            "title": "🧬 Medical Mysteries: Conditions That Baffle Doctors",
            "description": "Real medical conditions that defy explanation. Fatal insomnia. Turning to stone. Mysterious illnesses that terrify even doctors. Body horror meets medical mystery.",
            "keywords": [
                "medical mystery", "condition", "disease", "illness", "syndrome",
                "doctors baffled", "can't explain", "no cure", "no treatment", "no survivors",
                "fatal", "deadly", "terminal", "progressive", "degenerative",
                "fatal insomnia", "couldn't sleep", "no sleep", "insomnia", "awake",
                "stone man", "turned to stone", "ossification", "bone", "calcification",
                "fibrodysplasia", "fop", "rare disease", "genetic", "mutation",
                "kuru", "laughing death", "prion", "brain disease", "neurodegenerative",
                "mysterious symptoms", "baffling", "unexplained", "impossible",
                "medical anomaly", "medical enigma", "case study", "patient zero",
                "epidemic", "outbreak", "spread", "contagious", "transmission",
                "minamata", "mercury", "poisoning", "contamination", "toxic",
                "blindness", "paralysis", "seizures", "convulsions", "spasms",
                "pain", "suffering", "agony", "torture", "unbearable",
                "diagnosis", "misdiagnosed", "undiagnosed", "unknown cause",
                "doctors", "specialists", "experts", "medical professionals",
                "hospital", "clinic", "medical facility", "treatment center",
                "research", "study", "investigation", "autopsy", "examination"
            ]
        },
        
        # 🆕 NEW CATEGORY: Dark Experiments (Thursdays)
        "dark_experiments": {
            "title": "🔬 Dark Experiments: Secret Research Exposed",
            "description": "Declassified experiments. MK-Ultra's lost subjects. Stanford Prison gone wrong. Secret research that crossed ethical lines. Government and corporate experiments that should never have happened.",
            "keywords": [
                "dark experiment", "secret research", "unethical", "classified", "declassified",
                "mk ultra", "mkultra", "cia", "mind control", "brainwashing", "lsd",
                "stanford prison", "prison experiment", "psychological", "abuse",
                "sleep deprivation", "couldn't sleep", "stayed awake", "experiment",
                "subjects", "participants", "volunteers", "prisoners", "patients",
                "testing", "tested on", "experimented on", "guinea pigs", "human trials",
                "government", "military", "cia", "fbi", "agency", "department",
                "project", "operation", "program", "initiative", "study",
                "secret", "classified", "top secret", "redacted", "hidden",
                "files", "documents", "records", "leaked", "released", "exposed",
                "whistleblower", "insider", "revealed", "uncovered", "discovered",
                "unethical", "illegal", "immoral", "wrong", "violation",
                "human rights", "ethics", "consent", "informed", "forced",
                "torture", "interrogation", "techniques", "methods", "procedures",
                "psychological", "mental", "trauma", "ptsd", "damage",
                "cover up", "covered up", "suppressed", "buried", "destroyed",
                "victims", "survivors", "testimony", "accounts", "reports",
                "shut down", "terminated", "ended", "stopped", "discontinued"
            ]
        },
        
        # EXISTING CATEGORY: Disappearances (Tuesdays & Fridays - your strength)
        "disappearance": {
            "title": "👤 Vanished: Unsolved Disappearances",
            "description": "People who vanished without a trace. Flight 19. DB Cooper. Maura Murray. Cases that remain unsolved. The search continues.",
            "keywords": [
                "vanished", "disappeared", "missing", "gone", "never found", "no trace",
                "unsolved", "mystery", "cold case", "still missing", "search",
                "last seen", "sighting", "witness", "evidence", "clues",
                "flight 19", "planes", "aircraft", "aviation", "triangle",
                "db cooper", "hijacker", "parachute", "jumped", "skyjacker",
                "maura murray", "car", "crash", "abandoned", "woods",
                "brandon swanson", "phone call", "field", "rural", "farm",
                "asha degree", "walked out", "highway", "storm", "night",
                "elisa lam", "hotel", "water tank", "elevator", "footage",
                "jonbenet ramsey", "pageant", "ransom", "note", "basement",
                "zodiac", "killer", "cipher", "code", "letters", "taunting",
                "investigation", "detective", "police", "fbi", "authorities",
                "family", "loved ones", "parents", "waiting", "hoping",
                "anniversary", "years later", "decades", "still looking",
                "reward", "tip", "lead", "breakthrough", "development"
            ]
        },
        
        # EXISTING CATEGORY: True Crime (Saturdays)
        "true_crime": {
            "title": "🔪 True Crime Files: Dark Reality",
            "description": "Real crime stories, criminal minds, and the investigations that brought them to justice. Warning: Dark content.",
            "keywords": [
                "true crime", "murder", "killer", "serial killer", "crime", "criminal",
                "investigation", "detective", "fbi", "police", "arrest", "convicted",
                "trial", "court", "guilty", "innocent", "justice", "victim", "suspect",
                "evidence", "forensic", "autopsy", "crime scene", "homicide", "death",
                "zodiac killer", "black dahlia", "somerton man", "unidentified",
                "cold case", "unsolved murder", "mystery", "baffling", "strange",
                "motive", "alibi", "witness", "testimony", "confession", "caught",
                "solved", "case closed", "breakthrough", "arrest", "manhunt",
                "wanted", "escaped", "fugitive", "prison", "jail", "sentenced"
            ]
        },
        
        # EXISTING CATEGORY: Paranormal (Sundays)
        "phenomena": {
            "title": "👁️ Unexplained Phenomena: Beyond Science",
            "description": "Strange occurrences, mysterious signals, impossible events. The Wow Signal. The Hum. Phenomena that defy scientific explanation.",
            "keywords": [
                "phenomena", "unexplained", "mysterious", "strange", "impossible",
                "paranormal", "supernatural", "unexplainable", "baffling",
                "wow signal", "signal", "radio", "space", "cosmos", "seti",
                "the hum", "sound", "noise", "frequency", "hear", "hearing",
                "hessdalen lights", "lights", "orbs", "ufo", "unidentified",
                "ball lightning", "lightning", "electrical", "phenomenon",
                "science", "scientists", "researchers", "experts", "baffled",
                "no explanation", "can't explain", "defies", "contradicts",
                "witnesses", "reported", "sightings", "observations", "documented",
                "investigation", "study", "research", "analysis", "examination"
            ]
        },
        
        # DEPRECATED: Merged into dark_history
        "ancient": {
            "title": "🗿 Ancient Mysteries: Lost Civilizations [DEPRECATED - Use Dark History]",
            "description": "Merged into Dark History category for better organization.",
            "keywords": ["deprecated"]
        },
        
        # DEPRECATED: Merged into dark_experiments
        "conspiracy": {
            "title": "🕵️ Conspiracy Theories [DEPRECATED - Use Dark Experiments]",
            "description": "Merged into Dark Experiments category for better framing.",
            "keywords": ["deprecated"]
        },
        
        # REMOVED: Low retention category
        "cryptids": {
            "title": "🦎 Cryptids & Creatures [REMOVED]",
            "description": "Removed due to low retention performance.",
            "keywords": ["removed"]
        }
    }
}

# 🆕 v6.0: Category aliases for backward compatibility
CATEGORY_ALIASES = {
    "historical": "dark_history",
    "historical_mystery": "dark_history",
    "ancient": "dark_history",
    "medical": "disturbing_medical",
    "medical_mystery": "disturbing_medical",
    "experiments": "dark_experiments",
    "dark_science": "dark_experiments",
    "conspiracy": "dark_experiments",
    "unsolved": "disappearance",
    "disappearance_mystery": "disappearance",
    "crime": "true_crime",
    "crime_mystery": "true_crime",
    "true_crime": "true_crime",
    "paranormal": "phenomena",
    "phenomena_mystery": "phenomena",
}


def fetch_and_map_existing_playlists(youtube, niche, config):
    """Fetch existing playlists and map to categories"""
    print("🔄 Fetching existing playlists...")
    existing_playlists = {}
    nextPageToken = None
    
    while True:
        response = youtube.playlists().list(
            part="snippet",
            mine=True,
            maxResults=50,
            pageToken=nextPageToken
        ).execute()
        
        for item in response.get("items", []):
            existing_playlists[item["snippet"]["title"].lower()] = item["id"]
        
        nextPageToken = response.get("nextPageToken")
        if not nextPageToken:
            break
    
    # Map to categories using fuzzy matching
    for category, rules in PLAYLIST_RULES[niche].items():
        # Skip deprecated/removed categories
        if rules.get("keywords") == ["deprecated"] or rules.get("keywords") == ["removed"]:
            continue
            
        key = f"{niche}_{category}"
        match = None
        
        for title, pid in existing_playlists.items():
            ratio = difflib.SequenceMatcher(None, rules["title"].lower(), title).ratio()
            if ratio > 0.6:
                match = pid
                break
        
        if match:
            if key in config and config[key] != match:
                print(f"♻️ Updated playlist ID for '{rules['title']}'")
            else:
                print(f"✅ Mapped '{rules['title']}'")
            config[key] = match
    
    return config


def load_upload_history():
    """Load video upload history"""
    if os.path.exists(UPLOAD_LOG):
        try:
            with open(UPLOAD_LOG, 'r') as f:
                return json.load(f)
        except:
            return []
    return []


def load_playlist_config():
    """Load playlist configuration"""
    if os.path.exists(PLAYLIST_CONFIG_FILE):
        try:
            with open(PLAYLIST_CONFIG_FILE, 'r') as f:
                return json.load(f)
        except:
            return {}
    return {}


def save_playlist_config(config):
    """Save playlist configuration"""
    with open(PLAYLIST_CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"💾 Saved playlist config: {len(config)} playlists")


def get_or_create_playlist(youtube, niche, category, config):
    """Get existing playlist or create new one"""
    playlist_key = f"{niche}_{category}"
    
    if playlist_key in config:
        print(f"✅ Using existing playlist: {playlist_key}")
        return config[playlist_key]
    
    # Create new playlist
    try:
        playlist_info = PLAYLIST_RULES[niche][category]
        
        # Skip deprecated/removed categories
        if playlist_info.get("keywords") == ["deprecated"] or playlist_info.get("keywords") == ["removed"]:
            print(f"⚠️ Category '{category}' is deprecated/removed, skipping playlist creation")
            return None
        
        title = playlist_info["title"]
        description = playlist_info["description"]
        
        # Add branding
        full_description = f"{description}\n\n🔮 Mythica Report v6.0 - Dark mysteries told with film noir intrigue.\nNew dark content every week. #mystery #unexplained #mythica #darkhistory"
        
        request = youtube.playlists().insert(
            part="snippet,status",
            body={
                "snippet": {
                    "title": title,
                    "description": full_description,
                },
                "status": {"privacyStatus": "public"}
            }
        )
        response = request.execute()
        playlist_id = response["id"]
        
        config[playlist_key] = playlist_id
        save_playlist_config(config)
        print(f"🎉 Created playlist: {title}")
        return playlist_id
        
    except Exception as e:
        print(f"❌ Failed to create playlist: {e}")
        return None


def categorize_video(video_metadata, niche):
    """
    🆕 v6.0: Smart categorization with dark content support
    Checks script metadata first, then falls back to keyword matching
    """
    
    # 🆕 NEW: Check script metadata for category (most accurate)
    mystery_category = video_metadata.get("mystery_category")
    if mystery_category:
        # Normalize using aliases
        normalized = CATEGORY_ALIASES.get(mystery_category, mystery_category)
        
        # Verify it's a valid, active category
        if normalized in PLAYLIST_RULES.get(niche, {}):
            rules = PLAYLIST_RULES[niche][normalized]
            if rules.get("keywords") not in [["deprecated"], ["removed"]]:
                print(f"   📂 Using script category: {normalized} (from metadata)")
                return normalized
    
    # Fallback to keyword matching
    text = " ".join([
        video_metadata.get("title", ""),
        video_metadata.get("description", ""),
        video_metadata.get("hook", ""),
        video_metadata.get("key_phrase", ""),
        " ".join(video_metadata.get("hashtags", []))
    ]).lower()
    
    if niche not in PLAYLIST_RULES:
        return None
    
    scores = {}
    for category, rules in PLAYLIST_RULES[niche].items():
        # Skip deprecated/removed categories
        if rules.get("keywords") in [["deprecated"], ["removed"]]:
            continue
        
        score = 0
        
        # Exact keyword matches
        for kw in rules["keywords"]:
            kw_lower = kw.lower()
            if kw_lower in text:
                score += 5
            
            # Partial matches
            for word in kw_lower.split():
                if len(word) > 3 and word in text:
                    score += 2
                else:
                    # Fuzzy matching
                    for text_word in text.split():
                        if len(text_word) > 3:
                            ratio = difflib.SequenceMatcher(None, word, text_word).ratio()
                            if ratio > 0.85:
                                score += 1
        
        # 🆕 Bonus for category-specific power phrases
        power_phrases = {
            "dark_history": ["dark history", "radium girls", "glowed", "centralia", "tragedy", "forgotten", "buried"],
            "disturbing_medical": ["medical mystery", "doctors baffled", "can't explain", "fatal insomnia", "turned to stone", "condition"],
            "dark_experiments": ["mk ultra", "stanford prison", "experiment", "classified", "cia", "secret research", "declassified"],
            "disappearance": ["vanished", "disappeared", "missing", "never found", "flight 19", "db cooper"],
            "true_crime": ["true crime", "serial killer", "murder", "zodiac", "investigation"],
            "phenomena": ["unexplained", "wow signal", "the hum", "phenomenon", "scientists baffled"]
        }
        
        if category in power_phrases:
            for phrase in power_phrases[category]:
                if phrase in text:
                    score += 3
        
        if score > 0:
            scores[category] = score
    
    if scores:
        best = max(scores, key=scores.get)
        print(f"   📂 Categorized as: {best} (score: {scores[best]})")
        return best
    
    # Default to disappearance (your proven strength)
    print("   ⚠️ No match, defaulting to 'disappearance'")
    return "disappearance"


def add_video_to_playlist(youtube, video_id, playlist_id):
    """Add video to playlist with retry logic"""
    
    # Check if already in playlist
    existing_videos = set()
    max_retries = 3
    
    for attempt in range(max_retries):
        try:
            nextPageToken = None
            while True:
                response = youtube.playlistItems().list(
                    part="snippet",
                    playlistId=playlist_id,
                    maxResults=50,
                    pageToken=nextPageToken
                ).execute()
                
                for item in response.get("items", []):
                    existing_videos.add(item["snippet"]["resourceId"]["videoId"])
                
                nextPageToken = response.get("nextPageToken")
                if not nextPageToken:
                    break
            break
            
        except HttpError as e:
            if e.resp.status == 404 and attempt < max_retries - 1:
                wait = 2 ** attempt
                print(f"      ⏳ Playlist not ready, waiting {wait}s...")
                time.sleep(wait)
            else:
                print(f"      ⚠️ Could not check playlist: {e}")
                break
    
    if video_id in existing_videos:
        print("      ℹ️ Already in playlist")
        return False
    
    # Add video
    for attempt in range(max_retries):
        try:
            youtube.playlistItems().insert(
                part="snippet",
                body={
                    "snippet": {
                        "playlistId": playlist_id,
                        "resourceId": {"kind": "youtube#video", "videoId": video_id}
                    }
                }
            ).execute()
            print("      ✅ Added to playlist")
            return True
            
        except HttpError as e:
            if e.resp.status == 404 and attempt < max_retries - 1:
                wait = 2 ** attempt
                print(f"      ⏳ Retrying in {wait}s...")
                time.sleep(wait)
            else:
                print(f"      ❌ Failed: {e}")
                return False
    
    return False


def organize_playlists(youtube, history, config, niche):
    """Main organization function"""
    print(f"\n🎬 Organizing {len(history)} videos into playlists (v6.0 - Dark Content)...")
    
    stats = {
        "total_videos": len(history),
        "categorized": 0,
        "added_to_playlists": 0,
        "already_in_playlists": 0,
        "failed": 0,
        "by_category": defaultdict(int)
    }
    
    for video in history:
        video_id = video.get("video_id")
        title = video.get("title", "Unknown")
        
        if not video_id:
            continue
        
        print(f"\n📹 Processing: {title}")
        
        # Categorize
        category = categorize_video(video, niche)
        
        if not category:
            stats["failed"] += 1
            continue
        
        stats["categorized"] += 1
        stats["by_category"][category] += 1
        
        # Get/create playlist
        playlist_id = get_or_create_playlist(youtube, niche, category, config)
        
        if not playlist_id:
            stats["failed"] += 1
            continue
        
        # Add to playlist
        success = add_video_to_playlist(youtube, video_id, playlist_id)
        
        if success:
            stats["added_to_playlists"] += 1
        else:
            stats["already_in_playlists"] += 1
    
    return stats


def print_playlist_summary(config, niche, stats):
    """Print summary with v6.0 categories"""
    print("\n" + "="*70)
    print("🔮 MYTHICA REPORT v6.0 PLAYLIST SUMMARY")
    print("="*70)
    
    if niche in PLAYLIST_RULES:
        active_categories = []
        deprecated_categories = []
        
        for category, rules in PLAYLIST_RULES[niche].items():
            # Separate active from deprecated
            if rules.get("keywords") in [["deprecated"], ["removed"]]:
                deprecated_categories.append((category, rules))
            else:
                active_categories.append((category, rules))
        
        # Show active playlists
        print("\n🆕 ACTIVE PLAYLISTS (v6.0):")
        for category, rules in active_categories:
            key = f"{niche}_{category}"
            video_count = stats.get("by_category", {}).get(category, 0)
            
            if key in config:
                playlist_id = config[key]
                url = f"https://www.youtube.com/playlist?list={playlist_id}"
                
                print(f"\n{rules['title']}")
                print(f"   📍 Category: {category}")
                print(f"   📊 Videos: {video_count}")
                print(f"   🔗 URL: {url}")
                print(f"   📝 {rules['description'][:80]}...")
            else:
                print(f"\n⚠️ {rules['title']}")
                print(f"   Status: Will be auto-created on first video")
        
        # Show deprecated (if any exist)
        if deprecated_categories:
            print("\n⚠️ DEPRECATED CATEGORIES (merged into others):")
            for category, rules in deprecated_categories:
                print(f"   • {category} → {rules['description']}")


if __name__ == "__main__":
    print("🔮 Mythica Report v6.0 - YouTube Playlist Auto-Organizer")
    print("="*70)
    
    niche = "mystery"
    print(f"🎯 Channel Niche: {niche}")
    print(f"🆕 v6.0 Features: Dark History, Medical Mysteries, Dark Experiments")
    
    # Load data
    history = load_upload_history()
    config = load_playlist_config()
    
    if not history:
        print("⚠️ No upload history. Upload videos first!")
        exit(0)
    
    print(f"📂 Found {len(history)} videos")
    
    # Authenticate
    youtube = get_youtube_client()
    
    # Map existing playlists
    config = fetch_and_map_existing_playlists(youtube, niche, config)
    save_playlist_config(config)
    
    # Organize
    stats = organize_playlists(youtube, history, config, niche)
    
    # Results
    print("\n" + "="*70)
    print("📊 ORGANIZATION RESULTS")
    print("="*70)
    print(f"🔮 Total videos: {stats['total_videos']}")
    print(f"✅ Categorized: {stats['categorized']}")
    print(f"📥 Added: {stats['added_to_playlists']}")
    print(f"📋 Already in playlists: {stats['already_in_playlists']}")
    print(f"❌ Failed: {stats['failed']}")
    
    # Breakdown by category
    if stats.get("by_category"):
        print("\n📂 Videos by Category:")
        for category, count in sorted(stats["by_category"].items(), key=lambda x: x[1], reverse=True):
            print(f"   • {category}: {count} videos")
    
    # Summary
    print_playlist_summary(config, niche, stats)
    
    print("\n✅ Playlist organization complete! 🔮")
    print("\n💡 Your dark mystery playlists will automatically grow with each upload!")
    print("   Categories: Dark History (Mon), Medical (Wed), Experiments (Thu)")
    print("   Keep your audience hooked on the unexplained! 👁️")