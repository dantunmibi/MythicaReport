#!/usr/bin/env python3
"""
🔮 Manage Mystery Playlists - MYTHICA REPORT VERSION
Organizes videos into content pillars:
- Unsolved Mysteries (25%)
- Paranormal & Supernatural (25%)
- True Crime Cases (20%)
- Ancient Mysteries (15%)
- Conspiracy Theories (10%)
- Cryptids & Creatures (5%)
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


# 🔮 MYSTERY PLAYLIST CONFIGURATION
PLAYLIST_RULES = {
    "mystery": {
        "unsolved": {
            "title": "🔍 Unsolved Mysteries - Cases That Remain Open",
            "description": "Cold cases, missing persons, and mysteries that still haunt investigators. The truth is still out there waiting to be discovered.",
            "keywords": [
                "unsolved", "mystery", "cold case", "missing", "disappeared", "vanished",
                "never found", "still missing", "unknown", "unexplained", "unanswered",
                "investigation", "detective", "police", "search", "looking for",
                "where is", "what happened", "no answers", "no explanation", "baffling",
                "puzzling", "strange disappearance", "mysterious death", "unresolved",
                "case", "remains open", "active investigation", "clues", "evidence",
                "witnesses", "last seen", "trail went cold", "no leads", "questions"
            ]
        },
        "paranormal": {
            "title": "👻 Paranormal & Supernatural - Beyond Explanation",
            "description": "Ghosts, hauntings, poltergeists, and phenomena that defy scientific explanation. Real encounters with the unknown.",
            "keywords": [
                "paranormal", "supernatural", "ghost", "haunted", "haunting", "spirit",
                "poltergeist", "phantom", "apparition", "specter", "entity", "presence",
                "demon", "possession", "exorcism", "evil", "dark", "shadow", "figure",
                "orb", "evp", "electronic voice phenomenon", "manifestation",
                "ouija", "seance", "medium", "psychic", "clairvoyant", "sixth sense",
                "premonition", "deja vu", "telekinesis", "levitation", "supernatural",
                "otherworldly", "unexplained phenomena", "strange occurrence",
                "haunted house", "haunted location", "cemetery", "graveyard",
                "investigation", "ghost hunter", "paranormal activity", "caught on camera"
            ]
        },
        "true_crime": {
            "title": "🔪 True Crime Cases - Dark Reality",
            "description": "Real crime stories, criminal minds, and the investigations that brought them to justice. Warning: Dark content.",
            "keywords": [
                "true crime", "murder", "killer", "serial killer", "crime", "criminal",
                "investigation", "detective", "fbi", "police", "arrest", "convicted",
                "trial", "court", "guilty", "innocent", "justice", "victim", "suspect",
                "evidence", "forensic", "autopsy", "crime scene", "homicide", "death",
                "kidnapping", "abduction", "assault", "robbery", "heist", "theft",
                "fraud", "scam", "con artist", "criminal mind", "psychopath", "sociopath",
                "motive", "alibi", "witness", "testimony", "confession", "caught",
                "solved", "case closed", "breakthrough", "arrest", "manhunt",
                "wanted", "escaped", "fugitive", "prison", "jail", "sentenced"
            ]
        },
        "ancient": {
            "title": "🗿 Ancient Mysteries - Lost Civilizations",
            "description": "Ancient ruins, lost civilizations, archaeological enigmas. Secrets buried in time that challenge everything we know about history.",
            "keywords": [
                "ancient", "civilization", "lost", "ruins", "archaeological", "archaeology",
                "egypt", "pyramid", "pharaoh", "tomb", "hieroglyphics", "mummy",
                "atlantis", "lemuria", "mu", "ancient astronaut", "alien", "extraterrestrial",
                "sumerian", "anunnaki", "babylonian", "mesopotamia", "ancient technology",
                "oopart", "out of place artifact", "ancient mystery", "unexplained structure",
                "megalith", "stonehenge", "easter island", "moai", "nazca lines",
                "machu picchu", "inca", "maya", "aztec", "olmec", "temple", "monument",
                "antikythera", "baghdad battery", "ancient knowledge", "forbidden",
                "hidden history", "alternative history", "ancient alien", "gods",
                "legend", "myth", "folklore", "ancient text", "dead sea scrolls",
                "ancient civilization", "advanced technology", "before history"
            ]
        },
        "conspiracy": {
            "title": "🕵️ Conspiracy Theories - Question Everything",
            "description": "Government secrets, cover-ups, and hidden agendas. Connecting dots others won't see. Think for yourself.",
            "keywords": [
                "conspiracy", "theory", "cover up", "coverup", "hidden", "secret",
                "government", "cia", "fbi", "nsa", "classified", "top secret", "redacted",
                "they don't want you to know", "suppressed", "censored", "banned",
                "illuminati", "new world order", "nwo", "elite", "cabal", "deep state",
                "shadow government", "puppet master", "control", "manipulation",
                "false flag", "psyop", "operation", "agenda", "plan", "scheme",
                "wake up", "truth", "real truth", "hidden truth", "exposed", "reveal",
                "leak", "whistleblower", "insider", "anonymous", "disclosure",
                "ufo", "area 51", "roswell", "alien", "extraterrestrial", "contact",
                "mk ultra", "project", "experiment", "testing", "program",
                "assassination", "jfk", "conspiracy theory", "questioned", "suspicious"
            ]
        },
        "cryptids": {
            "title": "🦎 Cryptids & Creatures - Things That Shouldn't Exist",
            "description": "Bigfoot, lake monsters, and creatures from the shadows. Eyewitness accounts of beings science won't acknowledge.",
            "keywords": [
                "cryptid", "creature", "monster", "beast", "unknown", "unidentified",
                "bigfoot", "sasquatch", "yeti", "abominable snowman", "skunk ape",
                "loch ness", "nessie", "lake monster", "sea serpent", "sea creature",
                "chupacabra", "mothman", "jersey devil", "dogman", "werewolf",
                "wendigo", "skinwalker", "crawler", "rake", "goatman", "lizard man",
                "thunderbird", "giant bird", "pterodactyl", "living dinosaur",
                "mokele mbembe", "kongamato", "giant", "huge", "massive creature",
                "sighting", "encounter", "spotted", "seen", "witness", "eyewitness",
                "caught on camera", "footage", "video", "photo", "evidence", "proof",
                "real", "exists", "found", "discovered", "track", "footprint", "print",
                "howl", "scream", "sound", "call", "cry", "roar", "mysterious animal",
                "unknown species", "undiscovered", "hidden", "elusive", "legendary"
            ]
        }
    }
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
        title = playlist_info["title"]
        description = playlist_info["description"]
        
        # Add branding
        full_description = f"{description}\n\n🔮 Mythica Report - Investigating mysteries the world forgot.\nSubscribe for weekly deep dives into the unexplained. #mystery #unexplained #mythica"
        
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
    """Smart categorization using keyword matching"""
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
        
        # Bonus for power phrases
        power_phrases = {
            "unsolved": ["unsolved mystery", "cold case", "still missing", "never found", "remains open"],
            "paranormal": ["paranormal activity", "caught on camera", "ghost", "haunted", "supernatural"],
            "true_crime": ["true crime", "serial killer", "murder investigation", "criminal mind"],
            "ancient": ["ancient mystery", "lost civilization", "ancient alien", "forbidden history"],
            "conspiracy": ["conspiracy theory", "cover up", "they don't want", "hidden truth", "exposed"],
            "cryptids": ["cryptid", "bigfoot", "creature", "caught on camera", "unknown species"]
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
    
    print("   ⚠️ No match, defaulting to 'unsolved'")
    return "unsolved"


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
    print(f"\n🎬 Organizing {len(history)} videos into playlists...")
    
    stats = {
        "total_videos": len(history),
        "categorized": 0,
        "added_to_playlists": 0,
        "already_in_playlists": 0,
        "failed": 0
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


def print_playlist_summary(config, niche):
    """Print summary"""
    print("\n" + "="*70)
    print("🔮 MYTHICA REPORT PLAYLIST SUMMARY")
    print("="*70)
    
    if niche in PLAYLIST_RULES:
        for category, rules in PLAYLIST_RULES[niche].items():
            key = f"{niche}_{category}"
            
            if key in config:
                playlist_id = config[key]
                url = f"https://www.youtube.com/playlist?list={playlist_id}"
                
                print(f"\n{rules['title']}")
                print(f"   📍 Category: {category}")
                print(f"   🔗 URL: {url}")
                print(f"   📝 {rules['description'][:80]}...")
            else:
                print(f"\n⚠️ {rules['title']}")
                print(f"   Status: Will be auto-created")


if __name__ == "__main__":
    print("🔮 Mythica Report - YouTube Playlist Auto-Organizer")
    print("="*70)
    
    niche = "mystery"
    print(f"🎯 Channel Niche: {niche}")
    
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
    
    # Summary
    print_playlist_summary(config, niche)
    
    print("\n✅ Playlist organization complete! 🔮")
    print("\n💡 Your mystery playlists will automatically grow with each upload!")
    print("   Keep your audience hooked on the unexplained! 👁️")