#!/usr/bin/env python3
"""
🔍 Generate Mystery Script - RETENTION-OPTIMIZED FOR MYTHICA REPORT
Based on proven performance data: 35-45 seconds = 64.1% retention

🚨 VERSION 6.0: DARK CONTENT EXPANSION + END HOOKS
✅ NEW: Dark History, Disturbing Medical, Dark Experiments categories
✅ NEW: 5-second end hooks (mystery deepening + category promise)
✅ NEW: Predictable schedule awareness (same category same day)
✅ PROVEN: 0:09 cliff fixed (64.1% avg retention vs 18.75% baseline)
"""

import os
import json
import re
import hashlib
import random
from datetime import datetime
import google.generativeai as genai
from tenacity import retry, stop_after_attempt, wait_exponential

TMP = os.getenv("GITHUB_WORKSPACE", ".") + "/tmp"
os.makedirs(TMP, exist_ok=True)
HISTORY_FILE = os.path.join(TMP, "content_history.json")

# 🎯 MYTHICA REPORT OPTIMIZED TARGETS (Updated for end hooks)
OPTIMAL_MIN_DURATION = 40  # Increased from 35 to accommodate end hook
OPTIMAL_MAX_DURATION = 50  # Increased from 45 to accommodate end hook

WORDS_PER_SECOND = 2.1

# Script body targets (before end hook)
SCRIPT_BODY_MIN = 73   # ~35 seconds
SCRIPT_BODY_MAX = 94   # ~45 seconds

# End hook constants
END_HOOK_WORDS = 13    # ~5 seconds (mystery deepening 8 words + category promise 5 words)

# Total targets (script body + end hook)
MINIMUM_WORDS = 86     # 73 + 13 = ~40 seconds total
OPTIMAL_MIN_WORDS = 86 # Script body min + end hook
OPTIMAL_MAX_WORDS = 107 # Script body max + end hook
HARD_LIMIT_WORDS = 120 # Absolute maximum

MYSTERY_WEIGHTS = {
    'disappearance': 0.30,      # Reduced from 0.40 to make room for new categories
    'dark_history': 0.20,       # NEW - Dark historical events
    'disturbing_medical': 0.15, # NEW - Medical mysteries
    'dark_experiments': 0.10,   # NEW - Unethical research
    'phenomena': 0.10,          # Reduced from 0.30
    'crime': 0.10,              # Reduced from 0.15
    'conspiracy': 0.05,         # Reduced from 0.10, merged into dark_experiments
}

print(f"🎯 Mythica Report Optimization v6.0 (DARK CONTENT + END HOOKS):")
print(f"   ✅ Total Target: {OPTIMAL_MIN_DURATION}-{OPTIMAL_MAX_DURATION}s ({OPTIMAL_MIN_WORDS}-{OPTIMAL_MAX_WORDS} words)")
print(f"   📝 Script Body: {SCRIPT_BODY_MIN}-{SCRIPT_BODY_MAX} words (35-45s)")
print(f"   🎬 End Hook: {END_HOOK_WORDS} words (5s - mystery + category promise)")
print(f"   ⚠️ Hard Limit: {HARD_LIMIT_WORDS} words")
print(f"   🔥 NEW CATEGORIES: Dark History (20%), Medical (15%), Experiments (10%)")
print(f"   🎯 0:09 Cliff: FIXED (64.1% avg retention)")

# Configure Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

try:
    models = genai.list_models()
    model_name = None
    for m in models:
        if 'generateContent' in m.supported_generation_methods:
            if '2.0-flash' in m.name or '2.5-flash' in m.name:
                model_name = m.name
                break
            elif '1.5-flash' in m.name and not model_name:
                model_name = m.name
    
    if not model_name:
        model_name = "models/gemini-1.5-flash"
    
    print(f"✅ Using model: {model_name}\n")
    model = genai.GenerativeModel(model_name)
except Exception as e:
    print(f"⚠️ Error listing models: {e}")
    model = genai.GenerativeModel("models/gemini-1.5-flash")


def load_history():
    """Load content history from previous runs"""
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
                history = json.load(f)
                print(f"📂 Loaded {len(history.get('topics', []))} topics from history")
                return history
        except Exception as e:
            print(f"⚠️ Could not load history: {e}")
            return {'topics': [], 'version': '6.0_dark_content_end_hooks'}
    
    print("📂 No previous history found, starting fresh")
    return {'topics': [], 'version': '6.0_dark_content_end_hooks'}


def save_to_history(topic, script_hash, title, script_data):
    """Save generated content to history"""
    history = load_history()
    
    history['topics'].append({
        'topic': topic,
        'title': title,
        'hash': script_hash,
        'hook': script_data.get('hook', ''),
        'key_phrase': script_data.get('key_phrase', ''),
        'mystery_category': script_data.get('mystery_category', 'unknown'),
        'content_type': script_data.get('content_type', 'general'),
        'word_count': script_data.get('word_count', 0),
        'estimated_duration': script_data.get('estimated_duration', 0),
        'has_end_hook': script_data.get('has_end_hook', False),
        'date': datetime.now().isoformat(),
        'timestamp': datetime.now().timestamp()
    })
    
    history['topics'] = history['topics'][-100:]
    history['last_updated'] = datetime.now().isoformat()
    history['version'] = '6.0_dark_content_end_hooks'
    
    with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=2, ensure_ascii=False)
    
    print(f"💾 Saved to history ({len(history['topics'])} total topics)")


def get_content_hash(data):
    """Generate hash of content to detect exact duplicates"""
    content_str = f"{data.get('title', '')}{data.get('hook', '')}{data.get('script', '')}"
    return hashlib.md5(content_str.encode()).hexdigest()


def load_trending():
    """Load trending topics from fetch_trending.py"""
    trending_file = os.path.join(TMP, "trending.json")
    if os.path.exists(trending_file):
        try:
            with open(trending_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"⚠️ Could not load trending data: {e}")
            return None
    return None

def filter_trending_by_category(trends, mystery_type):
    """
    🆕 v6.0.9: Filter trending topics to match mystery category
    
    Prevents category mismatches (e.g., Tunguska Event for medical category)
    
    Args:
        trends: Trending data from fetch_trending.py
        mystery_type: Target category (disturbing_medical, dark_experiments, etc.)
    
    Returns:
        Filtered trending data (None if no matches)
    """
    
    if not trends or not trends.get('topics'):
        return None
    
    # ========================================================================
    # EXPANDED CATEGORY KEYWORD LIBRARY
    # ========================================================================
    category_keywords = {
        'disturbing_medical': {
            'required': [
                # Medical conditions
                'disease', 'illness', 'syndrome', 'condition', 'disorder',
                'epidemic', 'outbreak', 'pandemic', 'infection', 'contagion',
                
                # Symptoms
                'fatal', 'insomnia', 'paralysis', 'paralyzed', 'blind', 'blindness',
                'deaf', 'mute', 'laughing', 'crying', 'shaking', 'tremors',
                'seizures', 'convulsions', 'coma', 'unconscious',
                
                # Body horror
                'turned to stone', 'petrified', 'ossification', 'fop',
                "couldn't sleep", "couldn't wake", "never slept", "never woke",
                'glowing', 'radium', 'radiation poisoning',
                
                # Medical terms
                'medical', 'hospital', 'doctor', 'physician', 'patient',
                'diagnosis', 'treatment', 'cure', 'therapy', 'symptoms',
                'prognosis', 'autopsy', 'pathology', 'virus', 'bacteria',
                
                # Specific conditions (case names)
                'kuru', 'fatal familial insomnia', 'ffi', 'fibrodysplasia',
                'minamata', 'dancing plague', 'encephalitis lethargica',
                'sleeping sickness', 'ergotism', 'tanganyika', 'mass hysteria',
                'conversion disorder', 'psychogenic', 'radiation sickness'
            ],
            'forbidden': [
                'explosion', 'blast', 'meteor', 'asteroid', 'comet',
                'earthquake', 'tsunami', 'volcano', 'natural disaster',
                'war', 'battle', 'military', 'soldier', 'army',
                'ship', 'plane crash', 'flight', 'aircraft', 'vessel',
                'treasure', 'gold', 'riches', 'artifact', 'manuscript',
                'code', 'cipher', 'puzzle', 'signal from space'
            ]
        },
        
        'dark_experiments': {
            'required': [
                # Experiment types
                'experiment', 'study', 'research', 'test', 'trial',
                'project', 'program', 'operation', 'investigation',
                
                # Organizations
                'cia', 'fbi', 'kgb', 'soviet', 'russian', 'government',
                'military', 'navy', 'army', 'agency', 'intelligence',
                
                # Classified/secret
                'classified', 'secret', 'hidden', 'covered up', 'coverup',
                'declassified', 'files', 'documents', 'leaked', 'exposed',
                
                # Specific experiments
                'mk-ultra', 'mkultra', 'mk ultra', 'artichoke', 'bluebird',
                'midnight climax', 'stanford prison', 'milgram', 'tuskegee',
                'unit 731', 'edgewood arsenal', 'project paperclip',
                
                # Research terms
                'subjects', 'participants', 'volunteers', 'tested on',
                'exposed to', 'administered', 'dosage', 'laboratory',
                'clinical trial', 'human testing', 'guinea pigs',
                
                # Psychological
                'psychological', 'mind control', 'brainwashing', 'torture',
                'interrogation', 'sleep deprivation', 'sensory deprivation',
                'behavior modification', 'conditioning', 'trauma-based'
            ],
            'forbidden': [
                'meteor', 'asteroid', 'space', 'cosmic', 'alien',
                'natural disaster', 'earthquake', 'tsunami', 'volcano',
                'ship sinking', 'plane crash', 'flight disappearance',
                'treasure hunt', 'lost gold', 'artifact mystery',
                'disease outbreak' # (unless caused by experiment)
            ]
        },
        
        'dark_history': {
            'required': [
                # Time periods (century markers)
                '1400s', '1500s', '1600s', '1700s', '1800s', '1900s',
                '15th century', '16th century', '17th century', '18th century',
                '19th century', '20th century', 'medieval', 'colonial',
                'victorian', 'elizabethan', 'renaissance',
                
                # Historical terms
                'historical', 'history', 'ancient', 'old', 'past',
                'era', 'period', 'age', 'time', 'years ago',
                
                # Places
                'town', 'village', 'city', 'colony', 'settlement',
                'castle', 'manor', 'estate', 'plantation', 'fort',
                
                # Events
                'event', 'incident', 'occurrence', 'phenomenon', 'disaster',
                'tragedy', 'catastrophe', 'outbreak', 'epidemic', 'plague',
                
                # Mysterious occurrences
                'dark day', 'black sun', 'sky turned', 'rained meat',
                'rained blood', 'dancing plague', 'dancing mania',
                'radium girls', 'glowing', 'luminous', 'mass hysteria',
                'mass poisoning', 'mysterious deaths', 'unexplained deaths',
                
                # Historical figures (generic)
                'people who', 'villagers who', 'workers who', 'settlers who',
                'colonists', 'factory workers', 'miners', 'sailors'
            ],
            'forbidden': [
                'modern', '2000s', '2010s', '2020s', 'recent', 'today',
                'this year', 'last year', 'current', 'ongoing',
                'space signal', 'satellite', 'internet', 'computer',
                'digital', 'online', 'social media', 'reddit', 'youtube'
            ]
        },
        
        'phenomena': {
            'required': [
                # Phenomena types
                'phenomenon', 'phenomena', 'anomaly', 'anomalies',
                'unexplained', 'mysterious', 'strange', 'bizarre',
                'impossible', 'defies explanation', 'paranormal',
                
                # Space/cosmic
                'signal', 'space signal', 'radio signal', 'transmission',
                'wow signal', 'fast radio burst', 'frb', 'pulsar',
                'cosmic', 'stellar', 'interstellar', 'extraterrestrial',
                
                # Lights/visual
                'lights', 'light', 'glow', 'glowing', 'illumination',
                'aurora', 'orbs', 'spheres', 'ufo', 'uap', 'sighting',
                'hessdalen lights', 'marfa lights', 'brown mountain lights',
                
                # Sound
                'sound', 'noise', 'hum', 'the hum', 'taos hum',
                'bloop', 'underwater sound', 'skyquake', 'trumpet sounds',
                'mysterious broadcasts', 'numbers stations', 'radio broadcasts',
                
                # Paranormal
                'ghost', 'apparition', 'haunting', 'poltergeist', 'spirit',
                'phantom', 'supernatural', 'otherworldly', 'dimensional',
                'portal', 'vortex', 'time slip', 'glitch in matrix',
                
                # Unexplained events
                'appeared', 'vanished', 'materialized', 'dematerialized',
                'teleportation', 'levitation', 'spontaneous combustion'
            ],
            'forbidden': [
                'murder', 'killed', 'crime', 'investigation', 'detective',
                'medical condition', 'disease', 'illness', 'syndrome',
                'experiment', 'study', 'research', 'classified'
            ]
        },
        
        'crime': {
            'required': [
                # Crime types
                'murder', 'killed', 'murdered', 'slain', 'death',
                'homicide', 'manslaughter', 'killing', 'serial killer',
                
                # Investigation
                'crime', 'case', 'cold case', 'unsolved', 'investigation',
                'detective', 'police', 'fbi', 'forensic', 'evidence',
                'clues', 'suspect', 'witness', 'testimony', 'trial',
                
                # Victims/criminals
                'victim', 'victims', 'killer', 'murderer', 'criminal',
                'perpetrator', 'suspect', 'person of interest',
                
                # Specific cases
                'zodiac', 'zodiac killer', 'jack the ripper', 'black dahlia',
                'jonbenet', 'jonbenet ramsey', 'missing person', 'amber alert',
                
                # True crime terms
                'true crime', 'mystery', 'who killed', 'who murdered',
                'unsolved murder', 'cold case files', 'disappeared',
                'last seen', 'body found', 'remains discovered'
            ],
            'forbidden': [
                'meteor', 'asteroid', 'space', 'cosmic', 'alien',
                'natural disaster', 'earthquake', 'volcano',
                'medical condition', 'disease outbreak',
                'signal', 'radio transmission', 'paranormal'
            ]
        },
        
        'disappearance': {
            'required': [
                # Core disappearance terms
                'vanished', 'disappeared', 'missing', 'gone', 'lost',
                'never found', 'no trace', 'without trace', 'last seen',
                'search', 'searched', 'searching', 'rescue', 'hunt',
                
                # Transportation
                'flight', 'plane', 'aircraft', 'ship', 'vessel', 'boat',
                'yacht', 'submarine', 'helicopter', 'pilot', 'crew',
                'passengers', 'mh370', 'mh 370', 'flight 19', 'amelia earhart',
                
                # People
                'hiker', 'hikers', 'climber', 'explorer', 'adventurer',
                'tourist', 'traveler', 'student', 'teenager', 'child',
                'woman', 'man', 'person', 'people', 'family', 'group',
                
                # Locations
                'national park', 'wilderness', 'forest', 'mountain',
                'desert', 'ocean', 'sea', 'bermuda triangle', 'alaska',
                
                # Specific cases
                'dyatlov pass', 'roanoke', 'mary celeste', 'db cooper',
                'd.b. cooper', 'maura murray', 'brian shaffer', 'asha degree',
                'brandon swanson', 'brandon lawson', 'leah roberts'
            ],
            'forbidden': [
                'found alive', 'returned', 'came back', 'survived',
                'hoax', 'faked disappearance', 'staged'
            ]
        }
    }
    
    # Get validator for this category
    if mystery_type not in category_keywords:
        # Category not in filter list - allow all trending topics
        print(f"   ℹ️ No category filter for '{mystery_type}' - using all trending topics")
        return trends
    
    validator = category_keywords[mystery_type]
    required_keywords = validator['required']
    forbidden_keywords = validator.get('forbidden', [])
    
    # Filter trending topics
    filtered_topics = []
    filtered_full_data = []
    
    full_data = trends.get('full_data', [])
    
    for idx, topic_title in enumerate(trends.get('topics', [])):
        topic_lower = topic_title.lower()
        
        # Get full topic data if available
        topic_data = None
        if idx < len(full_data):
            topic_data = full_data[idx]
            # Also check story_hook and core_mystery for keywords
            story_hook = topic_data.get('story_hook', '').lower()
            core_mystery = topic_data.get('core_mystery', '').lower()
            combined_text = f"{topic_lower} {story_hook} {core_mystery}"
        else:
            combined_text = topic_lower
        
        # Check forbidden keywords first (instant disqualification)
        has_forbidden = any(keyword in combined_text for keyword in forbidden_keywords)
        if has_forbidden:
            forbidden_found = [kw for kw in forbidden_keywords if kw in combined_text]
            print(f"   ❌ FILTERED OUT: '{topic_title[:60]}'")
            print(f"      Reason: Contains forbidden keywords: {forbidden_found[:3]}")
            continue
        
        # Check required keywords
        has_required = any(keyword in combined_text for keyword in required_keywords)
        
        if has_required:
            matched_keywords = [kw for kw in required_keywords if kw in combined_text]
            print(f"   ✅ MATCHED: '{topic_title[:60]}'")
            print(f"      Keywords: {matched_keywords[:3]}")
            
            filtered_topics.append(topic_title)
            if topic_data:
                filtered_full_data.append(topic_data)
        else:
            print(f"   ⚠️ FILTERED OUT: '{topic_title[:60]}'")
            print(f"      Reason: No required keywords for '{mystery_type}'")
    
    # Return filtered data or None
    if filtered_topics:
        filtered_trends = {
            'topics': filtered_topics,
            'full_data': filtered_full_data,
            'source': f"{trends.get('source', 'unknown')} (filtered for {mystery_type})",
            'generated_at': trends.get('generated_at'),
            'timestamp': trends.get('timestamp'),
            'niche': trends.get('niche'),
            'channel': trends.get('channel'),
            'version': f"{trends.get('version', 'unknown')}_category_filtered"
        }
        
        print(f"\n   ✅ CATEGORY FILTER RESULT:")
        print(f"      Original topics: {len(trends.get('topics', []))}")
        print(f"      Matched topics: {len(filtered_topics)}")
        print(f"      Category: {mystery_type}")
        
        return filtered_trends
    else:
        print(f"\n   ⚠️ NO TRENDING TOPICS MATCH CATEGORY '{mystery_type}'")
        print(f"      Will use pure category generation (ignore trending)")
        return None

def load_schedule():
    """Load posting schedule to determine category for today"""
    schedule_file = os.path.join(os.getenv("GITHUB_WORKSPACE", "."), "config", "posting_schedule.json")
    if os.path.exists(schedule_file):
        try:
            with open(schedule_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"⚠️ Could not load schedule: {e}")
            return None
    return None


def get_category_for_today():
    """
    Determine which category to generate based on current day of week
    This ensures end hook promises are TRUE (same category same day)
    """
    from datetime import datetime
    
    schedule = load_schedule()
    if not schedule:
        print("⚠️ No schedule found, using fallback category selection")
        return None
    
    # Get current day of week
    day_name = datetime.now().strftime('%A')  # Monday, Tuesday, etc.
    
    weekly_schedule = schedule.get('schedule', {}).get('weekly_schedule', {})
    
    if day_name in weekly_schedule:
        day_slots = weekly_schedule[day_name]
        if day_slots and len(day_slots) > 0:
            # Use first slot's mystery_type
            mystery_type = day_slots[0].get('mystery_type', None)
            if mystery_type:
                print(f"📅 Today is {day_name} → Category: {mystery_type}")
                return mystery_type
    
    print(f"⚠️ No category mapping for {day_name}, using fallback")
    return None


def is_similar_topic(new_title, previous_titles, similarity_threshold=0.6):
    """
    🔧 v6.0.10: FIXED - Hybrid duplicate detection
    
    Primary: Entity-based comparison (accurate, allows pattern reuse)
    Fallback: Improved string comparison (fixed time decay bug)
    
    Args:
        new_title: Title to check
        previous_titles: List of previous titles
        similarity_threshold: NOT USED (kept for compatibility)
    
    Returns:
        True if duplicate detected, False if unique
    """
    
    # ========================================================================
    # PRIMARY: ENTITY-BASED DETECTION (Recommended)
    # ========================================================================
    
    is_dup, entity_sim, matched_title = is_duplicate_by_entities(
        new_title, 
        previous_titles,
        entity_threshold=0.70,  # 70% entity overlap = duplicate
        recent_count=30
    )
    
    if is_dup:
        print(f"⚠️ Entity-based duplicate detected ({entity_sim:.2%} overlap)")
        print(f"   Matched: {matched_title}")
        return True
    
    # ========================================================================
    # FALLBACK: IMPROVED STRING COMPARISON (if entity method unclear)
    # ========================================================================
    
    # Only check if entity similarity was borderline (50-70%)
    if entity_sim > 0.50:
        print(f"   ⚠️ Borderline entity similarity ({entity_sim:.2%}), checking string similarity...")
        
        new_words = set(new_title.lower().split())
        
        for idx, prev_title in enumerate(reversed(previous_titles[-30:])):
            prev_words = set(prev_title.lower().split())
            
            intersection = len(new_words & prev_words)
            union = len(new_words | prev_words)
            
            if union > 0:
                base_similarity = intersection / union
                
                # FIXED: Time decay (older = MORE lenient, was backwards)
                videos_ago = idx
                leniency_boost = videos_ago * 0.02  # 2% more lenient per video
                adjusted_threshold = min(0.85, 0.70 + leniency_boost)  # Cap at 85%
                
                if base_similarity > adjusted_threshold:
                    print(f"   ❌ String similarity also high:")
                    print(f"      {base_similarity:.2%} > {adjusted_threshold:.2%}")
                    print(f"      Vs: {prev_title} (~{videos_ago} videos ago)")
                    return True
    
    # ========================================================================
    # RESULT: UNIQUE TOPIC
    # ========================================================================
    
    print(f"   ✅ Topic validation: UNIQUE (passed entity + string checks)")
    return False

def extract_story_entities(title):
    """
    🆕 v6.0.10: Extract unique story identifiers (entities) from title
    
    Removes pattern words and common words, keeps:
    - Proper nouns (names, places, organizations)
    - Specific objects (plane, ship, desert, forest)
    - Unique action verbs (glowed, jumped, danced, rained)
    - Numbers/dates (1780, MH370, Flight 19)
    
    Returns:
        Set of unique entity words (lowercase)
    """
    
    # Pattern words to ignore (your proven title patterns)
    PATTERN_WORDS = {
        'the', 'who', 'that', 'which', 'where', 'when',
        'vanished', 'disappeared', 'missing', 'gone', 'lost',
        'from', 'in', 'at', 'to', 'of', 'a', 'an',
        'mystery', 'case', 'unsolved', 'secret', 'hidden',
        'found', 'discovered', 'never', 'no', 'one'
    }
    
    # Clean and tokenize
    title_lower = title.lower()
    
    # Remove subtitle separator (keep both parts)
    title_lower = title_lower.replace(':', ' ')
    
    # Split into words
    words = title_lower.split()
    
    # Filter out pattern words and short words
    entities = {
        word for word in words 
        if word not in PATTERN_WORDS 
        and len(word) > 2  # Keep words >2 chars
        and not word.isdigit()  # Remove standalone numbers (keep "mh370")
    }
    
    return entities

def is_duplicate_by_entities(new_title, previous_titles, entity_threshold=0.70, recent_count=30):
    """
    🆕 v6.0.10: Entity-based duplicate detection
    
    Compares story entities (nouns, proper names, locations) instead of full titles.
    Allows pattern word reuse ("The [X] Who Vanished").
    
    Args:
        new_title: Current title to check
        previous_titles: List of previous title strings
        entity_threshold: Minimum entity overlap to consider duplicate (default 70%)
        recent_count: How many recent videos to check (default 30)
    
    Returns:
        Tuple: (is_duplicate: bool, similarity: float, matched_title: str or None)
    """
    
    # Extract entities from new title
    new_entities = extract_story_entities(new_title)
    
    if not new_entities:
        print(f"   ⚠️ No entities extracted from: '{new_title}'")
        return False, 0.0, None
    
    print(f"   🔍 New title entities: {new_entities}")
    
    # Check against recent titles
    for idx, prev_title in enumerate(reversed(previous_titles[-recent_count:])):
        prev_entities = extract_story_entities(prev_title)
        
        if not prev_entities:
            continue
        
        # Calculate entity overlap
        intersection = len(new_entities & prev_entities)
        union = len(new_entities | prev_entities)
        
        if union == 0:
            continue
        
        entity_similarity = intersection / union
        
        # Apply time decay (older = MORE lenient, fixed from current bug)
        videos_ago = idx
        decay_factor = max(0.7, 1.0 - (videos_ago * 0.01))  # 1% more lenient per video back
        adjusted_threshold = entity_threshold * decay_factor
        
        if entity_similarity > adjusted_threshold:
            print(f"   ❌ ENTITY DUPLICATE DETECTED:")
            print(f"      New: {new_title}")
            print(f"      Vs:  {prev_title} (~{videos_ago} videos ago)")
            print(f"      Entity overlap: {entity_similarity:.2%} > {adjusted_threshold:.2%}")
            print(f"      Shared entities: {new_entities & prev_entities}")
            return True, entity_similarity, prev_title
    
    print(f"   ✅ Entity check: UNIQUE (no duplicates in last {recent_count} videos)")
    return False, 0.0, None

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def generate_script_with_retry(prompt):
    """Generate script with automatic retry on failure"""
    response = model.generate_content(prompt)
    return response.text.strip()


def generate_end_hook(mystery_type):
    """
    🎬 Generate 5-second end hook (Hybrid approach v6.0)
    
    Structure:
    - 3 seconds (8 words): Mystery deepening for THIS case
    - 2 seconds (5 words): Category promise (enforceable via schedule)
    
    Total: ~13 words, 5 seconds
    
    NO FAKE PROMISES:
    - Category promises are TRUE (enforced by posting_schedule.json)
    - No "Part 2" or specific case promises
    - Each video is standalone
    """
    
    # Mystery deepening templates (3 seconds, 8 words max)
    mystery_endings = {
        'dark_history': [
            "The truth was buried. History forgot.",
            "To this day, no explanation exists.",
            "The evidence vanished. Questions remain.",
            "They covered it up. Files destroyed.",
        ],
        'disturbing_medical': [
            "Doctors still can't explain it today.",
            "Medical science has no answer yet.",
            "The condition defies all known medicine.",
            "No cure exists. No survivors either.",
        ],
        'dark_experiments': [
            "The files remain classified. Secrets buried.",
            "Government denies it ever happened officially.",
            "The truth is still being hidden.",
            "Subjects never spoke. Dead within years.",
        ],
        'disappearance': [
            "They were never found. No trace.",
            "To this day, they're still missing.",
            "The evidence doesn't add up still.",
            "Search continues. Families still wait.",
        ],
        'phenomena': [
            "No explanation exists. None makes sense.",
            "Science cannot explain this phenomenon today.",
            "The mystery deepens every year still.",
            "Witnesses vanished. Evidence destroyed. Silence remains.",
        ],
        'crime': [
            "The case remains unsolved to today.",
            "No arrests. No answers. Nothing resolved.",
            "The killer was never caught ever.",
            "Cold case. Files sealed. Justice denied.",
        ],
        'conspiracy': [
            "Official story contradicts all evidence here.",
            "The truth was buried. Witnesses silenced.",
            "Documents declassified. Questions remain unanswered still.",
        ],
        'historical': [
            "Historians still debate what happened here.",
            "The truth died with them all.",
            "No records survived. Mystery remains forever.",
        ],
    }
    
    # Category promise templates (v6.0.9 - Subscribe CTA with "more" benefit)
    # Format: "Subscribe for more [category] every [day]."
    # Total: ~8-9 words, 4 seconds
    category_promises = {
        'dark_history': "Subscribe for more dark history every Monday.",
        'disturbing_medical': "Subscribe for more medical mysteries every Wednesday.",
        'dark_experiments': "Subscribe for more secret research every Thursday.",
        'disappearance': "Subscribe for more unsolved cases every Tuesday.",
        'phenomena': "Subscribe for more phenomena every Sunday.",
        'crime': "Subscribe for more true crime every Saturday.",
        'conspiracy': "Subscribe for more mysteries every week.",
        'historical': "Subscribe for more dark history every Monday.",
    }
    
    # Select random mystery ending for variety
    mystery_end = random.choice(mystery_endings.get(mystery_type, mystery_endings['disappearance']))
    
    # Get category promise (MUST be consistent with schedule)
    category_promise = category_promises.get(mystery_type, "More mysteries every week.")
    
    # Combine
    end_hook = f"{mystery_end} {category_promise}"
    
    print(f"   🎬 Generated end hook ({len(end_hook.split())} words): '{end_hook}'")
    
    return end_hook


def validate_script_uses_trending_topic(script_data, trending_topics):
    """Validate that script actually uses one of the trending topics"""
    if not trending_topics:
        return True
    
    script_text = f"{script_data['title']} {script_data['hook']} {script_data.get('script', '')}".lower()
    
    trend_keywords = []
    for topic in trending_topics:
        words = [w for w in topic.lower().split() if len(w) > 4 and w not in {
            'this', 'that', 'with', 'from', 'will', 'just', 'your', 'they',
            'them', 'what', 'when', 'where', 'which', 'while', 'about',
            'have', 'been', 'were', 'their', 'there', 'these', 'those',
            'make', 'made', 'take', 'took', 'very', 'more', 'most', 'some',
            'other', 'into', 'than', 'then', 'here'
        }]
        trend_keywords.extend(words)
    
    trend_keywords = list(set(trend_keywords))
    matches = sum(1 for kw in trend_keywords if kw in script_text)
    
    if matches < 2:
        print(f"⚠️ Script doesn't use trending topics! Only {matches} matches.")
        return False
    
    print(f"✅ Script uses trending topics ({matches} keyword matches)")
    return True


def validate_script_structure(script_text):
    """
    🚨 v6.0: Structural validation (unchanged from v3.1.1)
    
    Prevents retention collapse at 0:09 (18.75% → 64.1% proven)
    """
    
    print("\n🔍 STRUCTURAL VALIDATION v6.0 (0:09 Cliff Prevention):")
    
    words = script_text.split()
    first_50 = ' '.join(words[:50]).lower()
    first_100 = ' '.join(words[:100]).lower()
    
    # ✅ CHECK 1: Early reveal (prevents 0:09 drop)
    reveal_words = ['vanished', 'disappeared', 'found', 'discovered', 'never', 'no one', 'gone', 'missing', 'died', 'killed']
    has_early_reveal = any(word in first_50 for word in reveal_words)
    
    if not has_early_reveal:
        print("   ❌ BLOCKED: No reveal in first 50 words")
        raise ValueError(
            "Script REJECTED: Mystery reveal must appear in first 50 words. "
            f"Required words: {', '.join(reveal_words)}"
        )
    
    print(f"   ✅ Early reveal: YES ({[w for w in reveal_words if w in first_50]})")
    
    # ✅ CHECK 2: Twist/contradiction phrase
    twist_phrases = [
        'but here', 'the strange part', 'the impossible part', 
        'the terrifying part', 'the mystery', 'what makes this',
        'but what', 'the twist', 'the catch', 'the problem',
        'but the', 'the disturbing', 'the shocking'
    ]
    has_twist_phrase = any(phrase in script_text.lower() for phrase in twist_phrases)
    
    if not has_twist_phrase:
        print("   ⚠️ WARNING: Missing twist phrase (recommended but not blocking)")
    else:
        matching_phrases = [p for p in twist_phrases if p in script_text.lower()]
        print(f"   ✅ Twist phrase: YES ({matching_phrases[0]})")
    
    # ✅ CHECK 3: CHRONOLOGICAL BACKSTORY DETECTION (Leah Roberts killer)
    # v6.0.8 UPDATE: Reduced false positives, check only first 50 words
    backstory_indicators = [
        # TIER 1: Absolute backstory killers (always block in first 50)
        'was born', 'grew up', 'was raised',
        'was known for', 'had always', 'had always wanted',
        'at the age of', 'years old when', 'x years old',
        'graduated from', 'studied at', 'enrolled in',
        
        # TIER 2: Activity patterns (only block if in first 30 words)
        # These are moved to separate check below
    ]
    
    # Separate check for activity patterns (only in FIRST 30 words)
    first_30 = ' '.join(words[:30]).lower()
    activity_patterns = [
        'was road-tripping', 'was traveling to', 'was driving to',
        'was hiking in', 'was visiting',
        'was on a trip to', 'was on a journey',
        'had been traveling', 'had planned to visit',
        'was heading to', 'was going to'
    ]
    
    # Check Tier 1 backstory in first 50 words (was first 100)
    has_early_backstory = any(phrase in first_50 for phrase in backstory_indicators)
    
    # Check activity patterns only in first 30 words
    has_activity_opening = any(phrase in first_30 for phrase in activity_patterns)
    
    if has_early_backstory:
        backstory_found = [phrase for phrase in backstory_indicators if phrase in first_50]
        print(f"   ❌ BLOCKED: Chronological backstory in first 50 words")
        print(f"   💡 Found: {backstory_found}")
        raise ValueError(
            "Script REJECTED: Uses chronological backstory in first 50 words. "
            "This causes 0:09 retention drop. Start with OUTCOME, not biography."
        )
    
    if has_activity_opening:
        activity_found = [phrase for phrase in activity_patterns if phrase in first_30]
        print(f"   ❌ BLOCKED: Activity-based opening in first 30 words")
        print(f"   💡 Found: {activity_found}")
        raise ValueError(
            "Script REJECTED: Starts with activity description in first 30 words. "
            "Start with MYSTERY REVEAL, then add context. (Leah Roberts pattern)"
        )
    
    print("   ✅ No early backstory: YES (0:09 cliff avoided)")
    print("   ✅ No activity opening: YES (mystery-first confirmed)")
    
    print("   ✅ STRUCTURAL VALIDATION PASSED\n")
    return True

def validate_category_topic_alignment(data, mystery_type):
    """
    🆕 v6.0.9: Validate that script topic matches mystery category
    
    Prevents off-topic scripts (e.g., Tunguska Event for medical category)
    
    Args:
        data: Script data dict
        mystery_type: Expected category
    
    Raises:
        ValueError: If topic doesn't match category
    
    Returns:
        True if validation passes
    """
    
    title = data.get("title", "").lower()
    script = data.get("script", "").lower()
    hook = data.get("hook", "").lower()
    combined_text = f"{title} {hook} {script}"
    
    print(f"\n🔍 CATEGORY-TOPIC ALIGNMENT VALIDATION:")
    print(f"   Expected category: {mystery_type}")
    print(f"   Title: {data.get('title', 'N/A')[:70]}")
    
    # ========================================================================
    # CATEGORY VALIDATORS (same keywords as filter, expanded)
    # ========================================================================
    category_validators = {
        'disturbing_medical': {
            'required_keywords': [
                # Medical conditions
                'disease', 'illness', 'syndrome', 'condition', 'disorder',
                'epidemic', 'outbreak', 'pandemic', 'infection', 'contagion',
                
                # Symptoms
                'fatal', 'insomnia', 'paralysis', 'paralyzed', 'blind', 'blindness',
                'deaf', 'mute', 'laughing', 'crying', 'shaking', 'tremors',
                'seizures', 'convulsions', 'coma', 'unconscious',
                
                # Body horror
                'turned to stone', 'petrified', 'ossification', 'fop',
                "couldn't sleep", "couldn't wake", "never slept", "never woke",
                'glowing', 'radium', 'radiation poisoning',
                
                # Medical terms
                'medical', 'hospital', 'doctor', 'physician', 'patient',
                'diagnosis', 'treatment', 'cure', 'therapy', 'symptoms',
                'prognosis', 'autopsy', 'pathology', 'virus', 'bacteria',
                
                # Specific conditions
                'kuru', 'fatal familial insomnia', 'ffi', 'fibrodysplasia',
                'minamata', 'dancing plague', 'encephalitis lethargica',
                'sleeping sickness', 'ergotism', 'tanganyika', 'mass hysteria'
            ],
            'forbidden_keywords': [
                'explosion', 'blast', 'detonation', 'meteor', 'asteroid', 'comet',
                'earthquake', 'tsunami', 'volcano', 'eruption', 'natural disaster',
                'war', 'battle', 'combat', 'military operation', 'bombing',
                'plane crash', 'aircraft', 'flight disappeared', 'ship sank',
                'treasure', 'gold', 'riches', 'artifact', 'manuscript',
                'code', 'cipher', 'cryptic message', 'signal from space'
            ],
            'error_message': (
                "❌ CATEGORY MISMATCH: This topic is NOT a medical mystery.\n"
                "   Disturbing medical mysteries MUST involve:\n"
                "   - Diseases, conditions, syndromes, or health phenomena\n"
                "   - Medical anomalies or unexplained symptoms\n"
                "   - Body horror elements with medical basis\n"
                "\n"
                "   This topic appears to be about: {detected_topic}\n"
                "\n"
                "   ✅ CORRECT EXAMPLES:\n"
                "      • 'The Man Who Never Slept: Fatal Insomnia Mystery'\n"
                "      • 'The Girls Who Glowed: Radium Poisoning Case'\n"
                "      • 'The Town That Went Blind: Minamata Disease'\n"
                "\n"
                "   ❌ WRONG (what you generated):\n"
                "      • Space disasters (Tunguska Event)\n"
                "      • Natural disasters (earthquakes, meteors)\n"
                "      • Historical events without medical aspect"
            )
        },
        
        'dark_experiments': {
            'required_keywords': [
                'experiment', 'study', 'research', 'test', 'trial',
                'project', 'program', 'operation', 'investigation',
                'cia', 'fbi', 'kgb', 'soviet', 'government', 'military',
                'classified', 'secret', 'hidden', 'covered up', 'coverup',
                'mk-ultra', 'mkultra', 'artichoke', 'stanford prison',
                'milgram', 'tuskegee', 'unit 731', 'edgewood',
                'subjects', 'participants', 'tested', 'exposed',
                'psychological', 'mind control', 'brainwashing', 'torture',
                'sleep deprivation', 'sensory deprivation', 'conditioning'
            ],
            'forbidden_keywords': [
                'meteor', 'asteroid', 'space object', 'cosmic event',
                'natural disaster', 'earthquake', 'tsunami', 'volcano',
                'ship sinking', 'plane crash', 'flight vanished',
                'treasure hunt', 'lost gold', 'buried riches',
                'disease outbreak' # (unless caused by experiment)
            ],
            'error_message': (
                "❌ CATEGORY MISMATCH: This topic is NOT a dark experiment.\n"
                "   Dark experiments MUST involve:\n"
                "   - Human/psychological research or testing\n"
                "   - Government/military classified programs\n"
                "   - Unethical scientific studies\n"
                "   - Secret research with mysterious outcomes\n"
                "\n"
                "   This topic appears to be about: {detected_topic}\n"
                "\n"
                "   ✅ CORRECT EXAMPLES:\n"
                "      • 'The Sleep Study That Failed: Russian Experiment'\n"
                "      • 'The CIA Project That Erased Memories: MK-Ultra'\n"
                "      • 'The Prison Experiment Gone Wrong: Stanford 1971'\n"
                "\n"
                "   ❌ WRONG (what you generated):\n"
                "      • Space events (Tunguska, meteors)\n"
                "      • Natural phenomena\n"
                "      • Historical events without experimental aspect"
            )
        },
        
        'dark_history': {
            'required_keywords': [
                # Time markers
                '1400', '1500', '1600', '1700', '1800', '1900',
                'century', 'historical', 'history', 'ancient', 'era', 'period',
                
                # Places
                'town', 'village', 'city', 'colony', 'settlement',
                
                # Events
                'event', 'incident', 'occurrence', 'phenomenon',
                'dark day', 'sky turned', 'rained', 'dancing plague',
                'radium girls', 'glowing', 'mass hysteria', 'mysterious deaths',
                
                # People groups
                'people who', 'villagers', 'workers', 'settlers', 'colonists'
            ],
            'forbidden_keywords': [
                'modern experiment', '2000s', '2010s', '2020s',
                'recent years', 'this decade', 'current events',
                'space signal', 'satellite', 'internet mystery',
                'social media', 'reddit mystery', 'youtube mystery'
            ],
            'error_message': (
                "❌ CATEGORY MISMATCH: This topic is NOT a dark history mystery.\n"
                "   Dark history MUST involve:\n"
                "   - Historical events (pre-2000) with mysterious elements\n"
                "   - Unexplained occurrences from the past\n"
                "   - Dark/disturbing historical phenomena\n"
                "\n"
                "   This topic appears to be about: {detected_topic}\n"
                "\n"
                "   ✅ CORRECT EXAMPLES:\n"
                "      • 'The Town That Couldn't Stop Dancing: 1518 Plague'\n"
                "      • 'The Day It Rained Meat: 1876 Kentucky Mystery'\n"
                "      • 'The Girls Who Glowed: Radium Factory 1917'\n"
                "\n"
                "   ❌ WRONG (what you generated):\n"
                "      • Modern events (post-2000)\n"
                "      • Space mysteries without historical context\n"
                "      • Internet/digital age mysteries"
            )
        },
        
        'phenomena': {
            'required_keywords': [
                'phenomenon', 'phenomena', 'anomaly', 'unexplained',
                'signal', 'lights', 'sound', 'hum', 'broadcast',
                'wow signal', 'hessdalen', 'marfa lights', 'bloop',
                'ufo', 'uap', 'sighting', 'ghost', 'haunting',
                'paranormal', 'supernatural', 'portal', 'vortex',
                'appeared', 'materialized', 'spontaneous'
            ],
            'forbidden_keywords': [
                'murder', 'killed', 'crime scene', 'investigation',
                'medical condition', 'disease', 'syndrome',
                'experiment conducted', 'study performed', 'research project'
            ],
            'error_message': (
                "❌ CATEGORY MISMATCH: This topic is NOT an unexplained phenomenon.\n"
                "   Phenomena MUST involve:\n"
                "   - Unexplained events or paranormal activity\n"
                "   - Mysterious signals, lights, or sounds\n"
                "   - Supernatural or impossible occurrences\n"
                "\n"
                "   This topic appears to be about: {detected_topic}"
            )
        },
        
        'crime': {
            'required_keywords': [
                'murder', 'killed', 'murdered', 'slain', 'death',
                'crime', 'cold case', 'unsolved', 'investigation',
                'detective', 'police', 'forensic', 'evidence',
                'victim', 'killer', 'suspect', 'zodiac', 'ripper',
                'true crime', 'who killed', 'body found'
            ],
            'forbidden_keywords': [
                'meteor', 'asteroid', 'space', 'cosmic',
                'natural disaster', 'medical condition',
                'paranormal', 'signal', 'phenomenon'
            ],
            'error_message': (
                "❌ CATEGORY MISMATCH: This topic is NOT a true crime mystery.\n"
                "   True crime MUST involve:\n"
                "   - Murders or unsolved deaths\n"
                "   - Criminal investigations\n"
                "   - Cold cases\n"
                "\n"
                "   This topic appears to be about: {detected_topic}"
            )
        },
        
        'disappearance': {
            'required_keywords': [
                'vanished', 'disappeared', 'missing', 'gone', 'lost',
                'never found', 'no trace', 'last seen', 'search',
                'flight', 'ship', 'hiker', 'person', 'crew',
                'mh370', 'flight 19', 'amelia earhart', 'db cooper',
                'dyatlov pass', 'roanoke', 'mary celeste'
            ],
            'forbidden_keywords': [
                'found alive', 'returned', 'came back', 'hoax'
            ],
            'error_message': (
                "❌ CATEGORY MISMATCH: This topic is NOT a disappearance mystery.\n"
                "   Disappearances MUST involve:\n"
                "   - People, vehicles, or groups that vanished\n"
                "   - Unexplained absences\n"
                "   - Missing persons cases\n"
                "\n"
                "   This topic appears to be about: {detected_topic}"
            )
        }
    }
    
    # Skip validation if category not in validators
    if mystery_type not in category_validators:
        print(f"   ℹ️ No validator for '{mystery_type}' - skipping alignment check")
        return True
    
    validator = category_validators[mystery_type]
    required = validator['required_keywords']
    forbidden = validator.get('forbidden_keywords', [])
    
    # Check required keywords
    matching_required = [kw for kw in required if kw in combined_text]
    has_required = len(matching_required) > 0
    
    # Check forbidden keywords
    matching_forbidden = [kw for kw in forbidden if kw in combined_text]
    has_forbidden = len(matching_forbidden) > 0
    
    # Detect topic type for error message
    detected_topic = "unknown topic"
    if 'explosion' in combined_text or 'blast' in combined_text:
        detected_topic = "space/natural disaster"
    elif 'meteor' in combined_text or 'asteroid' in combined_text:
        detected_topic = "cosmic event"
    elif 'ship' in combined_text and 'sank' in combined_text:
        detected_topic = "maritime disaster"
    elif 'plane' in combined_text or 'flight' in combined_text:
        detected_topic = "aviation incident"
    elif 'war' in combined_text or 'battle' in combined_text:
        detected_topic = "military history"
    elif 'treasure' in combined_text or 'gold' in combined_text:
        detected_topic = "treasure hunt/artifact mystery"
    elif 'code' in combined_text or 'cipher' in combined_text:
        detected_topic = "cryptography/code mystery"
    
    # Validation logic
    if has_forbidden:
        print(f"   ❌ FORBIDDEN KEYWORDS DETECTED: {matching_forbidden[:3]}")
        raise ValueError(
            validator['error_message'].format(detected_topic=detected_topic)
        )
    
    if not has_required:
        print(f"   ❌ NO REQUIRED KEYWORDS FOUND")
        print(f"      Required (any of): {required[:10]}...")
        raise ValueError(
            validator['error_message'].format(detected_topic=detected_topic)
        )
    
    print(f"   ✅ Category-topic alignment: VALID")
    print(f"      Matched keywords: {matching_required[:3]}")
    
    return True

def validate_script_data(data):
    """✅ v6.0: Word count + structural + end hook validation"""
    
    required_fields = ["title", "topic", "hook", "script", "cta"]
    
    for field in required_fields:
        if field not in data:
            raise ValueError(f"❌ Missing required field: {field}")
    
    if not isinstance(data["script"], str):
        raise ValueError("❌ Script must be a string")
    
    # ✅ WORD COUNT VALIDATION (script body only, before end hook)
    script_body = data["script"]
    word_count = len(script_body.split())
    estimated_duration = word_count / WORDS_PER_SECOND
    
    print(f"\n📊 Script Body Analysis (before end hook):")
    print(f"   Words: {word_count}")
    print(f"   Estimated duration: {estimated_duration:.1f}s")
    print(f"   Target range: {SCRIPT_BODY_MIN}-{SCRIPT_BODY_MAX} words (35-45s)")
    
    if word_count < SCRIPT_BODY_MIN - 10:  # Allow 10-word buffer
        print(f"   ❌ TOO SHORT: {word_count} words")
        raise ValueError(f"Script body too short: {word_count} words (minimum {SCRIPT_BODY_MIN})")
    
    if word_count > SCRIPT_BODY_MAX + 10:  # Allow 10-word buffer
        print(f"   ❌ TOO LONG: {word_count} words")
        raise ValueError(f"Script body too long: {word_count} words (max {SCRIPT_BODY_MAX})")
    
    if SCRIPT_BODY_MIN <= word_count <= SCRIPT_BODY_MAX:
        print(f"   ✅ PERFECT: {word_count} words (optimal retention zone)")
    else:
        print(f"   ✅ ACCEPTABLE: {word_count} words (within buffer)")
    
    data["word_count"] = word_count
    data["estimated_duration"] = estimated_duration
    data["optimization_version"] = "6.0_dark_content_end_hooks"
    
    # ✅ STRUCTURAL VALIDATION (prevents 0:09 cliff)
    validate_script_structure(script_body)
    
    # Validate title
    if len(data["title"]) > 100:
        print(f"⚠️ Title too long ({len(data['title'])} chars), truncating...")
        data["title"] = data["title"][:97] + "..."
    
    # ✅ Hook validation (v6.0.8 - expanded for dark content categories)
    hook_text = data.get("hook", "")
    power_words = [
        # Core mystery words (disappearances)
        'vanished', 'disappeared', 'found', 'discovered', 'missing', 'gone',
        
        # Dark outcomes
        'died', 'killed', 'dead', 'death', 'murdered',
        
        # Mystery intensifiers
        'mystery', 'never', 'impossible', 'unexplained', 'unknown',
        
        # Dark history specific
        'glowed', 'turned', 'rained', 'burning', 'screaming', 'dancing',
        "couldn't stop", "can't explain", 'buried', 'frozen', 'petrified',
        
        # Medical mysteries
        "couldn't sleep", "couldn't wake", "turned to stone", 'paralyzed',
        'blind', 'deaf', 'mute', 'laughing', 'crying', 'shaking',
        
        # Experiments
        'classified', 'secret', 'hidden', 'covered up', 'experiments',
        'subjects', 'tested', 'exposed',
        
        # Phenomena
        'appeared', 'vanished', 'signal', 'lights', 'sounds', 'voices'
    ]
    has_power_word = any(word in hook_text.lower() for word in power_words)
    
    if not has_power_word:
        print(f"   ❌ BLOCKED: Hook lacks power words")
        raise ValueError(f"Hook must contain mystery word. Current: {hook_text}")
    
    print(f"   ✅ Hook power words: YES")
    
    # ✅ Title specificity validation (v6.0.9 - prevent generic 0-view titles)
    title_text = data.get("title", "")
    
    # Generic patterns that need subtitles for specificity
    generic_patterns = [
        (r'^The Man Who\s+\w+(?!.*:)', "The Man Who [X]"),
        (r'^The Woman Who\s+\w+(?!.*:)', "The Woman Who [X]"),
        (r'^The Person Who\s+\w+(?!.*:)', "The Person Who [X]"),
        (r'^The People Who\s+\w+(?!.*:)', "The People Who [X]"),
        (r'^The Girl Who\s+\w+(?!.*:)', "The Girl Who [X]"),
        (r'^The Boy Who\s+\w+(?!.*:)', "The Boy Who [X]"),
        (r'^The Condition That\s+\w+(?!.*:)', "The Condition That [X]"),
        (r'^The Disease That\s+\w+(?!.*:)', "The Disease That [X]"),
        (r'^The Experiment That\s+\w+(?!.*:)', "The Experiment That [X]"),
    ]
    
    # Check if title matches generic pattern without subtitle
    is_too_generic = False
    matched_pattern = None
    
    for pattern, pattern_name in generic_patterns:
        if re.search(pattern, title_text):
            is_too_generic = True
            matched_pattern = pattern_name
            break
    
    # Exception: Allow if title has subtitle (contains colon)
    if is_too_generic and ':' not in title_text:
        print(f"   ❌ BLOCKED: Title too generic without subtitle")
        print(f"   💡 Pattern: {matched_pattern}")
        raise ValueError(
            f"Title too generic: '{title_text}'. "
            f"Generic pattern '{matched_pattern}' requires subtitle for specificity. "
            "Add location/name/detail after colon. "
            "Examples:\n"
            "  ✅ 'The Man Who Couldn't Sleep: Fatal Insomnia Case'\n"
            "  ✅ 'The Experiment That Failed: MK-Ultra Files'\n"
            "  ✅ 'The Town That Vanished: Roanoke Mystery'\n"
            "  ❌ 'The Man Who Couldn't Sleep' (too generic)"
        )
    
    if is_too_generic and ':' in title_text:
        print(f"   ✅ Title specificity: YES (subtitle adds context)")
    elif not is_too_generic:
        print(f"   ✅ Title specificity: YES (specific enough)")
    
    # Title pattern check
    if title_text.startswith("The ") and any(word in title_text.lower() for word in ['vanished', 'disappeared', 'glowed', 'turned', 'died']):
        print(f"   ✅ Title follows proven pattern: 'The [X] Who/That [Y]'")
    
    # 🆕 v6.0.9: Category-topic alignment validation
    # Get mystery_type from data (set during generation)
    mystery_type = data.get('mystery_category', 'unknown')
    if mystery_type != 'unknown':
        validate_category_topic_alignment(data, mystery_type)

    print(f"✅ Script validation PASSED (v6.0.9)\n")
    return True


def extract_json_from_response(raw_text):
    """Extract JSON from Gemini response"""
    json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', raw_text, re.DOTALL)
    if json_match:
        print("✅ Found JSON in code block")
        return json_match.group(1)
    
    json_match = re.search(r'\{.*\}', raw_text, re.DOTALL)
    if json_match:
        print("✅ Found raw JSON")
        return json_match.group(0)
    
    raise ValueError("❌ No JSON found in response")


def clean_script_text(text):
    """Clean script text of problematic characters"""
    text = text.replace('"', '').replace('"', '')
    text = text.replace(''', "'").replace(''', "'")
    text = text.replace('\u2018', "'").replace('\u2019', "'")
    text = text.replace('\u201c', '').replace('\u201d', '')
    return text


def select_weighted_mystery_type(content_type, user_mystery_type):
    """
    Select mystery type based on performance weights + schedule awareness
    v6.0: Checks posting_schedule.json for today's category
    """
    
    # First, check if user specified a type
    if user_mystery_type and user_mystery_type != 'auto':
        print(f"🎯 User-specified mystery type: {user_mystery_type}")
        return user_mystery_type
    
    # Second, check schedule for today's category (ensures end hook promises are TRUE)
    scheduled_category = get_category_for_today()
    if scheduled_category:
        print(f"📅 Using scheduled category for today: {scheduled_category}")
        return scheduled_category
    
    # Fallback: weighted random selection (for manual runs or schedule failures)
    print("⚠️ No schedule found, using weighted random selection")
    
    if content_type == 'evening_prime':
        rand = hash(str(datetime.now())) % 100
        if rand < 30:
            weighted_choice = 'disappearance'
        elif rand < 50:
            weighted_choice = 'dark_history'
        elif rand < 65:
            weighted_choice = 'disturbing_medical'
        elif rand < 75:
            weighted_choice = 'dark_experiments'
        else:
            weighted_choice = 'crime'
    elif content_type == 'weekend_binge':
        rand = hash(str(datetime.now())) % 100
        if rand < 30:
            weighted_choice = 'disappearance'
        elif rand < 50:
            weighted_choice = 'dark_history'
        elif rand < 65:
            weighted_choice = 'disturbing_medical'
        elif rand < 80:
            weighted_choice = 'dark_experiments'
        else:
            weighted_choice = 'phenomena'
    else:
        rand = hash(str(datetime.now())) % 100
        if rand < 30:
            weighted_choice = 'disappearance'
        elif rand < 50:
            weighted_choice = 'dark_history'
        elif rand < 65:
            weighted_choice = 'disturbing_medical'
        elif rand < 75:
            weighted_choice = 'dark_experiments'
        elif rand < 85:
            weighted_choice = 'crime'
        else:
            weighted_choice = 'phenomena'
    
    print(f"🎲 Weighted selection: {weighted_choice}")
    return weighted_choice


def get_content_type_guidance(content_type):
    """Get specific guidance for mystery content type"""
    guidance = {
        'evening_prime': f"""
EVENING PRIME FOCUS (7-9 PM):
- Target: Evening viewers unwinding, ready for dark intrigue
- Tone: Mysterious, film noir aesthetic, NOT educational documentary
- Examples: Dark historical events, medical mysteries, unethical experiments
- DURATION: {OPTIMAL_MIN_DURATION}-{OPTIMAL_MAX_DURATION}s TOTAL (includes 5s end hook)
- SCRIPT BODY: {SCRIPT_BODY_MIN}-{SCRIPT_BODY_MAX} words (35-45s, END HOOK AUTO-ADDED)
""",
        'weekend_binge': f"""
WEEKEND BINGE FOCUS (Sat/Sun 8-11 PM):
- Target: More time, want compelling dark stories
- Tone: Film noir thriller, mysterious, emotionally gripping
- Examples: Complex disappearances, disturbing medical cases, dark experiments
- DURATION: {OPTIMAL_MIN_DURATION}-{OPTIMAL_MAX_DURATION}s TOTAL (includes 5s end hook)
- SCRIPT BODY: {SCRIPT_BODY_MIN}-{SCRIPT_BODY_MAX} words (35-45s, END HOOK AUTO-ADDED)
""",
        'general': f"""
GENERAL MYSTERY FOCUS:
- Target: General dark mystery enthusiasts
- Tone: Mysterious, film noir, NOT educational
- Examples: Disappearances, dark history, medical mysteries
- DURATION: {OPTIMAL_MIN_DURATION}-{OPTIMAL_MAX_DURATION}s TOTAL (includes 5s end hook)
- SCRIPT BODY: {SCRIPT_BODY_MIN}-{SCRIPT_BODY_MAX} words (35-45s, END HOOK AUTO-ADDED)
"""
    }
    return guidance.get(content_type, guidance['general'])


def get_mystery_type_guidance(mystery_type):
    """
    🆕 v6.0: Get guidance for mystery category (EXPANDED with new categories)
    """
    types = {
        'disappearance': f"""
DISAPPEARANCE MYSTERIES (59.3% RETENTION - PROVEN STRENGTH):
- Focus: Vanishing without a trace
- Hook formula: "[Date], [Location]. [Person/Group] vanished. [Impossible detail]."
- Key elements: Last known location, search efforts, zero evidence
- Examples: Flight 19, DB Cooper, Maura Murray, Brandon Swanson
- Tone: Film noir mystery, NOT documentary
- MANDATORY: Use "Vanished" or "Disappeared" in title
- SCRIPT LENGTH: {SCRIPT_BODY_MIN}-{SCRIPT_BODY_MAX} words (end hook auto-added)
""",
        
        'dark_history': f"""
🆕 DARK HISTORY (65-75% TARGET RETENTION - MYSTERY-FIRST APPROACH):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🚨 CRITICAL: This is a MYSTERY category, NOT a history lesson!
Focus on the IMPOSSIBLE/UNEXPLAINED elements, NOT historical context.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CORE CONCEPT:
- Historical events where something IMPOSSIBLE/UNEXPLAINED happened
- Emphasis: MYSTERY first, history second
- Tone: Film noir thriller investigating a historical anomaly
- NOT: History Channel documentary about past events

MANDATORY OPENING STRUCTURE:
Formula: "In [Year], [IMPOSSIBLE EVENT happened]. [Witnesses]. [No explanation]."

✅ CORRECT EXAMPLES:
"1780. The sky turned black at noon. No eclipse. No explanation."
"1876. It rained meat from the sky. Kentucky. Chunks of flesh."
"1917. Girls started glowing in the dark. Their bones too."

❌ WRONG EXAMPLES (Historical narrative):
"In 1780, New Englanders were going about their daily lives when..." ❌
"The Radium Girls were factory workers who painted watch dials..." ❌
"During the 18th century, a mysterious event occurred..." ❌

STRUCTURE BREAKDOWN:

🎬 FIRST 15 WORDS (0-5 seconds):
THE IMPOSSIBLE EVENT - Start with what DEFIES EXPLANATION
"[Year]. [Impossible thing happened]. [Location/witnesses]. [Mystery word]."
MUST include mystery power words: vanished/glowed/turned/rained/died/appeared

🎬 WORDS 16-45 (5-20 seconds):
WHO/WHERE/WHEN context (NOW you can add historical details)
"[Historical group]. [Their activity]. [What they noticed]."
This is where historical context belongs - AFTER the mystery hook.

🎬 WORDS 46-75 (20-35 seconds):
THE TWIST - What made it IMPOSSIBLE/UNEXPLAINED
"But here's the impossible part. [Contradiction]. [Evidence that makes no sense]."

🎬 WORDS 76-94 (35-45 seconds):
UNRESOLVED MYSTERY - Dark outcome, no answers
"[Deaths/disappearances]. [Investigations failed]. [Mystery remains]."

EXAMPLES OF CORRECT MYSTERY-FIRST STRUCTURE:

✅ EXAMPLE 1: "The Day It Rained Meat"
First 15 words: "1876. Meat rained from the sky. Kentucky. Chunks of flesh."
Next 30 words: "Farmers watched. Pieces falling. Some the size of fists. Lab tests confirmed. Lung tissue. Muscle. From something."
Twist: "But here's the impossible part. No birds overhead. Clear skies. No aircraft existed yet."
Resolution: "Samples analyzed. Origin unknown. No explanation. Still unsolved."

✅ EXAMPLE 2: "The Girls Who Glowed"
First 15 words: "1917. Factory workers started glowing in the dark. Their hair. Skin. Breath."
Next 30 words: "The Radium Girls painted watch dials. Radium paint. Managers said it was safe. Told them to lick brushes."
Twist: "But here's the disturbing part. Their jaws started crumbling. Bones disintegrating. Anemia. Tumors."
Resolution: "Five died before the trial ended. Bodies so radioactive they glowed in coffins."

FORBIDDEN PATTERNS (CAUSE 36% RETENTION):
❌ "In [year], [people] were [activity]..." (Historical setup)
❌ "The [group] were [profession] who..." (Biographical intro)
❌ "During the [time period], [background]..." (Chronological narrative)
❌ Starting with person's background before the mystery
❌ Educational history documentary tone

MANDATORY REQUIREMENTS:
✅ First 15 words = THE IMPOSSIBLE EVENT (not background)
✅ Mystery power word in first 15 words (glowed/rained/turned/vanished/died)
✅ "But here's the [adjective] part" twist phrase
✅ Film noir mystery tone (NOT educational)
✅ Focus on UNEXPLAINED aspects throughout
✅ {SCRIPT_BODY_MIN}-{SCRIPT_BODY_MAX} words total

CONTENT GUIDELINES:
✅ ALLOW: Mysterious deaths, dark historical events, unexplained phenomena
✅ ALLOW: Medical mysteries, strange events, impossible occurrences
❌ AVOID: War crimes, graphic torture, child harm details, gore descriptions
❌ AVOID: Nazi experiments (too sensitive)
❌ AVOID: Graphic violence (focus on mystery, not brutality)

TOPIC SELECTION:
Focus on events with MYSTERIOUS elements:
- Unexplained natural phenomena (Dark Day, meat rain, strange lights)
- Medical mysteries with historical context (Radium Girls, Dancing Plague)
- Disappearances/deaths with impossible circumstances
- Events that defy scientific explanation

TITLE FORMATS:
"The [Event] That [Impossible Outcome]" - "The Day It Rained Meat"
"The [People] Who [Mysterious Fate]" - "The Girls Who Glowed In The Dark"
"The [Thing] That [Defied Explanation]" - "The Town That's Been Burning For 60 Years"

🚨 YOUR SCRIPT WILL BE REJECTED IF:
1. First 15 words don't contain impossible/mystery element
2. Starts with biographical/historical background
3. Uses educational documentary tone
4. Lacks "But..." twist phrase
5. Word count outside {SCRIPT_BODY_MIN}-{SCRIPT_BODY_MAX}

REMEMBER: You're writing a MYSTERY that happens to be historical, NOT a history lesson that happens to be mysterious.
The viewer should feel intrigue and impossibility, NOT like they're learning facts.
""",
        
        'disturbing_medical': f"""
🆕 DISTURBING MEDICAL MYSTERIES (70-80% TARGET RETENTION - NEW CATEGORY):
- Focus: Real medical conditions that defy explanation or horrify
- Hook formula: "[Person/Group] started [symptom]. Doctors couldn't explain it. [Outcome]."
- Key elements: Verified medical case, baffling symptoms, mysterious cause, dark outcome

🚨 TITLE REQUIREMENTS (CRITICAL - PREVENTS 0-VIEW ALGORITHM REJECTION):
- MUST include subtitle after colon for specificity
- MUST include: Condition name OR location OR specific case identifier
- Generic titles get 0 views from algorithm

✅ CORRECT EXAMPLES (High CTR):
"The Man Who Never Slept: Fatal Insomnia Mystery"
"The Girl Who Turned To Stone: The FOP Case"
"The Town That Went Blind: Minamata Disease Outbreak"
"The Condition That Prevents Sleep: FFI Mystery"

❌ WRONG EXAMPLES (0 views - too generic):
"The Man Who Couldn't Sleep" (no subtitle = algorithm rejection)
"The Condition That Turns You To Stone" (no specificity)
"The People Who Started Laughing" (no location/name)

SUBTITLE MUST ADD:
- Medical condition name (Fatal Insomnia, FOP, Kuru)
- OR location (Minamata, Pont-Saint-Esprit, Tanganyika)
- OR specific identifier (The 1518 Case, The Mystery Disease)

- Tone: BODY HORROR MYSTERY, NOT clinical textbook
- AVOID: Graphic surgical descriptions, gore, self-harm methods, eating disorder glorification
- ALLOW: Mysterious symptoms, baffling conditions, medical anomalies (focus on MYSTERY)
- SCRIPT LENGTH: {SCRIPT_BODY_MIN}-{SCRIPT_BODY_MAX} words (end hook auto-added)
""",
        
        'dark_experiments': f"""
🆕 DARK EXPERIMENTS (68-75% TARGET RETENTION - NEW CATEGORY):
- Focus: Real unethical/secret experiments with mysterious or horrifying outcomes
- Hook formula: "In [Year], [organization] conducted [experiment]. [Subjects]. [Result]."
- Key elements: Declassified files, secret research, ethical violations, mysterious outcomes

🚨 TITLE REQUIREMENTS (CRITICAL - PREVENTS 0-VIEW ALGORITHM REJECTION):
- MUST include subtitle after colon for specificity
- MUST include: Experiment name OR organization OR project code
- Generic titles get 0 views from algorithm

✅ CORRECT EXAMPLES (High CTR):
"The Experiment That Made People Vanish: MK-Ultra Files"
"The Sleep Study That Failed: Russian Experiment"
"The Prison Experiment Gone Wrong: Stanford 1971"
"The Secret Research They Buried: Project Artichoke"

❌ WRONG EXAMPLES (0 views - too generic):
"The Secret Study That Made People Vanish" (no specificity)
"The Experiment They Had To Shut Down" (no name)
"The Research That Went Wrong" (no organization)

SUBTITLE MUST ADD:
- Experiment name (MK-Ultra, Stanford Prison, Tuskegee)
- OR organization (CIA, Soviet, Unit 731)
- OR project code (Artichoke, Midnight Climax, Bluebird)

- Tone: CONSPIRACY THRILLER FILM NOIR, NOT documentary
- AVOID: Nazi experiments (too sensitive/banned), torture details, animal cruelty specifics
- ALLOW: CIA experiments, psychological studies, unethical research (focus on MYSTERY/COVER-UP)
- SCRIPT LENGTH: {SCRIPT_BODY_MIN}-{SCRIPT_BODY_MAX} words (end hook auto-added)
""",
        
        'crime': f"""
TRUE CRIME MYSTERIES (56.1% RETENTION - PROVEN):
- Focus: Unsolved murders, cold cases with mysterious elements
- Hook formula: "[Victim] was found [location]. Evidence vanished. [Mystery]."
- Key elements: Unsolved case, missing evidence, bizarre circumstances
- Examples: Zodiac Killer, Black Dahlia, Somerton Man
- Tone: Film noir crime thriller
- ETHICAL: Focus on mystery aspect, respect victims, avoid graphic violence
- SCRIPT LENGTH: {SCRIPT_BODY_MIN}-{SCRIPT_BODY_MAX} words (end hook auto-added)
""",
        
        'phenomena': f"""
UNEXPLAINED PHENOMENA (46.9% RETENTION - USE SPARINGLY):
- Focus: Strange occurrences, mysterious signals, impossible events
- Hook formula: "In [Year], [phenomenon] appeared. Scientists can't explain it."
- Key elements: Verified phenomena, multiple witnesses, no explanation
- Examples: The Hum, Wow Signal, Hessdalen Lights
- Tone: Mysterious, film noir
- SCRIPT LENGTH: {SCRIPT_BODY_MIN}-{SCRIPT_BODY_MAX} words (end hook auto-added)
""",
        
        'conspiracy': f"""
CONSPIRACY THEORIES (MERGED INTO dark_experiments):
- Use dark_experiments category instead for better retention
- Focus on declassified experiments and secret research
- SCRIPT LENGTH: {SCRIPT_BODY_MIN}-{SCRIPT_BODY_MAX} words (end hook auto-added)
""",
        
        'historical': f"""
HISTORICAL MYSTERIES (MERGED INTO dark_history):
- Use dark_history category for better framing
- Focus on mysterious/dark historical events
- SCRIPT LENGTH: {SCRIPT_BODY_MIN}-{SCRIPT_BODY_MAX} words (end hook auto-added)
"""
    }
    return types.get(mystery_type, types['disappearance'])


def build_mystery_prompt(content_type, priority, mystery_type, trends, history):
    """
    🚨 v6.0: Updated prompt with new categories + end hook instructions
    """
    
    previous_topics = [f"{t.get('topic', 'unknown')}: {t.get('title', '')}" for t in history['topics'][-20:]]
    previous_titles = [t.get('title', '') for t in history['topics'][-30:]]
    
    trending_topics = []
    trending_summaries = []
    
    if trends and trends.get('topics'):
        trending_topics = trends['topics'][:5]
        full_data = trends.get('full_data', [])
        
        if full_data:
            for item in full_data[:5]:
                viral_score = item.get('viral_score', 'N/A')
                trending_summaries.append(
                    f"• [{viral_score}] {item['topic_title']}\n"
                    f"  Hook: {item.get('story_hook', 'N/A')}\n"
                    f"  Mystery: {item.get('core_mystery', 'N/A')}"
                )
        else:
            trending_summaries = [f"• {t}" for t in trending_topics]
        
        print(f"🔍 Using {len(trending_topics)} trending mystery topics")
    
    if trending_topics:
        trending_mandate = f"""
⚠️ CRITICAL: YOU MUST CREATE A SCRIPT ABOUT ONE OF THESE REAL TRENDING MYSTERIES:

{chr(10).join(trending_summaries)}

YOU MUST USE ONE OF THESE REAL TRENDING TOPICS - NOT YOUR OWN INVENTION!
"""
    else:
        trending_mandate = "⚠️ No trending data - create original mystery content\n"
    
    content_type_guidance = get_content_type_guidance(content_type)
    mystery_type_guidance = get_mystery_type_guidance(mystery_type)
    
    prompt = f"""You are an expert mystery storyteller for MYTHICA REPORT.

🎯 CRITICAL REQUIREMENTS (v6.0 - DARK CONTENT + END HOOKS):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SCRIPT BODY TARGET: {SCRIPT_BODY_MIN}-{SCRIPT_BODY_MAX} WORDS (35-45 seconds)
END HOOK: AUTO-ADDED AFTER YOUR SCRIPT (5 seconds, DO NOT INCLUDE)
TOTAL VIDEO: {OPTIMAL_MIN_DURATION}-{OPTIMAL_MAX_DURATION} seconds (your script + auto end hook)

TTS Speed: {WORDS_PER_SECOND} words/second
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🚨 PROVEN RETENTION DATA (REAL EXAMPLES):

✅ NATASHA RYAN (72.9% RETENTION - THE GOLD STANDARD):
Title: "The Girl Who Vanished: From Her Own Backyard"
Hook: "In 1998, a teen vanished. Five years later, she was found alive."
Script: "In 1998, a teen vanished. Five years later, she was found alive.

14-year-old Natasha Ryan disappeared from Rockhampton, Australia. Clothes found near a river. Search teams found nothing.

But five years later, police raided a house during a murder investigation. They found Natasha. Alive. Hiding in a closet.

She'd been living there the entire time. Just blocks from her family. Her boyfriend kept her hidden. His parents never knew."

WHY IT WORKED:
- First 15 words: "Teen vanished. Found alive." ✅ IMMEDIATE REVEAL
- NO backstory before the reveal ✅
- Uses "But five years later..." ✅ TWIST PHRASE
- 64.1% average retention ✅

❌ LEAH ROBERTS (18.75% RETENTION - THE FAILURE PATTERN):
Script opening: "20-year-old Leah was road-tripping Australia. Her white Subaru was found..."

WHY IT FAILED:
- Uses "was road-tripping" ❌ ACTIVITY-BASED BACKSTORY
- Starts with age + name ❌ CHRONOLOGICAL SETUP
- Retention at 0:09: 23% (CATASTROPHIC DROP) ❌

🚨 MANDATORY STRUCTURE (FOLLOW NATASHA, NOT LEAH):

CONTEXT:
- Content type: {content_type}
- Mystery type: {mystery_type}
- Priority: {priority}

PREVIOUSLY COVERED (DO NOT REPEAT):
{chr(10).join(f"  • {t}" for t in previous_topics[-15:]) if previous_topics else '  None yet'}

{trending_mandate}

{content_type_guidance}

{mystery_type_guidance}

🎬 SCRIPT STRUCTURE (INVERTED PYRAMID - 35-45s SCRIPT BODY):

🚨 SECTION 1: IMMEDIATE REVEAL (0-10 seconds, 15-20 words):
FORMULA: "[Date/Year], [Event]. [Outcome/Mystery]."
✅ CORRECT: "In 1917, girls started glowing. Doctors couldn't explain it."
✅ CORRECT: "December 1872. A ship found drifting. Crew vanished."
❌ WRONG: "The Radium Girls were factory workers who painted watch dials..."

🎬 SECTION 2: CONTEXT (10-20 seconds, 25-35 words):
NOW add WHO, WHAT, WHERE - focus on EVIDENCE/MYSTERY, not backstory.
✅ CORRECT: "Factory workers painting watch dials. Radium paint. Told it was safe."
❌ WRONG: "They were young women who needed money and worked long hours..."

🎬 SECTION 3: THE TWIST (20-35 seconds, 20-30 words):
FORMULA: "But [here's where it gets strange/the impossible part]..."
MANDATORY: Must include a "But" twist phrase.

🎬 SECTION 4: RESOLUTION (35-45 seconds, 15-20 words):
The final reveal or deepening of the mystery.

🚨 DO NOT WRITE AN END HOOK - IT WILL BE AUTO-GENERATED
Your script should END at the resolution/mystery deepening.
The system will automatically add: "[Mystery ending]. [Category promise]."

TOTAL SCRIPT BODY: {SCRIPT_BODY_MIN}-{SCRIPT_BODY_MAX} WORDS (end hook added separately)

MANDATORY REQUIREMENTS:
✅ REVEAL THE MYSTERY IN FIRST 15 WORDS
✅ NO backstory before the reveal (blocks 0:09 cliff)
✅ Use mystery power words in first 15 words (vanished/disappeared/died/glowed/turned)
✅ Include "But" twist phrase
✅ SHORT PUNCHY SENTENCES (6-10 words each)
✅ NO bullet points - paragraph breaks only (\\n\\n)
✅ STAY BETWEEN {SCRIPT_BODY_MIN}-{SCRIPT_BODY_MAX} WORDS
✅ MYSTERIOUS TONE (film noir), NOT educational documentary

AVOID (RETENTION KILLERS):
❌ "was road-tripping/traveling/working/studying" in first 100 words
❌ "X-year-old [Name] was..." pattern
❌ Starting with activity/backstory
❌ Excessive details before hook
❌ Going under {SCRIPT_BODY_MIN} or over {SCRIPT_BODY_MAX} words
❌ Educational lecture tone (use mysterious film noir instead)

OUTPUT FORMAT (JSON ONLY):
{{
  "title": "The [X] Who/That [Y]",
  "topic": "mystery",
  "hook": "[8-12 words max with mystery power word]",
  "script": "[SCRIPT BODY ONLY - {SCRIPT_BODY_MIN} to {SCRIPT_BODY_MAX} words - DO NOT INCLUDE END HOOK]",
  "cta": "[Question under 12 words]",
  "hashtags": ["#mystery", "#unsolved", "#shorts"],
  "description": "[2 sentences for YouTube]",
  "key_phrase": "[3-5 CAPS words for thumbnail]",
  "mystery_category": "{mystery_type}",
  "visual_prompts": [
    "Film noir: [scene 1], dramatic shadows, noir aesthetic",
    "Film noir: [scene 2], vintage documentary style",
    "Film noir: [scene 3], dark and ominous",
    "Film noir: [scene 4], final mysterious reveal"
  ]
}}

🚨 YOUR SCRIPT WILL BE AUTOMATICALLY REJECTED IF:
1. First 50 words lack mystery reveal words
2. First 100 words contain "was [activity]" backstory patterns
3. Word count < {SCRIPT_BODY_MIN} or > {SCRIPT_BODY_MAX}
4. Includes end hook (it's auto-added, don't write it)

Generate the mystery story NOW. 
Target {SCRIPT_BODY_MIN}-{SCRIPT_BODY_MAX} words for SCRIPT BODY ONLY.
REVEAL THE MYSTERY IN THE FIRST 15 WORDS!
END HOOK WILL BE AUTO-ADDED - DO NOT INCLUDE IT IN YOUR SCRIPT!
"""

    return prompt


def get_fallback_script(content_type, mystery_type):
    """
    Fallback scripts with end hooks
    v6.0: Includes new dark content categories
    """
    
    fallback_scripts = {
        'dark_history': {
            'title': "The Girls Who Glowed In The Dark",
            'hook': "In 1917, girls started glowing. Doctors couldn't explain it.",
            'script': """In 1917, factory workers started glowing in the dark. Their hair. Their skin. Even their breath.

The Radium Girls painted watch dials. Radium paint. Managers told them it was safe. Told them to lick the brushes for precision.

But here's the disturbing part. Their jaws started crumbling. Bones disintegrating. Anemia. Tumors. Necrosis.

Company doctors blamed it on syphilis. Tried to silence them. The girls sued.

Five died before the trial ended. Their bodies were so radioactive they glowed in their coffins.

The case changed labor laws forever.""",
            'key_phrase': "RADIUM GIRLS",
            'mystery_category': 'dark_history'
        },
        
        'disturbing_medical': {
            'title': "The Man Who Couldn't Sleep",
            'hook': "He stopped sleeping. Never slept again. Dead in 18 months.",
            'script': """In 1984, a man stopped sleeping. Completely. No sleep. Ever.

Doctors ran tests. Sleep deprivation. Hallucinations. Panic attacks. Nothing worked. He couldn't sleep no matter what they tried.

But here's the terrifying part. It was genetic. Fatal Familial Insomnia. His brain was destroying itself.

First, insomnia. Then paranoia. Then complete inability to distinguish reality. Rapid weight loss. Dementia.

Eighteen months after it started, he died. Awake. Aware. Unable to escape.

To this day, no cure exists. No treatment. No survivors.""",
            'key_phrase': "FATAL INSOMNIA",
            'mystery_category': 'disturbing_medical'
        },
        
        'dark_experiments': {
            'title': "The Sleep Experiment They Had To Shut Down",
            'hook': "Five subjects. No sleep. Thirty days. None survived psychologically.",
            'script': """In the 1940s, Soviet researchers tested sleep deprivation. Five political prisoners. Sealed chamber. Thirty days. Stimulant gas pumped in.

Day five, they stopped talking. Day eleven, screaming started. Day fifteen, they begged not to sleep ever again.

But here's the disturbing part. When researchers opened the chamber, the subjects had mutilated themselves. Laughing. Begging to stay awake.

One tore out his own vocal cords. Another ate his own flesh. When doctors tried to sedate them, they fought back with inhuman strength.

The experiment was classified immediately. Files sealed for decades.

The subjects? All declared criminally insane. Kept in isolation until death.""",
            'key_phrase': "SLEEP EXPERIMENT",
            'mystery_category': 'dark_experiments'
        },
        
        'disappearance': {
            'title': "Flight 19: Five Planes That Vanished",
            'hook': "December 5th, 1945. Five planes vanished without trace.",
            'script': """December 5th, 1945. Five torpedo bombers vanished. Fourteen experienced crew. Gone.

But here's the impossible part. Two hours into a routine training flight, all compasses failed. Simultaneously. Lieutenant Taylor radioed: 'We can't find west. Everything looks wrong.'

The Navy launched the biggest search in history. Two hundred forty thousand square miles. Three hundred aircraft. Five days straight.

Zero debris. No oil slicks. Nothing. Five massive planes. Vanished.

The most terrifying part? The rescue plane vanished too. Same night. Thirteen more crew.

To this day, twenty-seven men and six aircraft. No evidence.""",
            'key_phrase': "FLIGHT 19",
            'mystery_category': 'disappearance'
        }
    }
    
    # Select appropriate fallback
    if mystery_type in fallback_scripts:
        selected = fallback_scripts[mystery_type]
    else:
        selected = fallback_scripts['disappearance']
    
    print(f"📋 Using fallback script: {selected['title']}")
    
    word_count = len(selected['script'].split())
    estimated_duration = word_count / WORDS_PER_SECOND
    
    # Generate end hook for fallback
    end_hook = generate_end_hook(selected['mystery_category'])
    
    # Combine script + end hook
    full_script = f"{selected['script']}\n\n{end_hook}"
    total_word_count = len(full_script.split())
    total_duration = total_word_count / WORDS_PER_SECOND
    
    print(f"   ✅ Fallback script body: {word_count} words ({estimated_duration:.1f}s)")
    print(f"   ✅ With end hook: {total_word_count} words ({total_duration:.1f}s total)")
    
    return {
        'title': selected['title'],
        'topic': 'mystery',
        'hook': selected['hook'],
        'script': full_script,  # Includes end hook
        'key_phrase': selected['key_phrase'],
        'mystery_category': selected['mystery_category'],
        'cta': 'What do you think happened?',
        'hashtags': ['#mystery', '#unsolved', '#vanished', '#shorts'],
        'description': f"{selected['title']} - A dark mystery that defies explanation. #mystery #shorts",
        'visual_prompts': [
            'Film noir: vintage historical photograph, dark moody lighting, noir aesthetic',
            'Film noir: evidence scene, shadowy mysterious, noir photography',
            'Film noir: dramatic reveal, noir lighting, ominous mood',
            'Film noir: final mysterious image, dark ominous, noir aesthetic'
        ],
        'content_type': content_type,
        'priority': 'fallback',
        'mystery_type': selected['mystery_category'],
        'word_count': total_word_count,
        'estimated_duration': total_duration,
        'has_end_hook': True,
        'is_fallback': True,
        'optimization_version': '6.0_dark_content_end_hooks',
        'generated_at': datetime.now().isoformat(),
        'niche': 'mystery'
    }


def generate_mystery_script():
    """Main script generation function (v6.0.8 - ITERATIVE RETRY WITH FEEDBACK)"""
    
    content_type = os.getenv('CONTENT_TYPE', 'evening_prime')
    priority = os.getenv('PRIORITY', 'medium')
    user_mystery_type = os.getenv('MYSTERY_TYPE', 'auto')
    
    print(f"\n{'='*70}")
    print(f"🔍 GENERATING MYTHICA REPORT SCRIPT v6.0.8 (ITERATIVE RETRY)")
    print(f"{'='*70}")
    print(f"📍 Content Type: {content_type}")
    print(f"⭐ Priority: {priority}")
    print(f"🎭 User Mystery Type: {user_mystery_type}")
    print(f"🎯 TARGET: 64.1% retention + subscriber conversion via end hooks")
    
    history = load_history()
    trends = load_trending()
    
    if trends:
        print(f"✅ Loaded trending data from {trends.get('source', 'unknown')}")
    else:
        print("⚠️ No trending data available")
    
    # Select mystery type (schedule-aware)
    mystery_type = select_weighted_mystery_type(content_type, user_mystery_type)
    print(f"🎯 Final mystery type: {mystery_type}")
    
    # 🆕 v6.0.9: Filter trending topics by category
    if trends:
        print(f"\n🔍 FILTERING TRENDING TOPICS FOR CATEGORY: {mystery_type}")
        filtered_trends = filter_trending_by_category(trends, mystery_type)
        
        if filtered_trends:
            trends = filtered_trends
            print(f"✅ Using {len(filtered_trends.get('topics', []))} category-matched trending topics")
        else:
            trends = None
            print(f"⚠️ No trending topics match '{mystery_type}' - using pure category generation")
    
    # Build base prompt
    base_prompt = build_mystery_prompt(content_type, priority, mystery_type, trends, history)
    
    max_attempts = 5
    attempt = 0
    data = None
    previous_errors = []  # Track errors for iterative feedback
    
    while attempt < max_attempts:
        try:
            attempt += 1
            print(f"\n🔍 Generation attempt {attempt}/{max_attempts}...")
            
            # v6.0.8: Add error feedback to prompt for attempts 2+
            if attempt > 1 and previous_errors:
                last_error = previous_errors[-1]
                correction_prompt = f"""

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🚨 PREVIOUS ATTEMPT #{attempt-1} FAILED - CORRECTION REQUIRED:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ERROR: {last_error}

🔧 SPECIFIC CORRECTIONS NEEDED:

"""
                if 'backstory' in last_error.lower():
                    correction_prompt += """
1. DO NOT start with biographical information
2. DO NOT use "was [activity]" in first 30 words
3. START with the IMPOSSIBLE/MYSTERIOUS event
4. Example: "1917. Girls started glowing." NOT "The Radium Girls were factory workers who..."
"""
                
                if 'word count' in last_error.lower() or 'short' in last_error.lower():
                    correction_prompt += f"""
1. Your script body must be {SCRIPT_BODY_MIN}-{SCRIPT_BODY_MAX} words
2. Add more mystery details, evidence, witnesses
3. Expand the twist section ("But here's the [adjective] part...")
4. Add more specific facts and unexplained elements
"""
                
                if 'power word' in last_error.lower() or 'reveal' in last_error.lower():
                    correction_prompt += """
1. First 15 words MUST include: vanished/disappeared/died/glowed/turned/rained
2. Start with the OUTCOME: "In [year], [mystery happened]."
3. NOT the setup: "In [year], [people] were [doing activity]..."
"""
                
                if 'trending' in last_error.lower():
                    correction_prompt += """
1. You MUST use one of the provided trending topics
2. Don't invent your own mystery
3. Research the real case and focus on mysterious elements
"""
                
                correction_prompt += f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

NOW REGENERATE THE SCRIPT WITH THESE CORRECTIONS APPLIED.
This is attempt #{attempt} of {max_attempts}.
"""
                
                prompt = base_prompt + correction_prompt
                print(f"   🔄 Added correction feedback based on: {last_error[:100]}...")
            else:
                prompt = base_prompt
            
            raw_text = generate_script_with_retry(prompt)
            print(f"📝 Received response ({len(raw_text)} chars)")
            
            json_text = extract_json_from_response(raw_text)
            data = json.loads(json_text)
            
            # ✅ Validate script body (before adding end hook)
            validate_script_data(data)
            
            # 🎬 Generate and append end hook
            end_hook = generate_end_hook(mystery_type)
            original_script = data["script"]
            data["script"] = f"{original_script}\n\n{end_hook}"
            
            # Recalculate totals with end hook
            total_word_count = len(data["script"].split())
            total_duration = total_word_count / WORDS_PER_SECOND
            
            print(f"\n📊 FINAL SCRIPT ANALYSIS (with end hook):")
            print(f"   Script body: {data['word_count']} words")
            print(f"   End hook: {len(end_hook.split())} words")
            print(f"   Total: {total_word_count} words ({total_duration:.1f}s)")
            
            # Update metadata
            data["topic"] = "mystery"
            data["content_type"] = content_type
            data["priority"] = priority
            data["mystery_category"] = mystery_type
            data["optimization_version"] = "6.0_dark_content_end_hooks"
            data["generated_at"] = datetime.now().isoformat()
            data["niche"] = "mystery"
            data["has_end_hook"] = True
            data["end_hook_text"] = end_hook
            data["total_word_count"] = total_word_count
            data["total_duration"] = total_duration
            
            # Clean text
            data["title"] = clean_script_text(data["title"])
            data["hook"] = clean_script_text(data["hook"])
            data["cta"] = clean_script_text(data["cta"])
            data["script"] = clean_script_text(data["script"])
            
            # Ensure hashtags
            if "hashtags" not in data or not data["hashtags"]:
                data["hashtags"] = ["#mystery", "#unsolved", "#shorts"]
            
            if "description" not in data:
                data["description"] = f"{data['title']} - {data['hook']} #mystery #shorts"
            
            if "visual_prompts" not in data or len(data["visual_prompts"]) < 4:
                data["visual_prompts"] = [
                    "Film noir: mysterious scene, dark moody lighting, noir aesthetic",
                    "Film noir: evidence scene, shadowy mysterious, noir photography",
                    "Film noir: dramatic reveal, noir lighting, ominous mood",
                    "Film noir: final image, dramatic shadows, noir aesthetic"
                ]
            
            if "key_phrase" not in data:
                if ':' in data['title']:
                    key_phrase = data['title'].split(':')[0].strip()
                else:
                    words = data['title'].split()[:4]
                    key_phrase = ' '.join(words)
                data["key_phrase"] = key_phrase.upper()
            
            # Validate trending topic usage
            if trends and trends.get('topics'):
                if not validate_script_uses_trending_topic(data, trends['topics']):
                    raise ValueError("Script doesn't use trending topics - regenerating...")
            
            # Check for duplicates
            content_hash = get_content_hash(data)
            if content_hash in [t.get('hash') for t in history['topics']]:
                print("⚠️ Generated duplicate content, regenerating...")
                raise ValueError("Duplicate content detected")
            
            # Check similarity (v6.0.10 - entity-based + fixed time decay)
            previous_titles = [t.get('title', '') for t in history['topics']]
            if is_similar_topic(data['title'], previous_titles):
                print("⚠️ Topic too similar, regenerating...")
                raise ValueError("Duplicate topic detected (entity or string match)")
            
            # Save to history
            save_to_history(data['topic'], content_hash, data['title'], data)
            
            print(f"\n✅ SCRIPT GENERATED & VALIDATED")
            print(f"   Title: {data['title']}")
            print(f"   Category: {mystery_type}")
            print(f"   Total duration: ~{total_duration:.1f}s (includes 5s end hook)")
            print(f"   End hook: '{end_hook}'")
            print(f"   🎯 Optimization: v6.0 (dark content + end hooks)")
            
            break
            
        except json.JSONDecodeError as e:
            error_msg = f"JSON parse error: {str(e)}"
            previous_errors.append(error_msg)
            print(f"❌ Attempt {attempt} failed: {error_msg}")
            if attempt < max_attempts:
                import time
                time.sleep(2**attempt)
        
        except ValueError as e:
            error_msg = str(e)
            previous_errors.append(error_msg)
            print(f"❌ Attempt {attempt} failed: {error_msg}")
            if attempt < max_attempts:
                print(f"   🔄 Will retry with correction feedback...")
                import time
                time.sleep(2**attempt)
        
        except Exception as e:
            print(f"❌ Attempt {attempt} failed: {type(e).__name__} - {e}")
            if attempt < max_attempts:
                import time
                time.sleep(2**attempt)
        
        if attempt >= max_attempts:
            print("\n⚠️ Max attempts reached")
            
            # v6.0.8: Try simplified prompt for dark_history before fallback
            if mystery_type == 'dark_history' and not any('simplified' in err for err in previous_errors):
                print("   🔧 Attempting simplified dark_history prompt (last chance)...")
                try:
                    simplified_prompt = f"""Create a {SCRIPT_BODY_MIN}-{SCRIPT_BODY_MAX} word mystery script about a dark historical event.

MANDATORY FIRST 15 WORDS:
"[Year]. [Impossible event]. [Location]. [Outcome]."

Example: "1780. The sky turned black at noon. New England. No eclipse."

Then continue with:
- Who witnessed it (20 words)
- What made it impossible (30 words)  
- The dark outcome (20 words)

FORBIDDEN:
- Starting with person's background
- "was [activity]" before mystery reveal
- Educational history tone

Use film noir mysterious tone. Focus on the UNEXPLAINED.

Output JSON with: title, topic, hook, script, cta, hashtags, description, key_phrase, mystery_category, visual_prompts
"""
                    
                    previous_errors.append('simplified attempt')
                    raw_text = generate_script_with_retry(simplified_prompt)
                    json_text = extract_json_from_response(raw_text)
                    data = json.loads(json_text)
                    
                    # Quick validation
                    validate_script_data(data)
                    
                    # Add end hook
                    end_hook = generate_end_hook(mystery_type)
                    data["script"] = f"{data['script']}\n\n{end_hook}"
                    
                    # Success with simplified prompt!
                    print("   ✅ Simplified prompt succeeded!")
                    
                except Exception as e:
                    print(f"   ❌ Simplified prompt also failed: {e}")
                    print("   📋 Using fallback script")
                    data = get_fallback_script(content_type, mystery_type)
                    fallback_hash = get_content_hash(data)
                    save_to_history(data['topic'], fallback_hash, data['title'], data)
            else:
                # Use fallback for non-dark_history or if simplified already tried
                print("   📋 Using fallback script")
                data = get_fallback_script(content_type, mystery_type)
                fallback_hash = get_content_hash(data)
                save_to_history(data['topic'], fallback_hash, data['title'], data)
    
    # Save final script
    script_path = os.path.join(TMP, "script.json")
    with open(script_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Saved to {script_path}")
    
    print(f"\n{'='*70}")
    print(f"✅ MYTHICA REPORT SCRIPT READY v6.0")
    print(f"{'='*70}")
    print(f"Script body: {data.get('word_count', 0)} words")
    print(f"Total (with end hook): {data.get('total_word_count', 0)} words")
    print(f"Duration: {data.get('total_duration', 0):.1f}s")
    print(f"Category: {data.get('mystery_category', 'unknown')}")
    print(f"Has end hook: {data.get('has_end_hook', False)}")
    print(f"Status: ✅ READY FOR PRODUCTION")
    
    return data


if __name__ == '__main__':
    try:
        generate_mystery_script()
        print("\n✅ Script generation complete!")
    except Exception as e:
        print(f"\n❌ FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        exit(1)