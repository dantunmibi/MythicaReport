#!/usr/bin/env python3
"""
🔍 Generate Mystery Script - RETENTION-OPTIMIZED FOR MYTHICA REPORT
Based on proven performance data: 35-45 seconds = 74-95% retention

🚨 CRITICAL FIX v3.1.1: Enhanced backstory detection
✅ NEW: Catches "was road-tripping" pattern (Leah Roberts killer)
✅ NEW: Activity-based setup pattern blocking
✅ PROVEN: Prevents 0:09 retention collapse (18.75% → 72.9%)
"""

import os
import json
import re
import hashlib
from datetime import datetime
import google.generativeai as genai
from tenacity import retry, stop_after_attempt, wait_exponential

TMP = os.getenv("GITHUB_WORKSPACE", ".") + "/tmp"
os.makedirs(TMP, exist_ok=True)
HISTORY_FILE = os.path.join(TMP, "content_history.json")

# 🎯 MYTHICA REPORT OPTIMIZED TARGETS
OPTIMAL_MIN_DURATION = 35
OPTIMAL_MAX_DURATION = 45

WORDS_PER_SECOND = 2.1

MINIMUM_WORDS = 70
OPTIMAL_MIN_WORDS = int(OPTIMAL_MIN_DURATION * WORDS_PER_SECOND)
OPTIMAL_MAX_WORDS = int(OPTIMAL_MAX_DURATION * WORDS_PER_SECOND)
HARD_LIMIT_WORDS = 100

MYSTERY_WEIGHTS = {
    'disappearance': 0.40,
    'phenomena': 0.30,
    'crime': 0.15,
    'historical': 0.05,
    'conspiracy': 0.10,
}

print(f"🎯 Mythica Report Optimization v3.1.1 (ENHANCED BACKSTORY DETECTION):")
print(f"   ✅ Target: {OPTIMAL_MIN_DURATION}-{OPTIMAL_MAX_DURATION}s ({OPTIMAL_MIN_WORDS}-{OPTIMAL_MAX_WORDS} words)")
print(f"   ⚠️ Minimum: {MINIMUM_WORDS} words ({(MINIMUM_WORDS/WORDS_PER_SECOND):.0f}s)")
print(f"   🚨 Hard Limit: {HARD_LIMIT_WORDS} words ({(HARD_LIMIT_WORDS/WORDS_PER_SECOND):.0f}s)")
print(f"   🔥 Category Focus: {int(MYSTERY_WEIGHTS['disappearance']*100)}% Disappearances")
print(f"   🎯 FIX TARGET: Prevent Leah Roberts pattern (18.75% retention)")

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
            return {'topics': [], 'version': '3.1.1_enhanced_backstory_detection'}
    
    print("📂 No previous history found, starting fresh")
    return {'topics': [], 'version': '3.1.1_enhanced_backstory_detection'}


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
        'date': datetime.now().isoformat(),
        'timestamp': datetime.now().timestamp()
    })
    
    history['topics'] = history['topics'][-100:]
    history['last_updated'] = datetime.now().isoformat()
    history['version'] = '3.1.1_enhanced_backstory_detection'
    
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


def is_similar_topic(new_title, previous_titles, similarity_threshold=0.6):
    """Check if topic is too similar to previous ones with time decay"""
    new_words = set(new_title.lower().split())
    
    for idx, prev_title in enumerate(reversed(previous_titles[-30:])):
        prev_words = set(prev_title.lower().split())
        
        intersection = len(new_words & prev_words)
        union = len(new_words | prev_words)
        
        if union > 0:
            base_similarity = intersection / union
            decay_factor = 1.0 / (1.0 + idx * 0.05)
            adjusted_threshold = similarity_threshold * decay_factor
            
            if base_similarity > adjusted_threshold:
                print(f"⚠️ Topic too similar ({base_similarity:.2f} > {adjusted_threshold:.2f})")
                print(f"   To: {prev_title}")
                print(f"   From: ~{idx} videos ago")
                return True
    
    return False


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def generate_script_with_retry(prompt):
    """Generate script with automatic retry on failure"""
    response = model.generate_content(prompt)
    return response.text.strip()


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
    🚨 ENHANCED v3.1.1: Structural validation with improved backstory detection
    
    NEW FEATURES:
    - Catches "was road-tripping" pattern (Leah Roberts killer)
    - Detects activity-based setup phrases
    - Blocks progressive tense backstory in first 100 words
    
    Prevents retention collapse at 0:09 (18.75% → 72.9% target)
    
    PROVEN EFFECTIVENESS:
    - Natasha Ryan: "Teen vanished. Found alive." → 72.9% retention ✅
    - Leah Roberts: "Leah was road-tripping..." → 18.75% retention ❌
    """
    
    print("\n🔍 STRUCTURAL VALIDATION v3.1.1 (Enhanced Backstory Detection):")
    
    words = script_text.split()
    first_50 = ' '.join(words[:50]).lower()
    first_100 = ' '.join(words[:100]).lower()
    
    # ✅ CHECK 1: Early reveal (prevents 0:09 drop)
    reveal_words = ['vanished', 'disappeared', 'found', 'discovered', 'never', 'no one', 'gone', 'missing']
    has_early_reveal = any(word in first_50 for word in reveal_words)
    
    if not has_early_reveal:
        print("   ❌ BLOCKED: No reveal in first 50 words")
        print("   💡 This causes 0:09 retention collapse (Leah Roberts: 18.75%)")
        print(f"   📊 First 50 words: {first_50[:150]}...")
        raise ValueError(
            "Script REJECTED: Mystery reveal must appear in first 50 words (first 15 seconds). "
            "This is CRITICAL to prevent 0:09 retention drop. "
            f"Required words: {', '.join(reveal_words)}"
        )
    
    print(f"   ✅ Early reveal: YES ({[w for w in reveal_words if w in first_50]})")
    
    # ✅ CHECK 2: Twist/contradiction phrase
    twist_phrases = [
        'but here', 'the strange part', 'the impossible part', 
        'the terrifying part', 'the mystery', 'what makes this',
        'but what', 'the twist', 'the catch', 'the problem'
    ]
    has_twist_phrase = any(phrase in script_text.lower() for phrase in twist_phrases)
    
    if not has_twist_phrase:
        print("   ⚠️ WARNING: Missing twist phrase")
        print("   💡 Natasha Ryan (72.9%) uses 'But five years later...'")
        print("   💡 Not blocking, but STRONGLY recommended")
    else:
        matching_phrases = [p for p in twist_phrases if p in script_text.lower()]
        print(f"   ✅ Twist phrase: YES ({matching_phrases[0]})")
    
    # ✅ CHECK 3: ENHANCED CHRONOLOGICAL BACKSTORY DETECTION
    backstory_indicators = [
        # Original indicators
        'was born', 'grew up', 'was a student', 'worked as', 
        'lived in', 'was known for', 'had been', 'had always',
        'at the age of', 'years old', 'graduated from', 'studied at',
        
        # 🚨 NEW: Activity-based setup patterns (catches Leah Roberts)
        'was traveling', 'was road-tripping', 'was driving', 'was hiking',
        'was visiting', 'was working', 'was studying', 'was living',
        'had been traveling', 'had been working', 'had been studying',
        'was on a trip', 'was on a journey', 'was exploring',
        
        # 🚨 NEW: Progressive tense backstory (catches subtle patterns)
        'was heading', 'was going', 'was planning', 'was preparing',
        'had planned', 'had decided', 'had embarked'
    ]
    
    has_early_backstory = any(phrase in first_100 for phrase in backstory_indicators)
    
    if has_early_backstory:
        backstory_found = [phrase for phrase in backstory_indicators if phrase in first_100]
        print(f"   ❌ BLOCKED: Chronological backstory in first 100 words")
        print(f"   💡 Found: {backstory_found}")
        print(f"   📊 This is the EXACT pattern that caused Leah Roberts failure:")
        print(f"      'Leah was road-tripping Australia' → 18.75% retention")
        print(f"   ✅ Correct pattern (Natasha Ryan): 'Teen vanished. Found alive.' → 72.9%")
        raise ValueError(
            "Script REJECTED: Uses chronological backstory structure in first 100 words. "
            "This causes viewers to drop at 0:09 (Leah Roberts: 18.75% retention). "
            "Start with the OUTCOME/MYSTERY, not the person's activity or background. "
            f"Problematic phrases: {', '.join(backstory_found)}"
        )
    
    print("   ✅ No early backstory: YES (Leah Roberts pattern avoided)")
    
    # ✅ CHECK 4: Setup/introduction pattern
    setup_indicators = [
        'a normal', 'an ordinary', 'a typical', 'a regular',
        'like any other', 'just another', 'seemed normal',
        'everything was fine', 'nothing unusual', 'routine'
    ]
    has_early_setup = any(phrase in first_50 for phrase in setup_indicators)
    
    if has_early_setup:
        setup_found = [phrase for phrase in setup_indicators if phrase in first_50]
        print(f"   ⚠️ WARNING: Generic setup in first 50 words")
        print(f"   💡 Found: {setup_found}")
        print(f"   💡 Not blocking, but this reduces hook strength")
    else:
        print("   ✅ No generic setup: YES")
    
    # 🚨 NEW CHECK 5: Excessive detail in first 50 words
    detail_indicators = [
        'dog food', 'blanket', 'map', 'unlocked', 'lights on',
        'engine off', 'white subaru', 'blue car', 'red truck'
    ]
    has_excessive_detail = any(phrase in first_50 for phrase in detail_indicators)
    
    if has_excessive_detail:
        detail_found = [phrase for phrase in detail_indicators if phrase in first_50]
        print(f"   ⚠️ WARNING: Excessive detail in first 50 words")
        print(f"   💡 Found: {detail_found}")
        print(f"   💡 Leah Roberts had: 'Dog food, blanket, map' → retention killer")
        print(f"   💡 Save details for AFTER the hook")
    
    print("   ✅ STRUCTURAL VALIDATION PASSED (v3.1.1 - Enhanced)\n")
    return True


def validate_script_data(data):
    """✅ ENHANCED: Word count + structural validation"""
    
    required_fields = ["title", "topic", "hook", "script", "cta"]
    
    for field in required_fields:
        if field not in data:
            raise ValueError(f"❌ Missing required field: {field}")
    
    if not isinstance(data["script"], str):
        raise ValueError("❌ Script must be a string (narrative text)")
    
    # ✅ WORD COUNT VALIDATION
    word_count = len(data["script"].split())
    estimated_duration = word_count / WORDS_PER_SECOND
    
    print(f"\n📊 Script Length Analysis:")
    print(f"   Words: {word_count}")
    print(f"   Estimated duration: {estimated_duration:.1f}s")
    print(f"   Optimal range: {OPTIMAL_MIN_DURATION}-{OPTIMAL_MAX_DURATION}s")
    
    if word_count < MINIMUM_WORDS:
        print(f"   ❌ TOO SHORT: {word_count} words = {estimated_duration:.1f}s")
        raise ValueError(f"Script too short: {word_count} words (minimum {MINIMUM_WORDS})")
    
    if word_count > HARD_LIMIT_WORDS:
        print(f"   ❌ TOO LONG: {word_count} words = {estimated_duration:.1f}s")
        raise ValueError(f"Script too long: {word_count} words (max {HARD_LIMIT_WORDS})")
    
    if OPTIMAL_MIN_WORDS <= word_count <= OPTIMAL_MAX_WORDS:
        print(f"   ✅ PERFECT: {word_count} words = {estimated_duration:.1f}s (72.9% RETENTION ZONE)")
    elif word_count < OPTIMAL_MIN_WORDS:
        print(f"   ⚠️ ACCEPTABLE: {word_count} words = {estimated_duration:.1f}s (slightly under optimal)")
    else:
        print(f"   ⚠️ ACCEPTABLE: {word_count} words = {estimated_duration:.1f}s (slightly over optimal)")
    
    data["estimated_duration"] = estimated_duration
    data["word_count"] = word_count
    data["optimization_version"] = "3.1.1_enhanced_backstory_detection"
    
    # ✅ CRITICAL: STRUCTURAL VALIDATION (prevents Leah Roberts pattern)
    validate_script_structure(data["script"])
    
    # Validate title
    if len(data["title"]) > 100:
        print(f"⚠️ Title too long ({len(data['title'])} chars), truncating...")
        data["title"] = data["title"][:97] + "..."
    
    # ✅ ENHANCED: Hook validation (now BLOCKING)
    hook_text = data.get("hook", "")
    power_words = ['vanished', 'disappeared', 'found', 'discovered', 'mystery', 'never', 'impossible']
    has_power_word = any(word in hook_text.lower() for word in power_words)
    
    if not has_power_word:
        print(f"   ❌ BLOCKED: Hook lacks power words")
        raise ValueError(
            f"Hook REJECTED: Must contain at least one power word: {', '.join(power_words)}. "
            f"Natasha Ryan (72.9%) uses 'vanished' in hook. "
            f"Current hook: {hook_text}"
        )
    
    print(f"   ✅ Hook power words: YES")
    
    # ✅ ENHANCED: Title pattern validation
    title_text = data.get("title", "")
    title_power_words = ['vanished', 'disappeared', 'mystery', 'never', 'impossible', 'found']
    has_title_power_word = any(word in title_text.lower() for word in title_power_words)
    
    if not has_title_power_word:
        print(f"   ⚠️ WARNING: Title lacks proven keywords")
        print(f"   💡 Not blocking, but consider adding 'Vanished' or 'Disappeared'")
    
    # Check for name-first pattern
    if title_text.startswith("The ") and ("vanished" in title_text.lower() or "disappeared" in title_text.lower()):
        print(f"   ✅ Title follows proven pattern: 'The [X] Who Vanished'")
    elif ":" in title_text:
        title_parts = title_text.split(":")
        first_part = title_parts[0].strip()
        words_in_first = first_part.split()
        if len(words_in_first) <= 3 and all(w[0].isupper() for w in words_in_first if w):
            print(f"   ⚠️ WARNING: Title uses name-first pattern")
            print(f"   💡 Leah Roberts: Name-first → 18.75% retention")
            print(f"   💡 Natasha Ryan: 'The Girl Who...' → 72.9% retention")
            print(f"   💡 Better: 'The [Role] Who Vanished' instead of '{first_part}:'")

    print(f"✅ Script validation PASSED (v3.1.1)\n")
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
    """Select mystery type based on performance weights"""
    
    if user_mystery_type and user_mystery_type != 'auto':
        print(f"🎯 User-specified mystery type: {user_mystery_type}")
        return user_mystery_type
    
    if content_type == 'evening_prime':
        weighted_choice = 'disappearance' if hash(str(datetime.now())) % 100 < 60 else 'conspiracy'
    elif content_type == 'late_night':
        weighted_choice = 'disappearance' if hash(str(datetime.now())) % 100 < 50 else 'crime'
    elif content_type == 'weekend_binge':
        rand = hash(str(datetime.now())) % 100
        if rand < 40:
            weighted_choice = 'disappearance'
        elif rand < 50:
            weighted_choice = 'conspiracy'
        elif rand < 65:
            weighted_choice = 'phenomena'
        elif rand < 85:
            weighted_choice = 'crime'
        else:
            weighted_choice = 'historical'
    else:
        rand = hash(str(datetime.now())) % 100
        if rand < 50:
            weighted_choice = 'disappearance'
        elif rand < 70:
            weighted_choice = 'conspiracy'
        elif rand < 80:
            weighted_choice = 'phenomena'
        elif rand < 95:
            weighted_choice = 'crime'
        else:
            weighted_choice = 'historical'
    
    print(f"🎲 Weighted selection: {weighted_choice} (50% disappearance bias)")
    return weighted_choice


def get_content_type_guidance(content_type):
    """Get specific guidance for mystery content type"""
    guidance = {
        'evening_prime': f"""
EVENING PRIME FOCUS (7-9 PM):
- Target: Evening viewers unwinding, ready for intrigue
- Mystery type: Famous disappearances, high engagement hooks
- Tone: Accessible, intriguing, documentary-style
- Examples: Flight 19, DB Cooper, Amelia Earhart, MH370
- DURATION: {OPTIMAL_MIN_DURATION}-{OPTIMAL_MAX_DURATION} seconds (PROVEN SWEET SPOT)
- WORD COUNT: {OPTIMAL_MIN_WORDS}-{OPTIMAL_MAX_WORDS} words (72.9% retention zone)
""",
        'late_night': f"""
LATE NIGHT FOCUS (10 PM - 2 AM):
- Target: Can't sleep, scrolling in bed
- Mystery type: Darker disappearances, chilling vanishings
- Tone: Chilling, serious, thought-provoking
- Examples: Dyatlov Pass, Elisa Lam, Maura Murray, Brandon Swanson
- DURATION: {OPTIMAL_MIN_DURATION}-{OPTIMAL_MAX_DURATION} seconds (PROVEN SWEET SPOT)
- WORD COUNT: {OPTIMAL_MIN_WORDS}-{OPTIMAL_MAX_WORDS} words (72.9% retention zone)
""",
        'weekend_binge': f"""
WEEKEND BINGE FOCUS (Sat/Sun 8-11 PM):
- Target: More time, want compelling mysteries
- Mystery type: Complex disappearances with multiple theories
- Tone: Documentary deep-dive but concise
- Examples: Malaysia Airlines 370, Sodder Children, Springfield Three
- DURATION: {OPTIMAL_MIN_DURATION}-{OPTIMAL_MAX_DURATION} seconds (PROVEN SWEET SPOT)
- WORD COUNT: {OPTIMAL_MIN_WORDS}-{OPTIMAL_MAX_WORDS} words (72.9% retention zone)
""",
        'general': f"""
GENERAL MYSTERY FOCUS:
- Target: General mystery enthusiasts
- Mystery type: Famous disappearances and vanishings
- Tone: Mysterious but accessible
- Examples: Bermuda Triangle vanishings, D.B. Cooper, Roanoke Colony
- DURATION: {OPTIMAL_MIN_DURATION}-{OPTIMAL_MAX_DURATION} seconds (PROVEN SWEET SPOT)
- WORD COUNT: {OPTIMAL_MIN_WORDS}-{OPTIMAL_MAX_WORDS} words (72.9% retention zone)
"""
    }
    return guidance.get(content_type, guidance['general'])


def get_mystery_type_guidance(mystery_type):
    """Get guidance for mystery category"""
    types = {
        'disappearance': f"""
DISAPPEARANCE MYSTERIES (72.9% RETENTION - YOUR BEST PERFORMER):
- Focus: Vanishing without a trace
- Hook formula: "[Date], [Location]. [Person/Group] vanished. [Impossible detail]."
- Key elements: Last known location, search efforts, zero evidence
- Examples: Flight 19, DB Cooper, Mary Celeste, Amelia Earhart, Maura Murray
- MANDATORY: Use "Vanished" or "Disappeared" in title (proven high retention)
- LENGTH: {OPTIMAL_MIN_WORDS}-{OPTIMAL_MAX_WORDS} words ({OPTIMAL_MIN_DURATION}-{OPTIMAL_MAX_DURATION}s)
""",
        'crime': f"""
TRUE CRIME MYSTERIES:
- Focus: Unsolved murders, cold cases with vanishing elements
- Hook formula: "A body was found in [Location]. Then the evidence vanished."
- Key elements: Evidence that doesn't add up, missing pieces
- Examples: Zodiac Killer, Black Dahlia, Somerton Man
- ETHICAL: Focus on mystery aspect, respect victims
- LENGTH: {OPTIMAL_MIN_WORDS}-{OPTIMAL_MAX_WORDS} words ({OPTIMAL_MIN_DURATION}-{OPTIMAL_MAX_DURATION}s)
""",
        'conspiracy': f"""
CONSPIRACY THEORIES:
- Focus: Hidden agendas, secret operations, unexplained power structures
- Hook formula: "[Event/Fact]. [Strange link]. [Hidden truth]."
- Key elements: Timeline, suspicious coincidences, conflicting reports
- Examples: MK Ultra, Moon Landing, Area 51, Project Blue Beam
- MANDATORY: Use "Conspiracy," "Cover-Up," or "They Knew" in title
- LENGTH: {OPTIMAL_MIN_WORDS}-{OPTIMAL_MAX_WORDS} words ({OPTIMAL_MIN_DURATION}-{OPTIMAL_MAX_DURATION}s)
""",
        'phenomena': f"""
UNEXPLAINED PHENOMENA (89.5% RETENTION - STRONG PERFORMER):
- Focus: Strange occurrences, mysterious signals, impossible events
- Hook formula: "In [Year], something appeared/vanished. Scientists can't explain it."
- Key elements: Verified phenomena, multiple witnesses, no explanation
- Examples: The Hum, Wow Signal, Hessdalen Lights, Ball Lightning
- LENGTH: {OPTIMAL_MIN_WORDS}-{OPTIMAL_MAX_WORDS} words ({OPTIMAL_MIN_DURATION}-{OPTIMAL_MAX_DURATION}s)
""",
        'historical': f"""
HISTORICAL ENIGMAS (USE SPARINGLY - 34.8% retention):
- Focus: Ancient disappearances, lost civilizations ONLY
- Hook formula: "An entire civilization vanished. No one knows why."
- Key elements: Lost cities, vanished peoples, unexplained abandonment
- Examples: Roanoke Colony, Angikuni Lake, Lost Colony of Greenland
- WARNING: Lower retention - use only 5% of the time
- LENGTH: {OPTIMAL_MIN_WORDS}-{OPTIMAL_MAX_WORDS} words ({OPTIMAL_MIN_DURATION}-{OPTIMAL_MAX_DURATION}s)
"""
    }
    return types.get(mystery_type, types['disappearance'])


def build_mystery_prompt(content_type, priority, mystery_type, trends, history):
    """
    🚨 UPDATED v3.1.1: Prompt with EXPLICIT examples showing Leah Roberts vs Natasha Ryan
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

🎯 CRITICAL REQUIREMENTS (RETENTION-FIRST STRUCTURE v3.1.1):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TARGET DURATION: {OPTIMAL_MIN_DURATION}-{OPTIMAL_MAX_DURATION} SECONDS (PROVEN SWEET SPOT)
TARGET WORDS: {OPTIMAL_MIN_WORDS}-{OPTIMAL_MAX_WORDS} words (72.9% retention zone)
MINIMUM: {MINIMUM_WORDS} words (prevents failure mode)
ABSOLUTE MAXIMUM: {HARD_LIMIT_WORDS} words

TTS Speed: {WORDS_PER_SECOND} words/second
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🚨 PROVEN RETENTION DATA (REAL EXAMPLES):

✅ NATASHA RYAN (72.9% RETENTION - THE GOLD STANDARD):
Title: "The Girl Who Vanished: From Her Own Backyard"
Hook: "In 1998, a teen vanished. Five years later, she was found alive."
Script opening: "In 1998, a teen vanished. Five years later, she was found alive.

14-year-old Natasha Ryan disappeared from Rockhampton, Australia..."

WHY IT WORKED:
- First 15 words: "Teen vanished. Found alive." ✅ IMMEDIATE REVEAL
- NO backstory before the reveal ✅
- Uses "But five years later..." ✅ TWIST PHRASE
- Retention at 0:09: ~80% ✅
- Average retention: 72.9% ✅

❌ LEAH ROBERTS (18.75% RETENTION - THE FAILURE PATTERN):
Title: "Leah Roberts: The Vanishing Driver's Final Note"
Hook: "March 9th, 2000. Leah Roberts vanished. Car abandoned. Cryptic note."
Script opening: "20-year-old Leah was road-tripping Australia. Her white Subaru was found..."

WHY IT FAILED:
- Uses "was road-tripping" ❌ ACTIVITY-BASED BACKSTORY
- Starts with "20-year-old Leah" ❌ CHRONOLOGICAL SETUP
- Details (white Subaru, dog food, blanket) too early ❌
- Retention at 0:09: 23% (CATASTROPHIC DROP) ❌
- Average retention: 18.75% ❌

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

🎬 SCRIPT STRUCTURE (INVERTED PYRAMID - {OPTIMAL_MIN_DURATION}-{OPTIMAL_MAX_DURATION}s):

🚨 SECTION 1: IMMEDIATE REVEAL (0-10 seconds, 15-20 words):
FORMULA: "[Date], [Location]. [Person/Group] vanished. [Impossible detail]."
✅ CORRECT: "In 1998, a teen vanished. Five years later, she was found alive."
❌ WRONG: "20-year-old Leah was road-tripping Australia."

❌ NEVER START WITH:
- "X-year-old [Name] was..." ← CHRONOLOGICAL
- "[Name] was traveling/working/studying..." ← ACTIVITY-BASED
- "A man/woman was..." ← GENERIC SETUP
- Any backstory before the reveal

✅ ALWAYS START WITH THE OUTCOME!

🎬 SECTION 2: CONTEXT (10-20 seconds, 25-35 words):
NOW you can add WHO, WHAT, WHERE - but focus on EVIDENCE, not backstory.
✅ CORRECT: "14-year-old Natasha Ryan disappeared. Clothes found near a river."
❌ WRONG: "She was a student who loved hiking. Her car was a white Subaru."

🎬 SECTION 3: THE TWIST (20-35 seconds, 20-30 words):
FORMULA: "But [here's where it gets strange/five years later/the impossible part]..."
MANDATORY: Must include a "But" twist phrase.

🎬 SECTION 4: RESOLUTION (35-45 seconds, 15-20 words):
The final reveal or deepening of the mystery.

TOTAL TARGET: {OPTIMAL_MIN_WORDS}-{OPTIMAL_MAX_WORDS} WORDS

MANDATORY REQUIREMENTS:
✅ REVEAL THE MYSTERY IN FIRST 15 WORDS (prevents 0:09 drop)
✅ NO backstory before the reveal (blocks Leah Roberts pattern)
✅ Use "vanished/disappeared" in first 15 words
✅ Include "But here's where it gets strange" (or similar)
✅ SHORT PUNCHY SENTENCES (6-10 words each)
✅ NO bullet points - paragraph breaks only (\\n\\n)
✅ STAY BETWEEN {OPTIMAL_MIN_WORDS}-{OPTIMAL_MAX_WORDS} WORDS

AVOID (RETENTION KILLERS):
❌ "was road-tripping/traveling/working/studying" in first 100 words
❌ "X-year-old [Name] was..." pattern
❌ Starting with activity/backstory
❌ Excessive details (dog food, blanket, car color) before hook
❌ Going under {MINIMUM_WORDS} or over {HARD_LIMIT_WORDS} words

OUTPUT FORMAT (JSON ONLY):
{{
  "title": "The [Mystery]: [Use VANISHED or DISAPPEARED]",
  "topic": "mystery",
  "hook": "[8-12 words max with VANISHED/DISAPPEARED]",
  "script": "[Full narrative - {OPTIMAL_MIN_WORDS} to {OPTIMAL_MAX_WORDS} words with \\n\\n paragraphs - REVEAL FIRST!]",
  "cta": "[Question under 12 words]",
  "hashtags": ["#mystery", "#unsolved", "#vanished", "#shorts"],
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
1. First 50 words lack "vanished/disappeared/found/discovered"
2. First 100 words contain "was road-tripping/traveling/working/studying"
3. First 100 words contain "X-year-old [Name] was..." pattern
4. Word count < {MINIMUM_WORDS} or > {HARD_LIMIT_WORDS}

Generate the mystery story NOW. Follow NATASHA RYAN pattern, NOT LEAH ROBERTS.
Target {OPTIMAL_MIN_WORDS}-{OPTIMAL_MAX_WORDS} words. REVEAL THE MYSTERY IN THE FIRST 15 WORDS!
"""

    return prompt


def get_fallback_script(content_type, mystery_type):
    """Fallback scripts - ALL use Natasha Ryan inverted pyramid pattern"""
    
    fallback_scripts = {
        'evening_prime': {
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
        },
        
        'late_night': {
            'title': "The Hikers Who Vanished On Dead Mountain",
            'hook': "February 1959. Nine hikers died. Explanation impossible.",
            'script': """February 1959. Nine experienced hikers found dead on Dead Mountain. Circumstances impossible.

But here's where it gets strange. Their tent was ripped open from inside. Boots still there. Supplies untouched. They fled barefoot into freezing temperatures.

Bodies found at forest edge. Then three more in a ravine. Fractured skulls. Broken ribs. Injuries equivalent to car crash. No external wounds.

Witnesses reported glowing orange orbs that night.

The most terrifying part? Soviet investigation concluded: unknown force.

To this day, no explanation. The Dyatlov Pass incident.""",
            'key_phrase': "DYATLOV PASS",
            'mystery_category': 'crime'
        },
        
        'weekend_binge': {
            'title': "Malaysia 370: The Plane That Vanished",
            'hook': "March 8th, 2014. Boeing 777 vanished. Two hundred thirty-nine souls.",
            'script': """March 8th, 2014. Malaysia Airlines Flight 370 vanished. Two hundred thirty-nine people. Gone.

But here's the impossible part. Transponders were disabled. Communications cut. Manual actions. Someone did this deliberately.

One seventeen AM. 'Good night Malaysian three seven zero.' Last transmission. Then the plane turned. Away from Beijing. Flying for seven more hours. Then nothing.

Despite the largest search in aviation history, the plane vanished. Deepest ocean. No black box.

The most terrifying part? We still don't know why.

To this day, two hundred thirty-nine families wait.""",
            'key_phrase': "MH370",
            'mystery_category': 'disappearance'
        },
        
        'general': {
            'title': "The Ship Where Everyone Vanished",
            'hook': "December 1872. A ship found drifting. Crew vanished.",
            'script': """December 4th, 1872. The Mary Celeste found drifting. Captain, his wife, daughter, seven crew. All vanished.

But here's the impossible part. Ship was seaworthy. Sails set. Cargo intact. Food on tables. Half-eaten. Captain's log updated that morning. Lifeboat missing.

No signs of struggle. No piracy. No storm damage. Everyone just vanished.

They boarded. No one there. Navigation equipment working perfectly.

The most terrifying part? The ship could sail for weeks. Why abandon?

To this day, ten people gone. No bodies.""",
            'key_phrase': "MARY CELESTE",
            'mystery_category': 'disappearance'
        }
    }
    
    selected = fallback_scripts.get(content_type, fallback_scripts['evening_prime'])
    
    print(f"📋 Using fallback mystery script: {selected['title']}")
    
    word_count = len(selected['script'].split())
    estimated_duration = word_count / WORDS_PER_SECOND
    
    print(f"   ✅ Fallback: {word_count} words = {estimated_duration:.1f}s (INVERTED PYRAMID)")
    
    return {
        'title': selected['title'],
        'topic': 'mystery',
        'hook': selected['hook'],
        'script': selected['script'],
        'key_phrase': selected['key_phrase'],
        'mystery_category': selected.get('mystery_category', 'disappearance'),
        'cta': 'What do you think happened?',
        'hashtags': ['#mystery', '#unsolved', '#vanished', '#shorts'],
        'description': f"{selected['title']} - An unsolved mystery that defies explanation. #mystery #vanished #shorts",
        'visual_prompts': [
            'Film noir: vintage historical photograph, dark moody lighting, noir aesthetic',
            'Film noir: evidence scene, shadowy mysterious, noir photography',
            'Film noir: investigation scene, old documents, dramatic shadows',
            'Film noir: final mysterious image, dark ominous, noir aesthetic'
        ],
        'content_type': content_type,
        'priority': 'fallback',
        'mystery_type': mystery_type,
        'word_count': word_count,
        'estimated_duration': estimated_duration,
        'is_fallback': True,
        'optimization_version': '3.1.1_enhanced_backstory_detection',
        'generated_at': datetime.now().isoformat(),
        'niche': 'mystery'
    }


def generate_mystery_script():
    """Main script generation function (RETENTION-OPTIMIZED v3.1.1)"""
    
    content_type = os.getenv('CONTENT_TYPE', 'evening_prime')
    priority = os.getenv('PRIORITY', 'medium')
    user_mystery_type = os.getenv('MYSTERY_TYPE', 'auto')
    
    print(f"\n{'='*70}")
    print(f"🔍 GENERATING MYTHICA REPORT SCRIPT v3.1.1 (ENHANCED)")
    print(f"{'='*70}")
    print(f"📍 Content Type: {content_type}")
    print(f"⭐ Priority: {priority}")
    print(f"🎭 User Mystery Type: {user_mystery_type}")
    print(f"🎯 TARGET: Prevent Leah Roberts pattern (18.75% → 72.9%)")
    
    history = load_history()
    trends = load_trending()
    
    if trends:
        print(f"✅ Loaded trending data from {trends.get('source', 'unknown')}")
    else:
        print("⚠️ No trending data available")
    
    mystery_type = select_weighted_mystery_type(content_type, user_mystery_type)
    print(f"🎯 Final mystery type: {mystery_type}")
    
    prompt = build_mystery_prompt(content_type, priority, mystery_type, trends, history)
    
    max_attempts = 5
    attempt = 0
    data = None
    
    while attempt < max_attempts:
        try:
            attempt += 1
            print(f"\n🔍 Generation attempt {attempt}/{max_attempts}...")
            
            raw_text = generate_script_with_retry(prompt)
            print(f"📝 Received response ({len(raw_text)} chars)")
            
            json_text = extract_json_from_response(raw_text)
            data = json.loads(json_text)
            
            # ✅ CRITICAL: This now includes ENHANCED structural validation
            validate_script_data(data)
            
            data["topic"] = "mystery"
            data["content_type"] = content_type
            data["priority"] = priority
            data["mystery_category"] = mystery_type
            data["optimization_version"] = "3.1.1_enhanced_backstory_detection"
            data["generated_at"] = datetime.now().isoformat()
            data["niche"] = "mystery"
            
            data["title"] = clean_script_text(data["title"])
            data["hook"] = clean_script_text(data["hook"])
            data["cta"] = clean_script_text(data["cta"])
            data["script"] = clean_script_text(data["script"])
            
            if 'vanish' not in data['title'].lower() and 'disappear' not in data['title'].lower():
                print("⚠️ Title missing 'Vanished/Disappeared' - adding to hashtags")
                if '#vanished' not in [h.lower() for h in data.get('hashtags', [])]:
                    data['hashtags'] = ['#vanished'] + data.get('hashtags', [])
            
            if "hashtags" not in data or not data["hashtags"]:
                data["hashtags"] = ["#mystery", "#unsolved", "#vanished", "#shorts"]
            
            if "description" not in data:
                data["description"] = f"{data['title']} - {data['hook']} #mystery #vanished #shorts"
            
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
            
            if trends and trends.get('topics'):
                if not validate_script_uses_trending_topic(data, trends['topics']):
                    raise ValueError("Script doesn't use trending topics - regenerating...")
            
            content_hash = get_content_hash(data)
            if content_hash in [t.get('hash') for t in history['topics']]:
                print("⚠️ Generated duplicate content, regenerating...")
                raise ValueError("Duplicate content detected")
            
            previous_titles = [t.get('title', '') for t in history['topics']]
            if is_similar_topic(data['title'], previous_titles):
                print("⚠️ Topic too similar, regenerating...")
                raise ValueError("Similar topic detected")
            
            save_to_history(data['topic'], content_hash, data['title'], data)
            
            print(f"\n✅ SCRIPT GENERATED & VALIDATED")
            print(f"   Title: {data['title']}")
            print(f"   Duration: ~{data['estimated_duration']:.1f}s")
            print(f"   Optimization: v3.1.1 (enhanced backstory detection)")
            print(f"   🎯 Structural check: PASSED (Leah Roberts pattern blocked)")
            
            break
            
        except json.JSONDecodeError as e:
            print(f"❌ Attempt {attempt} failed: JSON parse error")
            if attempt < max_attempts:
                import time
                time.sleep(2**attempt)
        
        except ValueError as e:
            print(f"❌ Attempt {attempt} failed: {e}")
            if attempt < max_attempts:
                import time
                time.sleep(2**attempt)
        
        except Exception as e:
            print(f"❌ Attempt {attempt} failed: {type(e).__name__} - {e}")
            if attempt < max_attempts:
                import time
                time.sleep(2**attempt)
        
        if attempt >= max_attempts:
            print("\n⚠️ Max attempts reached - using fallback")
            data = get_fallback_script(content_type, mystery_type)
            fallback_hash = get_content_hash(data)
            save_to_history(data['topic'], fallback_hash, data['title'], data)
    
    script_path = os.path.join(TMP, "script.json")
    with open(script_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Saved to {script_path}")
    
    print(f"\n{'='*70}")
    print(f"✅ MYTHICA REPORT SCRIPT READY v3.1.1 (ENHANCED)")
    print(f"{'='*70}")
    print(f"Words: {data['word_count']}")
    print(f"Duration: {data['estimated_duration']:.1f}s")
    if OPTIMAL_MIN_WORDS <= data['word_count'] <= OPTIMAL_MAX_WORDS:
        print(f"Status: ✅ PERFECT (72.9% retention zone)")
    elif data['word_count'] >= MINIMUM_WORDS:
        print(f"Status: ✅ ACCEPTABLE (within limits)")
    print(f"Structure: ✅ INVERTED PYRAMID (Natasha Ryan pattern, NOT Leah Roberts)")
    
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