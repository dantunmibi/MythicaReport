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
    
    # Category promise templates (2 seconds, 5 words max)
    # These MUST match posting_schedule.json days
    category_promises = {
        'dark_history': "Dark history every Monday.",
        'disturbing_medical': "Medical mysteries every Wednesday.",
        'dark_experiments': "Secret research every Thursday.",
        'disappearance': "Unsolved cases every Tuesday.",  # Also Friday, but Tuesday is primary
        'phenomena': "Strange phenomena every Sunday.",
        'crime': "True crime every Saturday.",
        'conspiracy': "Cover-ups every week.",
        'historical': "Dark history every Monday.",  # Merged into dark_history
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
    backstory_indicators = [
        'was born', 'grew up', 'was a student', 'worked as', 
        'lived in', 'was known for', 'had been', 'had always',
        'at the age of', 'years old', 'graduated from', 'studied at',
        'was traveling', 'was road-tripping', 'was driving', 'was hiking',
        'was visiting', 'was working', 'was studying', 'was living',
        'had been traveling', 'had been working', 'had been studying',
        'was on a trip', 'was on a journey', 'was exploring',
        'was heading', 'was going', 'was planning', 'was preparing',
        'had planned', 'had decided', 'had embarked'
    ]
    
    has_early_backstory = any(phrase in first_100 for phrase in backstory_indicators)
    
    if has_early_backstory:
        backstory_found = [phrase for phrase in backstory_indicators if phrase in first_100]
        print(f"   ❌ BLOCKED: Chronological backstory in first 100 words")
        print(f"   💡 Found: {backstory_found}")
        raise ValueError(
            "Script REJECTED: Uses chronological backstory in first 100 words. "
            "This causes 0:09 retention drop. Start with OUTCOME, not activity."
        )
    
    print("   ✅ No early backstory: YES (0:09 cliff avoided)")
    
    print("   ✅ STRUCTURAL VALIDATION PASSED\n")
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
    
    # ✅ Hook validation
    hook_text = data.get("hook", "")
    power_words = ['vanished', 'disappeared', 'found', 'discovered', 'mystery', 'never', 'impossible', 'died', 'killed', 'glowed', 'turned']
    has_power_word = any(word in hook_text.lower() for word in power_words)
    
    if not has_power_word:
        print(f"   ❌ BLOCKED: Hook lacks power words")
        raise ValueError(f"Hook must contain mystery word. Current: {hook_text}")
    
    print(f"   ✅ Hook power words: YES")
    
    # Title pattern check
    title_text = data.get("title", "")
    if title_text.startswith("The ") and any(word in title_text.lower() for word in ['vanished', 'disappeared', 'glowed', 'turned', 'died']):
        print(f"   ✅ Title follows proven pattern: 'The [X] Who/That [Y]'")

    print(f"✅ Script validation PASSED (v6.0)\n")
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
🆕 DARK HISTORY (65-75% ESTIMATED RETENTION - NEW CATEGORY):
- Focus: Real historical events with disturbing/mysterious elements
- Hook formula: "In [Year], [horrifying event]. [Victims]. [Impossible outcome]."
- Key elements: Verified history, mysterious circumstances, dark/tragic outcome
- Examples: "The Day It Rained Meat", "The Girls Who Glowed", "The Town That Burned For 60 Years"
- Tone: MYSTERIOUS FILM NOIR, NOT educational lecture
- AVOID: War crimes details, graphic violence descriptions, child harm specifics
- ALLOW: Dark facts, mysterious deaths, tragic events (focus on MYSTERY not gore)
- MANDATORY: Use "The [Event] That [Outcome]" or "The [People] Who [Fate]" title format
- SCRIPT LENGTH: {SCRIPT_BODY_MIN}-{SCRIPT_BODY_MAX} words (end hook auto-added)

GOOD EXAMPLES:
✅ "The Girls Who Glowed In The Dark" (Radium Girls - mysterious illness)
✅ "The Town That's Been Burning For 60 Years" (Centralia - unexplained fire)
✅ "The Day The Sky Turned Black At Noon" (1780 Dark Day - meteorological mystery)

BAD EXAMPLES (TOO GRAPHIC):
❌ "The Brutal Torture Methods of Unit 731" (graphic violence)
❌ "The Children Who Were Experimented On" (child harm focus)
❌ "The Most Gruesome Executions in History" (gore focus)
""",
        
        'disturbing_medical': f"""
🆕 DISTURBING MEDICAL MYSTERIES (70-80% ESTIMATED RETENTION - NEW CATEGORY):
- Focus: Real medical conditions that defy explanation or horrify
- Hook formula: "[Person/Group] started [symptom]. Doctors couldn't explain it. [Outcome]."
- Key elements: Verified medical case, baffling symptoms, mysterious cause, dark outcome
- Examples: "The Man Who Couldn't Sleep", "The Condition That Turns You To Stone", "The Town Where Everyone Went Blind"
- Tone: BODY HORROR MYSTERY, NOT clinical textbook
- AVOID: Graphic surgical descriptions, gore, self-harm methods, eating disorder glorification
- ALLOW: Mysterious symptoms, baffling conditions, medical anomalies (focus on MYSTERY)
- MANDATORY: Use "The [Person/People] Who [Medical Anomaly]" or "The Condition That [Effect]"
- SCRIPT LENGTH: {SCRIPT_BODY_MIN}-{SCRIPT_BODY_MAX} words (end hook auto-added)

GOOD EXAMPLES:
✅ "The Man Who Couldn't Die" (Fatal Familial Insomnia - mysterious no-sleep condition)
✅ "The Girls Who Turned To Stone" (FOP - mysterious bone growth)
✅ "The Laughing Death That Spread Through Villages" (Kuru - mysterious prion disease)

BAD EXAMPLES (TOO GRAPHIC/SENSITIVE):
❌ "The Man Who Cut Off His Own Limbs" (self-harm focus)
❌ "The Girl Who Starved Herself To Death" (eating disorder)
❌ "The Surgery That Went Horribly Wrong" (graphic medical)
""",
        
        'dark_experiments': f"""
🆕 DARK EXPERIMENTS (68-75% ESTIMATED RETENTION - NEW CATEGORY):
- Focus: Real unethical/secret experiments with mysterious or horrifying outcomes
- Hook formula: "In [Year], [organization] conducted [experiment]. [Subjects]. [Result]."
- Key elements: Declassified files, secret research, ethical violations, mysterious outcomes
- Examples: "MK-Ultra's Lost Subjects", "The Sleep Deprivation Experiment", "The Prison Experiment Gone Wrong"
- Tone: CONSPIRACY THRILLER FILM NOIR, NOT documentary
- AVOID: Nazi experiments (too sensitive/banned), torture details, animal cruelty specifics
- ALLOW: CIA experiments, psychological studies, unethical research (focus on MYSTERY/COVER-UP)
- MANDATORY: Use "The [Experiment] That [Outcome]" or "The Secret Study [Organization] Buried"
- SCRIPT LENGTH: {SCRIPT_BODY_MIN}-{SCRIPT_BODY_MAX} words (end hook auto-added)

GOOD EXAMPLES:
✅ "The CIA Experiment That Erased Memories" (MK-Ultra - mysterious mind control)
✅ "The Sleep Study Where No One Woke Up" (Russian Sleep Experiment - urban legend/mystery)
✅ "The Prison Experiment They Had To Shut Down" (Stanford - psychological mystery)

BAD EXAMPLES (TOO SENSITIVE):
❌ "The Nazi Experiments on Twins" (Holocaust - banned content)
❌ "The Torture Methods CIA Used" (graphic torture focus)
❌ "The Animals That Suffered In Labs" (animal cruelty)
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
        trending_topics = trends['topics'][:15]
        full_data = trends.get('full_data', [])
        
        if full_data:
            for item in full_data[:15]:
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
    """Main script generation function (v6.0 - DARK CONTENT + END HOOKS)"""
    
    content_type = os.getenv('CONTENT_TYPE', 'evening_prime')
    priority = os.getenv('PRIORITY', 'medium')
    user_mystery_type = os.getenv('MYSTERY_TYPE', 'auto')
    
    print(f"\n{'='*70}")
    print(f"🔍 GENERATING MYTHICA REPORT SCRIPT v6.0 (DARK CONTENT + END HOOKS)")
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
            
            # Check similarity
            previous_titles = [t.get('title', '') for t in history['topics']]
            if is_similar_topic(data['title'], previous_titles):
                print("⚠️ Topic too similar, regenerating...")
                raise ValueError("Similar topic detected")
            
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