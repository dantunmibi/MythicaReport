#!/usr/bin/env python3
"""
🔍 Generate Mystery Script - OPTIMIZED FOR MYTHICA REPORT
Based on proven performance data: 35-45 seconds = 74-95% retention

OPTIMIZATION CHANGES:
✅ Duration: 45-60s → 35-45s (proven sweet spot)
✅ Word count: 95-126 → 74-95 (optimal range)
✅ Minimum: 70 words (prevents <25s failures)
✅ Category weights: 50% disappearances (94.8% retention winner)
✅ Title bias: "Vanished" keyword (proven performer)
✅ Fallbacks: ALL trimmed to 74-95 words
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

# 🎯 MYTHICA REPORT OPTIMIZED TARGETS (Based on 94.8% retention data)
OPTIMAL_MIN_DURATION = 35   # Sweet spot minimum (was: 45)
OPTIMAL_MAX_DURATION = 45   # Sweet spot maximum (was: 60)

# TTS Reading speed: ~2.1 words/second (mystery narrator at 0.85x speed)
WORDS_PER_SECOND = 2.1

# ✅ DATA-DRIVEN WORD COUNT TARGETS
MINIMUM_WORDS = 70                                                    # NEW: Prevent <25s failures
OPTIMAL_MIN_WORDS = int(OPTIMAL_MIN_DURATION * WORDS_PER_SECOND)      # 74 words for 35s
OPTIMAL_MAX_WORDS = int(OPTIMAL_MAX_DURATION * WORDS_PER_SECOND)      # 95 words for 45s
HARD_LIMIT_WORDS = 100                                                # Safety buffer (was: 129)

# 🔥 CATEGORY WEIGHTS (Based on retention performance)
MYSTERY_WEIGHTS = {
    'disappearance': 0.40,    # 94.8% retention - PRIORITIZE
    'phenomena': 0.30,        # 89.5% retention - STRONG
    'crime': 0.15,            # 74.0% retention - GOOD
    'historical': 0.05,       # 34.8% retention - DEPRIORITIZE
    'conspiracy': 0.10,         # moderate for now
}

# 🎯 TITLE POWER WORDS (Proven performers)
TITLE_POWER_WORDS = ['Vanished', 'Vanished', 'Vanished', 'Disappeared']
# Note: 'Vanished' 3x more likely (your best performers all use this word)

print(f"🎯 Mythica Report Optimization (PROVEN SWEET SPOT):")
print(f"   ✅ Target: {OPTIMAL_MIN_DURATION}-{OPTIMAL_MAX_DURATION}s ({OPTIMAL_MIN_WORDS}-{OPTIMAL_MAX_WORDS} words)")
print(f"   ⚠️ Minimum: {MINIMUM_WORDS} words ({(MINIMUM_WORDS/WORDS_PER_SECOND):.0f}s) - prevents too-short failures")
print(f"   🚨 Hard Limit: {HARD_LIMIT_WORDS} words ({(HARD_LIMIT_WORDS/WORDS_PER_SECOND):.0f}s)")
print(f"   🔥 Category Focus: {int(MYSTERY_WEIGHTS['disappearance']*100)}% Disappearances (94.8% retention)")

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
            return {'topics': [], 'version': '3.0_optimized'}
    
    print("📂 No previous history found, starting fresh")
    return {'topics': [], 'version': '3.0_optimized'}


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
    
    # Keep last 100 topics
    history['topics'] = history['topics'][-100:]
    history['last_updated'] = datetime.now().isoformat()
    history['version'] = '3.0_optimized'
    
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


def validate_script_data(data):
    """✅ OPTIMIZED: Mythica Report proven sweet spot validation"""
    
    required_fields = ["title", "topic", "hook", "script", "cta"]
    
    for field in required_fields:
        if field not in data:
            raise ValueError(f"❌ Missing required field: {field}")
    
    if not isinstance(data["script"], str):
        raise ValueError("❌ Script must be a string (narrative text)")
    
    # ✅ STRICT WORD COUNT VALIDATION (OPTIMIZED RANGE)
    word_count = len(data["script"].split())
    estimated_duration = word_count / WORDS_PER_SECOND
    
    print(f"\n📊 Script Length Analysis:")
    print(f"   Words: {word_count}")
    print(f"   Estimated duration: {estimated_duration:.1f}s")
    print(f"   Optimal range: {OPTIMAL_MIN_DURATION}-{OPTIMAL_MAX_DURATION}s")
    
    # ✅ REJECT IF TOO SHORT (NEW - prevents <25s failures)
    if word_count < MINIMUM_WORDS:
        print(f"   ❌ TOO SHORT: {word_count} words = {estimated_duration:.1f}s")
        print(f"   💡 Your data shows videos under 25s fail (21-35% retention)")
        raise ValueError(f"Script too short: {word_count} words (minimum {MINIMUM_WORDS})")
    
    # ✅ REJECT IF TOO LONG
    if word_count > HARD_LIMIT_WORDS:
        print(f"   ❌ TOO LONG: {word_count} words = {estimated_duration:.1f}s")
        print(f"   💡 Exceeds proven sweet spot")
        raise ValueError(f"Script too long: {word_count} words (max {HARD_LIMIT_WORDS})")
    
    # ✅ PERFECT RANGE (35-45s sweet spot)
    if OPTIMAL_MIN_WORDS <= word_count <= OPTIMAL_MAX_WORDS:
        print(f"   ✅ PERFECT: {word_count} words = {estimated_duration:.1f}s (94.8% RETENTION ZONE)")
    elif word_count < OPTIMAL_MIN_WORDS:
        print(f"   ⚠️ ACCEPTABLE: {word_count} words = {estimated_duration:.1f}s (slightly under optimal)")
        print(f"   💡 Could add 5-10 more words for better engagement")
    else:
        print(f"   ⚠️ ACCEPTABLE: {word_count} words = {estimated_duration:.1f}s (slightly over optimal)")
        print(f"   💡 Could trim to {OPTIMAL_MAX_WORDS} words for peak performance")
    
    # Store metadata
    data["estimated_duration"] = estimated_duration
    data["word_count"] = word_count
    data["optimization_version"] = "3.0_proven_sweet_spot"
    
    # Validate title
    if len(data["title"]) > 100:
        print(f"⚠️ Title too long ({len(data['title'])} chars), truncating...")
        data["title"] = data["title"][:97] + "..."
    
    # Validate hook
    hook_words = len(data["hook"].split())
    if hook_words > 15:
        print(f"⚠️ Hook too long ({hook_words} words), keep under 12 words")
    
    # ✅ NEW: Validate hook effectiveness (prevents complex/abstract hooks)
    hook_text = data.get("hook", "")
    
    # Check for proven power words (your 94.8% retention videos use these)
    power_words = ['vanished', 'disappeared', 'found', 'discovered', 'mystery', 'never', 'impossible', 'conspiracy']
    has_power_word = any(word in hook_text.lower() for word in power_words)
    
    if not has_power_word:
        print(f"⚠️ Hook lacks power words (vanished, disappeared, etc.)")
        print(f"   💡 Your 94.8% retention videos all use these words")
    
    # Check for complex names/terms that need context
    complex_indicators = ['event', 'incident', 'case', 'phenomenon']
    unfamiliar_names = sum(1 for word in hook_text.split() if word and len(word) > 0 and word[0].isupper() and len(word) > 8)
    
    if unfamiliar_names > 1 or any(term in hook_text.lower() for term in complex_indicators):
        print(f"⚠️ Hook may be too complex for vertical video")
        print(f"   💡 Unfamiliar names: {unfamiliar_names}, Complex terms detected")
        print(f"   💡 Consider: More immediate, emotional hook")
    
    # Validate hook length (should be readable in 3 seconds)
    hook_words = len(hook_text.split())
    if hook_words > 12:
        print(f"⚠️ Hook too long: {hook_words} words (max 12 for 3-second read)")
    
    print(f"   Hook analysis: '{hook_text}'")
    
    # ✅ NEW: Validate title pattern (prevents name-first failures)
    title_text = data.get("title", "")
    
    # Check for proven keywords
    title_power_words = ['vanished', 'disappeared', 'mystery', 'never', 'impossible', 'found', 'conspiracy']
    has_title_power_word = any(word in title_text.lower() for word in title_power_words)
    
    if not has_title_power_word:
        print(f"⚠️ Title lacks proven keywords")
        print(f"   💡 Your 94.8% retention video uses 'Vanished' in title")
        print(f"   💡 Consider: Adding 'Vanished' or 'Disappeared'")
    
    # Prefer "The [Subject] Who Vanished" pattern over name-first
    if title_text.startswith("The ") and "vanished" in title_text.lower():
        print(f"   ✅ Title follows proven pattern: 'The [X] Who Vanished'")
    elif ":" in title_text:
        title_parts = title_text.split(":")
        first_part = title_parts[0].strip()
        # Check if first part is a proper name (capitalized words)
        words_in_first = first_part.split()
        if len(words_in_first) <= 3 and all(w[0].isupper() for w in words_in_first if w):
            print(f"   ⚠️ Title uses name-based pattern (lower performance)")
            print(f"   💡 Your data: Name-first titles = 20-35% retention")
            print(f"   💡 Better: 'The [Role] Who Vanished' instead of '{first_part}:'")
    
    print(f"   Title analysis: '{title_text}'")

    print(f"✅ Script validation PASSED")
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
    
    # If user specified a type, respect it
    if user_mystery_type and user_mystery_type != 'auto':
        print(f"🎯 User-specified mystery type: {user_mystery_type}")
        return user_mystery_type
    
    # Auto-select based on content type and weights
    if content_type == 'evening_prime':
        # Evening: prefer disappearances (your best performer)
        weighted_choice = 'disappearance' if hash(str(datetime.now())) % 100 < 60 else 'conspiracy'
    elif content_type == 'late_night':
        # Late night: prefer crime/disappearances
        weighted_choice = 'disappearance' if hash(str(datetime.now())) % 100 < 50 else 'crime'
    elif content_type == 'weekend_binge':
        # Weekend: more variety, but still favor disappearances
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
        # General: heavily weighted to disappearances
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
    """Get specific guidance for mystery content type (OPTIMIZED)"""
    guidance = {
        'evening_prime': f"""
EVENING PRIME FOCUS (7-9 PM):
- Target: Evening viewers unwinding, ready for intrigue
- Mystery type: Famous disappearances, high engagement hooks
- Tone: Accessible, intriguing, documentary-style
- Examples: Flight 19, DB Cooper, Amelia Earhart, MH370
- DURATION: {OPTIMAL_MIN_DURATION}-{OPTIMAL_MAX_DURATION} seconds (PROVEN SWEET SPOT)
- WORD COUNT: {OPTIMAL_MIN_WORDS}-{OPTIMAL_MAX_WORDS} words (94.8% retention zone)
""",
        'late_night': f"""
LATE NIGHT FOCUS (10 PM - 2 AM):
- Target: Can't sleep, scrolling in bed
- Mystery type: Darker disappearances, chilling vanishings
- Tone: Chilling, serious, thought-provoking
- Examples: Dyatlov Pass, Elisa Lam, Maura Murray, Brandon Swanson
- DURATION: {OPTIMAL_MIN_DURATION}-{OPTIMAL_MAX_DURATION} seconds (PROVEN SWEET SPOT)
- WORD COUNT: {OPTIMAL_MIN_WORDS}-{OPTIMAL_MAX_WORDS} words (94.8% retention zone)
""",
        'weekend_binge': f"""
WEEKEND BINGE FOCUS (Sat/Sun 8-11 PM):
- Target: More time, want compelling mysteries
- Mystery type: Complex disappearances with multiple theories
- Tone: Documentary deep-dive but concise
- Examples: Malaysia Airlines 370, Sodder Children, Springfield Three
- DURATION: {OPTIMAL_MIN_DURATION}-{OPTIMAL_MAX_DURATION} seconds (PROVEN SWEET SPOT)
- WORD COUNT: {OPTIMAL_MIN_WORDS}-{OPTIMAL_MAX_WORDS} words (94.8% retention zone)
""",
        'general': f"""
GENERAL MYSTERY FOCUS:
- Target: General mystery enthusiasts
- Mystery type: Famous disappearances and vanishings
- Tone: Mysterious but accessible
- Examples: Bermuda Triangle vanishings, D.B. Cooper, Roanoke Colony
- DURATION: {OPTIMAL_MIN_DURATION}-{OPTIMAL_MAX_DURATION} seconds (PROVEN SWEET SPOT)
- WORD COUNT: {OPTIMAL_MIN_WORDS}-{OPTIMAL_MAX_WORDS} words (94.8% retention zone)
"""
    }
    return guidance.get(content_type, guidance['general'])


def get_mystery_type_guidance(mystery_type):
    """Get guidance for mystery category (OPTIMIZED FOR SWEET SPOT)"""
    types = {
        'disappearance': f"""
DISAPPEARANCE MYSTERIES (94.8% RETENTION - YOUR BEST PERFORMER):
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
CONSPIRACY THEORIES (93.7% RETENTION - HIGH PERFORMER):
- Focus: Hidden agendas, secret operations, unexplained power structures
- Hook formula: "[Event/Fact]. [Strange link or contradiction]. [Hidden truth they don't want you to know]."
- Key elements: Timeline of the event, suspicious coincidences, conflicting reports, suppressed evidence
- Examples: MK Ultra, Moon Landing, Illuminati, Area 51, Project Blue Beam, Mandela Effect
- MANDATORY: Use "Conspiracy," "Cover-Up," or "They Knew" in the title (proven click-through boost)
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
    """Build the mystery script generation prompt (OPTIMIZED FOR MYTHICA REPORT)"""
    
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
                    f"  Hook: {item.get('hook_angle', 'N/A')}\n"
                    f"  Contradiction: {item.get('key_contradiction', 'N/A')}"
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

🎯 CRITICAL REQUIREMENTS (PROVEN 94.8% RETENTION FORMULA):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TARGET DURATION: {OPTIMAL_MIN_DURATION}-{OPTIMAL_MAX_DURATION} SECONDS (PROVEN SWEET SPOT)
TARGET WORDS: {OPTIMAL_MIN_WORDS}-{OPTIMAL_MAX_WORDS} words (94.8% retention zone)
MINIMUM: {MINIMUM_WORDS} words (prevents failure mode)
ABSOLUTE MAXIMUM: {HARD_LIMIT_WORDS} words

TTS Speed: {WORDS_PER_SECOND} words/second
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

⚠️ DATA-PROVEN REQUIREMENTS:
- Videos 35-45s = 74-95% retention ✅
- Videos under 25s = 21-35% retention ❌
- "Vanished" in title = high performance ✅
- Disappearance mysteries = 94.8% retention ✅
- Historical mysteries = 34.8% retention ⚠️

CONTEXT:
- Content type: {content_type}
- Mystery type: {mystery_type}
- Priority: {priority}

PREVIOUSLY COVERED (DO NOT REPEAT):
{chr(10).join(f"  • {t}" for t in previous_topics[-15:]) if previous_topics else '  None yet'}

{trending_mandate}

{content_type_guidance}

{mystery_type_guidance}

SCRIPT STRUCTURE (OPTIMIZED FOR {OPTIMAL_MIN_DURATION}-{OPTIMAL_MAX_DURATION}s):

🎬 HOOK (0-5 seconds, 8-12 words):
Formula: "[Date], [Location]. [Vanished/Found/Discovered]. [Impossible detail]."
Example: "December 5th, 1945. Five planes vanished. No wreckage found."
MANDATORY: Use "Vanished" or "Disappeared" (proven high retention)

🎬 SETUP (10-15 words):
WHO, WHAT, WHEN, WHERE - ultra-concise

🎬 INCIDENT (25-35 words):
What happened. Keep sentences SHORT. Build tension FAST.

🎬 CONTRADICTION (20-30 words):
"But here's where it gets strange..."
Stack impossibilities. Facts that don't add up.

🎬 TWIST (15-20 words):
"The most terrifying part?"
Save best detail for last.

🎬 CLIFFHANGER (8-12 words):
"To this day..." Open question.

TOTAL TARGET: {OPTIMAL_MIN_WORDS}-{OPTIMAL_MAX_WORDS} WORDS

MANDATORY REQUIREMENTS:
✅ Write as FLOWING NARRATIVE (conversational)
✅ SHORT PUNCHY SENTENCES (6-10 words each)
✅ NO bullet points - paragraph breaks only (\\n\\n)
✅ Specific details (dates, names, numbers)
✅ Build to impossible contradiction
✅ Leave mystery UNSOLVED
✅ STAY BETWEEN {OPTIMAL_MIN_WORDS}-{OPTIMAL_MAX_WORDS} WORDS
✅ Use "Vanished" or "Disappeared" in title (proven performer)
✅ Use \\n\\n between paragraphs

AVOID:
❌ Going under {MINIMUM_WORDS} words (automatic failure!)
❌ Going over {HARD_LIMIT_WORDS} words (exceeds sweet spot!)
❌ Long explanations or setup
❌ Generic descriptions
❌ Lists or bullet points
❌ Unverified theories
❌ Special characters in JSON

OUTPUT FORMAT (JSON ONLY):
{{
  "title": "The [Mystery]: [Use VANISHED or DISAPPEARED]",
  "topic": "mystery",
  "hook": "[8-12 words max with VANISHED/DISAPPEARED]",
  "script": "[Full narrative - {OPTIMAL_MIN_WORDS} to {OPTIMAL_MAX_WORDS} words with \\n\\n paragraphs]",
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

EXAMPLE (PERFECT 35-45s RANGE):
{{
  "title": "The Man Who Vanished Mid-Flight",
  "topic": "mystery",
  "hook": "November 24th, 1971. D.B. Cooper vanished mid-air.",
  "script": "A man in a suit boarded Flight 305. Seattle to Portland. Ordered a bourbon. Calm. Normal.\\n\\nMid-flight, he handed the flight attendant a note. 'I have a bomb.' Showed her a briefcase. Red wires. She believed him.\\n\\nCooper demanded two hundred thousand dollars. Four parachutes. Plane landed. FBI delivered everything. Passengers released. Plane took off again.\\n\\nBut here's where it gets strange. Somewhere over Washington, Cooper opened the rear stairs. Jumped. At night. In a rainstorm. Ten thousand feet. Wearing a business suit.\\n\\nThe most terrifying part? Despite the largest FBI manhunt in history, Cooper vanished. No body. No parachute. No evidence. Just twenty dollar bills found in a river nine years later.\\n\\nTo this day, D.B. Cooper remains the only unsolved hijacking in American history.",
  "cta": "Do you think he survived the jump?",
  "hashtags": ["#dbcooper", "#vanished", "#mystery", "#unsolved", "#shorts"],
  "description": "November 24, 1971: D.B. Cooper hijacked a plane, took the ransom, and vanished mid-air. The FBI never found him.",
  "key_phrase": "D.B. COOPER",
  "mystery_category": "disappearance",
  "visual_prompts": [
    "Film noir: 1970s airplane interior, dramatic shadows, vintage aesthetic",
    "Film noir: briefcase with money, noir lighting, mysterious mood",
    "Film noir: open airplane rear stairs at night, stormy sky, dramatic",
    "Film noir: FBI wanted poster, noir documentary style, unsolved case"
  ]
}}

🚨 CRITICAL WORD COUNT REQUIREMENTS:
- MINIMUM: {MINIMUM_WORDS} words (prevents too-short failures)
- OPTIMAL: {OPTIMAL_MIN_WORDS}-{OPTIMAL_MAX_WORDS} words (94.8% retention zone)
- MAXIMUM: {HARD_LIMIT_WORDS} words (hard limit)

🎯 TITLE REQUIREMENTS:
- MUST include "Vanished" or "Disappeared" (proven high retention)
- Format: "The [Subject]: [Use Vanished/Disappeared + Impossible Detail]"

Generate the mystery story NOW. Target {OPTIMAL_MIN_WORDS}-{OPTIMAL_MAX_WORDS} words.
"""

    return prompt


def get_fallback_script(content_type, mystery_type):
    """Fallback scripts - ALL OPTIMIZED to 74-95 word sweet spot"""
    
    fallback_scripts = {
        'evening_prime': {
            'title': "Flight 19: Five Planes That Vanished",
            'hook': "December 5th, 1945. Five planes vanished without trace.",
            'script': """Five torpedo bombers left Fort Lauderdale. Routine training. Fourteen experienced crew. Perfect weather.

Two hours later, Lieutenant Taylor radioed. 'We can't find west. Everything looks wrong.' All compasses failed. Simultaneously.

The Navy launched the biggest search in history. Two hundred forty thousand square miles. Three hundred aircraft. Five days straight.

But here's where it gets strange. Zero debris. No oil slicks. Nothing. Five massive planes. Gone.

The most terrifying part? The rescue plane vanished too. Same night. Thirteen more crew.

To this day, twenty-seven men and six aircraft. No evidence. As if they flew to Mars.""",
            'key_phrase': "FLIGHT 19",
            'mystery_category': 'disappearance'
        },
        
        'late_night': {
            'title': "The Hikers Who Vanished On Dead Mountain",
            'hook': "February 1959. Nine hikers died. Explanation impossible.",
            'script': """Nine experienced hikers entered the Ural Mountains. February 2nd, 1959. They set up camp on Dead Mountain.

They never returned.

Search teams found their tent February 26th. Ripped open from inside. Boots still there. Supplies untouched. Hikers gone.

Bodies found at forest edge. Barefoot. Freezing temperatures. Then three more in a ravine. Fractured skulls. Broken ribs.

But here's where it gets strange. Injuries equivalent to car crash. No external wounds. No struggle.

The most terrifying part? Witnesses reported glowing orange orbs that night. Soviet investigation concluded: unknown force.

To this day, no explanation.""",
            'key_phrase': "DYATLOV PASS",
            'mystery_category': 'crime'
        },
        
        'weekend_binge': {
            'title': "Malaysia 370: The Plane That Vanished",
            'hook': "March 8th, 2014. Boeing 777 vanished. Two hundred thirty-nine souls.",
            'script': """Malaysia Airlines Flight 370. Kuala Lumpur to Beijing. Experienced crew. Perfect conditions. Routine flight.

One seventeen AM. 'Good night Malaysian three seven zero.' Last transmission. Then silence.

Radar showed the plane turning. Away from Beijing. Flying for seven more hours. Then nothing.

But here's where it gets strange. Transponders disabled. Communications cut. Manual actions. Someone did this deliberately.

The most terrifying part? Despite the largest search in aviation history, the plane vanished. Deepest ocean. No black box. No answers.

To this day, families wait. Two hundred thirty-nine people. Gone.""",
            'key_phrase': "MH370",
            'mystery_category': 'disappearance'
        },
        
        'general': {
            'title': "The Ship Where Everyone Vanished",
            'hook': "December 1872. A ship found drifting. Crew vanished.",
            'script': """The Mary Celeste. Seaworthy vessel. Experienced captain. His wife and daughter aboard. Seven crew members.

December 4th, 1872. Another ship spotted her. Sails set. Cargo intact. Drifting aimlessly.

They boarded. No one there. Food on tables. Half-eaten. Captain's log updated that morning. Navigation equipment working. Lifeboat missing.

But here's where it gets strange. No signs of struggle. No piracy. No storm damage. Everyone just vanished.

The most terrifying part? The ship was seaworthy. Could sail for weeks. Why abandon perfectly?

To this day, ten people gone. No bodies. No explanation.""",
            'key_phrase': "MARY CELESTE",
            'mystery_category': 'disappearance'
        }
    }
    
    selected = fallback_scripts.get(content_type, fallback_scripts['evening_prime'])
    
    print(f"📋 Using fallback mystery script: {selected['title']}")
    
    word_count = len(selected['script'].split())
    estimated_duration = word_count / WORDS_PER_SECOND
    
    print(f"   ✅ Fallback: {word_count} words = {estimated_duration:.1f}s (OPTIMIZED)")
    
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
        'optimization_version': '3.0_proven_sweet_spot',
        'generated_at': datetime.now().isoformat(),
        'niche': 'mystery'
    }


def generate_mystery_script():
    """Main script generation function (OPTIMIZED FOR MYTHICA REPORT)"""
    
    content_type = os.getenv('CONTENT_TYPE', 'evening_prime')
    priority = os.getenv('PRIORITY', 'medium')
    user_mystery_type = os.getenv('MYSTERY_TYPE', 'auto')
    
    print(f"\n{'='*70}")
    print(f"🔍 GENERATING MYTHICA REPORT SCRIPT (OPTIMIZED)")
    print(f"{'='*70}")
    print(f"📍 Content Type: {content_type}")
    print(f"⭐ Priority: {priority}")
    print(f"🎭 User Mystery Type: {user_mystery_type}")
    
    history = load_history()
    trends = load_trending()
    
    if trends:
        print(f"✅ Loaded trending data from {trends.get('source', 'unknown')}")
    else:
        print("⚠️ No trending data available")
    
    # Select mystery type with weighting
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
            
            validate_script_data(data)
            
            data["topic"] = "mystery"
            data["content_type"] = content_type
            data["priority"] = priority
            data["mystery_category"] = mystery_type
            data["optimization_version"] = "3.0_proven_sweet_spot"
            data["generated_at"] = datetime.now().isoformat()
            data["niche"] = "mystery"
            
            data["title"] = clean_script_text(data["title"])
            data["hook"] = clean_script_text(data["hook"])
            data["cta"] = clean_script_text(data["cta"])
            data["script"] = clean_script_text(data["script"])
            
            # Verify "Vanished" or "Disappeared" in title (proven performer)
            if 'vanish' not in data['title'].lower() and 'disappear' not in data['title'].lower():
                print("⚠️ Title missing 'Vanished/Disappeared' - adding to hashtags for boost")
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
            print(f"   Optimization: v3.0 (proven sweet spot)")
            
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
    print(f"✅ MYTHICA REPORT SCRIPT READY (OPTIMIZED)")
    print(f"{'='*70}")
    print(f"Words: {data['word_count']}")
    print(f"Duration: {data['estimated_duration']:.1f}s")
    if OPTIMAL_MIN_WORDS <= data['word_count'] <= OPTIMAL_MAX_WORDS:
        print(f"Status: ✅ PERFECT (94.8% retention zone)")
    elif data['word_count'] >= MINIMUM_WORDS:
        print(f"Status: ✅ ACCEPTABLE (within limits)")
    
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