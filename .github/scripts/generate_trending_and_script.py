#!/usr/bin/env python3
"""
🔍 Generate Mystery Script - YOUTUBE SHORTS OPTIMIZED
YouTube Shorts: MAX 60 seconds (official limit)
Extended Shorts: Up to 120 seconds (if needed)
Optimal: 45-60 seconds for best performance
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

# 🎯 YOUTUBE SHORTS DURATION TARGETS
# YouTube Shorts official limit: 60 seconds
# Extended limit: 120 seconds (for longer vertical videos)
OPTIMAL_MIN_DURATION = 45   # Sweet spot minimum
OPTIMAL_MAX_DURATION = 60   # YouTube Shorts official max
ABSOLUTE_MAX_DURATION = 60 # Absolute maximum (use sparingly)

# TTS Reading speed: ~150 words/minute = 2.5 words/second
# At 0.85x speed (slower for mystery): ~2.1 words/second
WORDS_PER_SECOND = 2.1

# Word count targets
OPTIMAL_MIN_WORDS = int(OPTIMAL_MIN_DURATION * WORDS_PER_SECOND)  # ~95 words for 45s
OPTIMAL_MAX_WORDS = int(OPTIMAL_MAX_DURATION * WORDS_PER_SECOND)  # ~125 words for 60s
ABSOLUTE_MAX_WORDS = int(ABSOLUTE_MAX_DURATION * WORDS_PER_SECOND) # ~250 words for 120s

print(f"🎯 YouTube Shorts Target:")
print(f"   Optimal: {OPTIMAL_MIN_DURATION}-{OPTIMAL_MAX_DURATION}s ({OPTIMAL_MIN_WORDS}-{OPTIMAL_MAX_WORDS} words)")
print(f"   Max allowed: {ABSOLUTE_MAX_DURATION}s ({ABSOLUTE_MAX_WORDS} words)")

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
    
    print(f"✅ Using model: {model_name}")
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
            return {'topics': [], 'version': '2.0_mystery'}
    
    print("📂 No previous history found, starting fresh")
    return {'topics': [], 'version': '2.0_mystery'}


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
        'word_count': len(script_data.get('script', '').split()),
        'estimated_duration': script_data.get('estimated_duration', 0),
        'date': datetime.now().isoformat(),
        'timestamp': datetime.now().timestamp()
    })
    
    # Keep last 100 topics
    history['topics'] = history['topics'][-100:]
    history['last_updated'] = datetime.now().isoformat()
    history['version'] = '2.0_mystery'
    
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
                days_ago = idx
                print(f"⚠️ Topic too similar ({base_similarity:.2f} > {adjusted_threshold:.2f})")
                print(f"   To: {prev_title}")
                print(f"   From: ~{days_ago} videos ago")
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
        print(f"   Keywords: {trend_keywords[:10]}")
        return False
    
    print(f"✅ Script uses trending topics ({matches} keyword matches)")
    return True


def validate_script_data(data):
    """Validate generated script has all required fields (YOUTUBE SHORTS VERSION)"""
    
    required_fields = ["title", "topic", "hook", "script", "cta"]
    
    for field in required_fields:
        if field not in data:
            raise ValueError(f"Missing required field: {field}")
    
    if not isinstance(data["script"], str):
        raise ValueError("script must be a string (narrative text)")
    
    # ✅ UPDATED: Flexible word count validation
    word_count = len(data["script"].split())
    estimated_duration = word_count / WORDS_PER_SECOND
    
    print(f"\n📊 Script Length Analysis:")
    print(f"   Words: {word_count}")
    print(f"   Estimated duration: {estimated_duration:.1f}s")
    
    # Validation tiers
    if word_count < OPTIMAL_MIN_WORDS:
        raise ValueError(f"Script too short: {word_count} words (need {OPTIMAL_MIN_WORDS}-{OPTIMAL_MAX_WORDS} for optimal)")
    
    elif word_count <= OPTIMAL_MAX_WORDS:
        print(f"   ✅ OPTIMAL length for YouTube Shorts ({OPTIMAL_MIN_DURATION}-{OPTIMAL_MAX_DURATION}s)")
    
    elif word_count <= int(OPTIMAL_MAX_DURATION * 1.5 * WORDS_PER_SECOND):  # ~90s
        print(f"   ⚠️ LONGER than optimal but acceptable ({estimated_duration:.0f}s)")
        print(f"   Recommendation: Trim to {OPTIMAL_MAX_WORDS} words for better performance")
    
    elif word_count <= ABSOLUTE_MAX_WORDS:
        print(f"   ⚠️ WARNING: Very long ({estimated_duration:.0f}s)")
        print(f"   Approaching {ABSOLUTE_MAX_DURATION}s limit - may need trimming")
    
    else:
        print(f"   ❌ TOO LONG: {word_count} words = {estimated_duration:.0f}s")
        print(f"   YouTube Shorts extended limit: {ABSOLUTE_MAX_DURATION}s ({ABSOLUTE_MAX_WORDS} words)")
        raise ValueError(f"Script exceeds {ABSOLUTE_MAX_DURATION}s limit: {word_count} words")
    
    # Store estimated duration in data
    data["estimated_duration"] = estimated_duration
    data["word_count"] = word_count
    
    # Validate title length
    if len(data["title"]) > 100:
        print(f"⚠️ Title too long ({len(data['title'])} chars), truncating...")
        data["title"] = data["title"][:97] + "..."
    
    # Validate hook length
    hook_words = len(data["hook"].split())
    if hook_words > 15:
        print(f"⚠️ Hook too long ({hook_words} words), ideally ≤12 words")
    
    print(f"✅ Script validation passed")
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
    
    raise ValueError("No JSON found in response")


def clean_script_text(text):
    """Clean script text of problematic characters"""
    text = text.replace('"', '').replace('"', '')
    text = text.replace(''', "'").replace(''', "'")
    text = text.replace('\u2018', "'").replace('\u2019', "'")
    text = text.replace('\u201c', '').replace('\u201d', '')
    
    return text


def get_content_type_guidance(content_type):
    """Get specific guidance for mystery content type"""
    guidance = {
        'evening_prime': """
EVENING PRIME FOCUS (7-9 PM):
- Target: Evening viewers unwinding, ready for intrigue
- Mystery type: Famous mysteries, high engagement hooks
- Tone: Accessible, intriguing, documentary-style
- Examples: Flight 19, DB Cooper, Zodiac Killer, MH370
- Energy: Mysterious but not too dark, engaging narrative
- DURATION: 45-60 seconds (optimal for evening browsing)
""",
        'late_night': """
LATE NIGHT FOCUS (10 PM - 2 AM):
- Target: Can't sleep, scrolling in bed
- Mystery type: Darker, more disturbing mysteries
- Tone: Chilling, serious, thought-provoking
- Examples: Dyatlov Pass, Elisa Lam, Somerton Man
- Energy: Eerie, unsettling, keeps them thinking
- DURATION: 45-75 seconds (they have more time to watch)
""",
        'weekend_binge': """
WEEKEND BINGE FOCUS (Sat/Sun 8-11 PM):
- Target: More time, want depth
- Mystery type: Complex historical mysteries
- Tone: Documentary deep-dive, detailed
- Examples: Voynich Manuscript, Antikythera Mechanism
- Energy: Intellectually satisfying, detailed
- DURATION: 60-90 seconds (can be longer, they're binging)
""",
        'general': """
GENERAL MYSTERY FOCUS:
- Target: General mystery enthusiasts
- Mystery type: Balanced mix
- Tone: Mysterious but accessible
- Examples: Bermuda Triangle, Oak Island, Roanoke
- Energy: Intriguing, engaging, binge-worthy
- DURATION: 45-60 seconds (optimal for broad appeal)
"""
    }
    return guidance.get(content_type, guidance['general'])


def get_mystery_type_guidance(mystery_type):
    """Get guidance for mystery category"""
    types = {
        'disappearance': """
DISAPPEARANCE MYSTERIES:
- Focus: Vanishing without a trace
- Hook formula: "[Date], [Location]. [Person/Group] vanished. [Impossible detail]."
- Key elements: Last known location, search efforts, zero evidence
- Examples: Flight 19, DB Cooper, Mary Celeste, Amelia Earhart
- LENGTH: Keep tight - 45-60s optimal
""",
        'crime': """
TRUE CRIME MYSTERIES:
- Focus: Unsolved murders, cold cases
- Hook formula: "A body was found in [Location]. What investigators discovered was impossible."
- Key elements: Evidence that doesn't add up, multiple theories
- Examples: Zodiac Killer, Black Dahlia, Somerton Man
- ETHICAL: Focus on mystery aspect, respect victims
- LENGTH: 45-75s (can elaborate on evidence)
""",
        'historical': """
HISTORICAL ENIGMAS:
- Focus: Ancient artifacts, unexplained discoveries
- Hook formula: "In [Year], they discovered [Object]. Scientists still can't explain it."
- Key elements: Technology that shouldn't exist
- Examples: Antikythera Mechanism, Voynich Manuscript, Göbekli Tepe
- LENGTH: 60-90s (need context for historical mysteries)
""",
        'conspiracy': """
DECLASSIFIED CONSPIRACIES:
- Focus: Proven conspiracies with documents
- Hook formula: "Declassified documents prove [Operation] was real."
- Key elements: Government documents, verified facts
- Examples: MKUltra, Operation Northwoods, Tuskegee
- WARNING: ONLY proven conspiracies
- LENGTH: 45-60s (stick to facts, stay concise)
"""
    }
    return types.get(mystery_type, types['disappearance'])


def build_mystery_prompt(content_type, priority, mystery_type, trends, history):
    """Build the mystery script generation prompt (YOUTUBE SHORTS OPTIMIZED)"""
    
    previous_topics = [
        f"{t.get('topic', 'unknown')}: {t.get('title', '')}" 
        for t in history['topics'][-20:]
    ]
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
                    f"  Contradiction: {item.get('key_contradiction', 'N/A')}\n"
                    f"  Category: {item.get('category', 'mystery')}"
                )
        else:
            trending_summaries = [f"• {t}" for t in trending_topics]
        
        print(f"🔍 Using {len(trending_topics)} trending mystery topics in prompt")
    
    if trending_topics:
        trending_mandate = f"""
⚠️⚠️⚠️ CRITICAL MANDATORY REQUIREMENT ⚠️⚠️⚠️

YOU MUST CREATE A SCRIPT ABOUT ONE OF THESE REAL TRENDING MYSTERY TOPICS:

{chr(10).join(trending_summaries)}

These are REAL trending mysteries from today ({datetime.now().strftime('%Y-%m-%d %H:%M')}).

YOU MUST PICK ONE AND CREATE A COMPELLING {OPTIMAL_MIN_DURATION}-{OPTIMAL_MAX_DURATION} SECOND NARRATIVE.
"""
    else:
        trending_mandate = "⚠️ No trending data available - create original mystery content\n"
    
    content_type_guidance = get_content_type_guidance(content_type)
    mystery_type_guidance = get_mystery_type_guidance(mystery_type)
    
    # Determine target word count based on content type
    if content_type == 'weekend_binge':
        target_words = int((OPTIMAL_MAX_DURATION * 1.5) * WORDS_PER_SECOND)  # ~90s = 189 words
        duration_range = f"{OPTIMAL_MAX_DURATION}-90 seconds"
    elif content_type == 'late_night':
        target_words = int((OPTIMAL_MAX_DURATION * 1.25) * WORDS_PER_SECOND)  # ~75s = 158 words
        duration_range = f"{OPTIMAL_MIN_DURATION}-75 seconds"
    else:
        target_words = OPTIMAL_MAX_WORDS  # ~60s = 125 words
        duration_range = f"{OPTIMAL_MIN_DURATION}-{OPTIMAL_MAX_DURATION} seconds"
    
    prompt = f"""You are an expert mystery storyteller creating YOUTUBE SHORTS content.

🎯 CRITICAL YOUTUBE SHORTS REQUIREMENTS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TARGET DURATION: {duration_range}
TARGET WORDS: ~{target_words} words
MAX DURATION: {ABSOLUTE_MAX_DURATION} seconds (ABSOLUTE LIMIT)
MAX WORDS: {ABSOLUTE_MAX_WORDS} words (DO NOT EXCEED)

TTS Speed: {WORDS_PER_SECOND} words/second (slow mysterious pace)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CONTEXT:
- Current date: {datetime.now().strftime('%Y-%m-%d')}
- Content type: {content_type}
- Mystery type: {mystery_type}
- Priority: {priority}

PREVIOUSLY COVERED (DO NOT REPEAT):
{chr(10).join(f"  • {t}" for t in previous_topics[-15:]) if previous_topics else '  None yet'}

{trending_mandate}

{content_type_guidance}

{mystery_type_guidance}

CRITICAL STRUCTURE FOR SHORTS (Ultra-Concise):

🎬 ACT 1 - THE HOOK (0-5 seconds, 8-12 words):
Formula: "[Date], [Location]. [Impossible event]."

Examples:
✅ "December 5th, 1945. Five planes vanished. No wreckage found."
✅ "October 21st, 1978. Pilot radioed: It's not an aircraft. Then vanished."

🎬 ACT 2 - THE MYSTERY ({duration_range.split('-')[0]}-{duration_range.split('-')[1].split()[0]} seconds, ~{int(target_words * 0.75)} words):

SETUP (Quick context):
- WHO: Name
- WHAT: Situation  
- WHEN: Date/Time
- WHERE: Location
(20-30 words)

INCIDENT (What happened):
- The unexplainable event
- Keep sentences punchy
- Build tension fast
(40-50 words)

CONTRADICTION (Why it's impossible):
- "But here's where it gets strange..."
- Stack impossibilities
- Facts that don't add up
(30-40 words)

TWIST (Final chilling detail):
- "The most terrifying part?"
- Save best detail for last
(20-30 words)

🎬 ACT 3 - CLIFFHANGER (Last 5 seconds, 10-15 words):
- "To this day..." statement
- Prompt theories

NARRATIVE REQUIREMENTS:
✅ Write as FLOWING NARRATIVE (conversational storytelling)
✅ SHORT punchy sentences (6-10 words each)
✅ NO bullet points in script
✅ Specific details (dates, names, numbers)
✅ Build to impossible contradiction
✅ Leave mystery UNSOLVED
✅ {target_words} words MAXIMUM (strict limit)
✅ Paragraph breaks (\\n\\n) between acts
✅ Proper punctuation for TTS pauses

AVOID:
❌ Going over {target_words} words (YouTube Shorts has time limits!)
❌ Long explanations (no time in Shorts!)
❌ "Today I'll tell you..." (wastes precious seconds)
❌ Lists/bullet points
❌ Graphic violence
❌ Unverified theories
❌ Special characters in JSON

VISUAL PROMPTS:
Include "film noir photography, high contrast black and white, dramatic shadows, vintage aesthetic, 
mysterious atmosphere, documentary style, cinematic lighting, noir aesthetic"

OUTPUT FORMAT (JSON ONLY):
{{
  "title": "The [Mystery]: [Intriguing Statement] (under 60 chars)",
  "topic": "mystery",
  "hook": "[8-12 words max for first 5 seconds]",
  "script": "[Full narrative {target_words} words MAX, flowing paragraphs with \\n\\n breaks]",
  "cta": "[Question under 12 words]",
  "hashtags": ["#mystery", "#unsolved", "#truecrime", "#shorts"],
  "description": "[2 sentences for YouTube description]",
  "key_phrase": "[3-5 words for thumbnail, CAPS]",
  "mystery_category": "{mystery_type}",
  "visual_prompts": [
    "Film noir: [hook scene], dramatic shadows, mysterious atmosphere, noir aesthetic",
    "Film noir: [setup scene], vintage documentary style, film grain, noir lighting",
    "Film noir: [incident scene], dark and ominous, high contrast, noir photography",
    "Film noir: [twist scene], final mysterious reveal, dramatic noir aesthetic"
  ]
}}

EXAMPLE (60-SECOND TARGET):
{{
  "title": "Flight 19: Vanished Without Trace",
  "topic": "mystery",
  "hook": "December 5th, 1945. Five planes vanished. No wreckage found.",
  "script": "Five torpedo bombers took off from Fort Lauderdale. Routine training mission. Fourteen experienced crew. Perfect weather.\\n\\nTwo hours later, Lieutenant Taylor radioed controllers. 'We can't find west. Everything looks wrong.' All five planes. Compasses malfunctioning. Simultaneously.\\n\\nThe Navy launched the biggest search in history. Two hundred forty thousand square miles. Three hundred aircraft. Five straight days.\\n\\nBut here's where it gets strange. Zero debris. No oil slicks. No wreckage. Nothing. Five massive planes. Gone. As if they never existed.\\n\\nThe most terrifying part? The rescue plane sent to find them? Also vanished. Same night. Thirteen more crew. No distress call.\\n\\nTo this day, twenty-seven men and six aircraft. Zero evidence. The Navy's conclusion: 'as if they flew to Mars.'",
  "cta": "What do you think happened to them?",
  "hashtags": ["#flight19", "#bermudatriangle", "#mystery", "#unsolved", "#shorts"],
  "description": "December 5, 1945: Flight 19 disappeared with 14 crew. The rescue plane also vanished. 27 men, 6 aircraft, zero evidence. What happened?",
  "key_phrase": "FLIGHT 19",
  "mystery_category": "disappearance",
  "visual_prompts": [
    "Film noir: vintage 1940s torpedo bombers in formation over ocean, dramatic clouds, noir aesthetic, ominous mood",
    "Film noir: old military radio equipment, dramatic lighting, noir documentary style, mysterious shadows",
    "Film noir: vast empty ocean aerial view, foggy and mysterious, no wreckage, ominous atmosphere, noir photography",
    "Film noir: vintage search map with red pins, classified document aesthetic, dramatic shadows, noir lighting"
  ]
}}

⚠️ CRITICAL: Stay under {target_words} words! YouTube Shorts has strict time limits!

Generate the mystery story now.
"""

    return prompt


def get_fallback_script(content_type, mystery_type):
    """Fallback mystery scripts optimized for Shorts duration"""
    
    fallback_scripts = {
        'evening_prime': {
            'title': "Flight 19: Vanished Without Trace",
            'hook': "December 5th, 1945. Five planes vanished. No wreckage found.",
            'script': """Five torpedo bombers took off from Fort Lauderdale. Routine training mission. Fourteen experienced crew. Perfect weather.

Two hours later, Lieutenant Taylor radioed controllers. 'We can't find west. Everything looks wrong.' All five planes. Compasses malfunctioning. Simultaneously.

The Navy launched the biggest search in history. Two hundred forty thousand square miles. Three hundred aircraft. Five straight days.

But here's where it gets strange. Zero debris. No oil slicks. No wreckage. Nothing. Five massive planes. Gone. As if they never existed.

The most terrifying part? The rescue plane sent to find them? Also vanished. Same night. Thirteen more crew. No distress call.

To this day, twenty-seven men and six aircraft. Zero evidence. The Navy's conclusion: 'as if they flew to Mars.'""",
            'key_phrase': "FLIGHT 19",
            'mystery_category': 'disappearance'
        },
        
        'late_night': {
            'title': "Dyatlov Pass: Nine Dead Hikers",
            'hook': "February 1959. Nine hikers died. The explanation impossible.",
            'script': """Nine experienced hikers. Ural Mountains. February 2nd, 1959. They set up camp on Kholat Syakhl. 'Dead Mountain' in the local language.

They never came back.

Search teams found their tent February 26th. Ripped open. From the inside. Boots still there. Supplies untouched. But the hikers? Gone.

First bodies found at forest edge. Barefoot. Freezing temperatures. Dead from hypothermia. Then three more in a ravine. Fractured skulls. Broken ribs. One missing her tongue.

But here's where it gets strange. The injuries? Equivalent to a car crash. But no external wounds. No signs of struggle. Their clothes? High radiation levels.

The most terrifying part? Witnesses reported glowing orange orbs in the sky that night. The Soviet investigation concluded: 'death by unknown compelling force.'

To this day, no one can explain it.""",
            'key_phrase': "DYATLOV PASS",
            'mystery_category': 'crime'
        },
        
        'weekend_binge': {
            'title': "Voynich: The Unreadable Book",
            'hook': "A 600-year-old book. Written in unknown language.",
            'script': """1912. An Italian villa. Book dealer Wilfrid Voynich discovered a medieval manuscript. 240 pages of bizarre illustrations. Plants that don't exist. Impossible astronomical charts. Strange ceremonies. And text. Lots of text. In a language no one recognizes.

Carbon dating confirmed it. Early 15th century. Over 600 years old. Written on real vellum. Medieval ink. Not a hoax.

World War Two codebreakers tried to decode it. The NSA tried. Computer algorithms failed. The text follows language patterns. Grammar. Structure. Syntax. But the words? Complete gibberish.

But here's where it gets strange. The plants don't match any known species. Impossible hybrids. The star charts show constellations that don't exist. The anatomy drawings make no sense.

The most terrifying part? Linguistic analysis proves it's not random. Too much structure. Too much consistency. Someone spent years writing this.

After 600 years and countless experts, no one can read a single word.""",
            'key_phrase': "VOYNICH MANUSCRIPT",
            'mystery_category': 'historical'
        },
        
        'general': {
            'title': "Bermuda Triangle: Where Ships Vanish",
            'hook': "Hundreds of ships. All vanished without trace.",
            'script': """The Bermuda Triangle. Miami to Bermuda to Puerto Rico. Past century? Fifty ships. Twenty aircraft. All disappeared. No wreckage. No signals. Gone.

December 1945. Flight 19. Five bombers vanished. The rescue plane? Also disappeared. March 1918. USS Cyclops. Three hundred nine crew. Vanished. No SOS. Largest US Naval loss not in combat.

Weather doesn't explain it. No more storms than elsewhere. Magnetic anomalies? Normal. Methane gas? No evidence. Rogue waves? They don't erase radio signals.

But here's where it gets strange. Ships lose radio contact suddenly. No mayday. No beacons. Just silence. Search teams find nothing. No debris. No oil. Empty ocean. As if vessels ceased to exist.

The most terrifying part? It still happens. 2015. El Faro disappeared. Thirty-three crew. Modern systems. Satellite comms. The black box recording ends abruptly. Mid-transmission. No explanation.

What happens in the Bermuda Triangle?""",
            'key_phrase': "BERMUDA TRIANGLE",
            'mystery_category': 'disappearance'
        }
    }
    
    selected = fallback_scripts.get(content_type, fallback_scripts['evening_prime'])
    
    print(f"📋 Using fallback mystery script: {selected['title']}")
    
    word_count = len(selected['script'].split())
    estimated_duration = word_count / WORDS_PER_SECOND
    
    return {
        'title': selected['title'],
        'topic': 'mystery',
        'hook': selected['hook'],
        'script': selected['script'],
        'key_phrase': selected['key_phrase'],
        'mystery_category': selected.get('mystery_category', 'disappearance'),
        'cta': 'What do you think happened?',
        'hashtags': ['#mystery', '#unsolved', '#truecrime', '#shorts'],
        'description': f"{selected['title']} - An unsolved mystery that defies explanation. #mystery #unsolved #shorts",
        'visual_prompts': [
            'Film noir: vintage historical photograph, dark moody lighting, noir aesthetic, mysterious atmosphere',
            'Film noir: evidence scene, shadowy mysterious, vintage 1940s aesthetic, noir photography',
            'Film noir: investigation scene, old documents, dramatic shadows, noir lighting',
            'Film noir: final mysterious image, dark ominous, film grain, noir aesthetic'
        ],
        'content_type': content_type,
        'priority': 'fallback',
        'mystery_type': mystery_type,
        'word_count': word_count,
        'estimated_duration': estimated_duration,
        'is_fallback': True,
        'generated_at': datetime.now().isoformat(),
        'niche': 'mystery'
    }


def generate_mystery_script():
    """Main script generation function (YOUTUBE SHORTS OPTIMIZED)"""
    
    content_type = os.getenv('CONTENT_TYPE', 'evening_prime')
    priority = os.getenv('PRIORITY', 'medium')
    mystery_type = os.getenv('MYSTERY_TYPE', 'auto')
    
    print(f"\n{'='*70}")
    print(f"🔍 GENERATING YOUTUBE SHORTS MYSTERY SCRIPT")
    print(f"{'='*70}")
    print(f"📍 Content Type: {content_type}")
    print(f"⭐ Priority: {priority}")
    print(f"🎭 Mystery Type: {mystery_type}")
    print(f"⏱️ Target: {OPTIMAL_MIN_DURATION}-{OPTIMAL_MAX_DURATION}s ({OPTIMAL_MIN_WORDS}-{OPTIMAL_MAX_WORDS} words)")
    print(f"🚨 Max limit: {ABSOLUTE_MAX_DURATION}s ({ABSOLUTE_MAX_WORDS} words)")
    
    history = load_history()
    trends = load_trending()
    
    if trends:
        print(f"✅ Loaded trending data from {trends.get('source', 'unknown')}")
        print(f"   Topics: {len(trends.get('topics', []))}")
    else:
        print("⚠️ No trending data available")
    
    if mystery_type == 'auto':
        if content_type == 'evening_prime':
            mystery_type = 'disappearance'
        elif content_type == 'late_night':
            mystery_type = 'crime'
        elif content_type == 'weekend_binge':
            mystery_type = 'historical'
        else:
            mystery_type = 'disappearance'
        print(f"   Auto-selected mystery type: {mystery_type}")
    
    prompt = build_mystery_prompt(content_type, priority, mystery_type, trends, history)
    
    max_attempts = 5
    attempt = 0
    
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
            data["generated_at"] = datetime.now().isoformat()
            data["niche"] = "mystery"
            
            data["title"] = clean_script_text(data["title"])
            data["hook"] = clean_script_text(data["hook"])
            data["cta"] = clean_script_text(data["cta"])
            data["script"] = clean_script_text(data["script"])
            
            if "hashtags" not in data or not data["hashtags"]:
                data["hashtags"] = ["#mystery", "#unsolved", "#truecrime", "#shorts"]
            
            if "description" not in data:
                data["description"] = f"{data['title']} - {data['hook']} #mystery #unsolved #shorts"
            
            if "visual_prompts" not in data or len(data["visual_prompts"]) < 4:
                data["visual_prompts"] = [
                    f"Film noir: mysterious scene, dark moody lighting, noir aesthetic",
                    f"Film noir: evidence scene, shadowy mysterious, noir photography",
                    f"Film noir: dramatic reveal, noir lighting, ominous mood",
                    f"Film noir: final image, dramatic shadows, noir aesthetic"
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
            
            print(f"\n✅ YOUTUBE SHORTS SCRIPT GENERATED")
            print(f"   Title: {data['title']}")
            print(f"   Hook: {data['hook']}")
            print(f"   Words: {data['word_count']}")
            print(f"   Duration: ~{data['estimated_duration']:.1f}s")
            
            if data['estimated_duration'] <= OPTIMAL_MAX_DURATION:
                print(f"   ✅ OPTIMAL for YouTube Shorts!")
            elif data['estimated_duration'] <= ABSOLUTE_MAX_DURATION:
                print(f"   ⚠️ Longer than optimal but within limits")
            
            break
            
        except json.JSONDecodeError as e:
            print(f"❌ Attempt {attempt} failed: JSON parse error - {e}")
            if attempt < max_attempts:
                print(f"   Retrying in {2**attempt} seconds...")
                import time
                time.sleep(2**attempt)
        
        except ValueError as e:
            print(f"❌ Attempt {attempt} failed: {e}")
            if attempt < max_attempts:
                print(f"   Retrying...")
        
        except Exception as e:
            print(f"❌ Attempt {attempt} failed: {type(e).__name__} - {e}")
            if attempt < max_attempts:
                print(f"   Retrying in {2**attempt} seconds...")
                import time
                time.sleep(2**attempt)
        
        if attempt >= max_attempts:
            print("\n⚠️ Max attempts reached, using fallback...")
            data = get_fallback_script(content_type, mystery_type)
            
            fallback_hash = get_content_hash(data)
            save_to_history(data['topic'], fallback_hash, data['title'], data)
    
    script_path = os.path.join(TMP, "script.json")
    with open(script_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Saved script to {script_path}")
    
    script_text_path = os.path.join(TMP, "script.txt")
    full_script = data['script']
    
    with open(script_text_path, "w", encoding="utf-8") as f:
        f.write(full_script)
    
    print(f"💾 Saved script text to {script_text_path}")
    
    print(f"\n{'='*70}")
    print(f"📊 YOUTUBE SHORTS SUMMARY")
    print(f"{'='*70}")
    print(f"Words: {data['word_count']}")
    print(f"Estimated duration: {data['estimated_duration']:.1f}s")
    print(f"Target range: {OPTIMAL_MIN_DURATION}-{OPTIMAL_MAX_DURATION}s")
    
    if data['estimated_duration'] <= OPTIMAL_MAX_DURATION:
        print(f"✅ Perfect for YouTube Shorts!")
    elif data['estimated_duration'] <= 90:
        print(f"⚠️ Slightly long - consider trimming for better performance")
    elif data['estimated_duration'] <= ABSOLUTE_MAX_DURATION:
        print(f"⚠️ At extended limit - may need speed adjustment")
    else:
        print(f"🚨 WARNING: Exceeds Shorts limit!")
    
    print(f"Total history: {len(history['topics'])} topics")
    
    return data


if __name__ == '__main__':
    try:
        generate_mystery_script()
        print("\n✅ YouTube Shorts mystery script complete!")
    except Exception as e:
        print(f"\n❌ FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        exit(1)