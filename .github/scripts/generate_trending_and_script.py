#!/usr/bin/env python3
"""
🔍 Generate Mystery Script - YOUTUBE SHORTS OPTIMIZED
YouTube Shorts: MAX 60 seconds (official limit)
Optimal: 45-60 seconds for best performance

FIXED IMPROVEMENTS:
✅ Stricter word count validation
✅ Better error messages
✅ Fallback script always within limits
✅ Clear duration warnings
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

# 🎯 YOUTUBE SHORTS DURATION TARGETS - STRICT
OPTIMAL_MIN_DURATION = 45   # Sweet spot minimum
OPTIMAL_MAX_DURATION = 60   # YouTube Shorts official max (HARD LIMIT)

# TTS Reading speed: ~2.1 words/second (mystery narrator at 0.85x speed)
WORDS_PER_SECOND = 2.1

# ✅ STRICT WORD COUNT TARGETS
OPTIMAL_MIN_WORDS = int(OPTIMAL_MIN_DURATION * WORDS_PER_SECOND)      # ~95 words for 45s
OPTIMAL_MAX_WORDS = int(OPTIMAL_MAX_DURATION * WORDS_PER_SECOND)      # ~126 words for 60s
HARD_LIMIT_WORDS = int((OPTIMAL_MAX_DURATION + 2) * WORDS_PER_SECOND) # ~129 words for 62s (allow 2s buffer)

print(f"🎯 YouTube Shorts Target (STRICT):")
print(f"   ✅ Optimal: {OPTIMAL_MIN_DURATION}-{OPTIMAL_MAX_DURATION}s ({OPTIMAL_MIN_WORDS}-{OPTIMAL_MAX_WORDS} words)")
print(f"   ⚠️ Hard Limit: {HARD_LIMIT_WORDS} words ({(HARD_LIMIT_WORDS/WORDS_PER_SECOND):.0f}s)")
print(f"   🚨 REJECT: Over {HARD_LIMIT_WORDS} words (exceeds 62s)")

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
        'word_count': script_data.get('word_count', 0),
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
    """✅ FIXED: Strict YouTube Shorts validation"""
    
    required_fields = ["title", "topic", "hook", "script", "cta"]
    
    for field in required_fields:
        if field not in data:
            raise ValueError(f"❌ Missing required field: {field}")
    
    if not isinstance(data["script"], str):
        raise ValueError("❌ Script must be a string (narrative text)")
    
    # ✅ STRICT WORD COUNT VALIDATION
    word_count = len(data["script"].split())
    estimated_duration = word_count / WORDS_PER_SECOND
    
    print(f"\n📊 Script Length Analysis:")
    print(f"   Words: {word_count}")
    print(f"   Estimated duration: {estimated_duration:.1f}s")
    print(f"   Target: {OPTIMAL_MIN_DURATION}-{OPTIMAL_MAX_DURATION}s")
    
    # ✅ REJECT IF TOO SHORT
    if word_count < OPTIMAL_MIN_WORDS:
        print(f"   ❌ TOO SHORT: {word_count} words = {estimated_duration:.1f}s")
        raise ValueError(f"Script too short: {word_count} words (need {OPTIMAL_MIN_WORDS}-{OPTIMAL_MAX_WORDS})")
    
    # ✅ REJECT IF TOO LONG (HARD LIMIT)
    if word_count > HARD_LIMIT_WORDS:
        print(f"   ❌ TOO LONG: {word_count} words = {estimated_duration:.1f}s")
        print(f"   ⚠️ Exceeds YouTube Shorts limit!")
        raise ValueError(f"Script too long: {word_count} words (max {HARD_LIMIT_WORDS})")
    
    # ✅ OPTIMAL RANGE
    if word_count <= OPTIMAL_MAX_WORDS:
        print(f"   ✅ PERFECT: {word_count} words = {estimated_duration:.1f}s (OPTIMAL)")
    else:
        print(f"   ⚠️ ACCEPTABLE: {word_count} words = {estimated_duration:.1f}s (slightly over optimal)")
        print(f"   💡 Recommendation: Trim to {OPTIMAL_MAX_WORDS} words for better performance")
    
    # Store metadata
    data["estimated_duration"] = estimated_duration
    data["word_count"] = word_count
    
    # Validate title
    if len(data["title"]) > 100:
        print(f"⚠️ Title too long ({len(data['title'])} chars), truncating...")
        data["title"] = data["title"][:97] + "..."
    
    # Validate hook
    hook_words = len(data["hook"].split())
    if hook_words > 15:
        print(f"⚠️ Hook too long ({hook_words} words), keep under 12 words")
    
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


def get_content_type_guidance(content_type):
    """Get specific guidance for mystery content type"""
    guidance = {
        'evening_prime': """
EVENING PRIME FOCUS (7-9 PM):
- Target: Evening viewers unwinding, ready for intrigue
- Mystery type: Famous mysteries, high engagement hooks
- Tone: Accessible, intriguing, documentary-style
- Examples: Flight 19, DB Cooper, Zodiac Killer, MH370
- DURATION: 45-60 seconds (strict YouTube Shorts limit)
""",
        'late_night': """
LATE NIGHT FOCUS (10 PM - 2 AM):
- Target: Can't sleep, scrolling in bed
- Mystery type: Darker, more disturbing mysteries
- Tone: Chilling, serious, thought-provoking
- Examples: Dyatlov Pass, Elisa Lam, Somerton Man
- DURATION: 45-60 seconds (strict YouTube Shorts limit)
""",
        'weekend_binge': """
WEEKEND BINGE FOCUS (Sat/Sun 8-11 PM):
- Target: More time, want depth
- Mystery type: Complex historical mysteries
- Tone: Documentary deep-dive
- Examples: Voynich Manuscript, Antikythera Mechanism
- DURATION: 45-60 seconds (YouTube Shorts limit)
""",
        'general': """
GENERAL MYSTERY FOCUS:
- Target: General mystery enthusiasts
- Mystery type: Balanced mix
- Tone: Mysterious but accessible
- Examples: Bermuda Triangle, Oak Island, Roanoke
- DURATION: 45-60 seconds (strict YouTube Shorts limit)
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
- LENGTH: Strict 45-60s YouTube Shorts
""",
        'crime': """
TRUE CRIME MYSTERIES:
- Focus: Unsolved murders, cold cases
- Hook formula: "A body was found in [Location]. What investigators discovered was impossible."
- Key elements: Evidence that doesn't add up, multiple theories
- Examples: Zodiac Killer, Black Dahlia, Somerton Man
- ETHICAL: Focus on mystery aspect, respect victims
- LENGTH: Strict 45-60s YouTube Shorts
""",
        'historical': """
HISTORICAL ENIGMAS:
- Focus: Ancient artifacts, unexplained discoveries
- Hook formula: "In [Year], they discovered [Object]. Scientists still can't explain it."
- Key elements: Technology that shouldn't exist
- Examples: Antikythera Mechanism, Voynich Manuscript, Göbekli Tepe
- LENGTH: Strict 45-60s YouTube Shorts
""",
        'conspiracy': """
DECLASSIFIED CONSPIRACIES:
- Focus: Proven conspiracies with documents
- Hook formula: "Declassified documents prove [Operation] was real."
- Key elements: Government documents, verified facts
- Examples: MKUltra, Operation Northwoods, Tuskegee
- WARNING: ONLY proven conspiracies
- LENGTH: Strict 45-60s YouTube Shorts
"""
    }
    return types.get(mystery_type, types['disappearance'])


def build_mystery_prompt(content_type, priority, mystery_type, trends, history):
    """Build the mystery script generation prompt (YOUTUBE SHORTS OPTIMIZED)"""
    
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
    
    prompt = f"""You are an expert mystery storyteller creating YOUTUBE SHORTS.

🎯 CRITICAL YOUTUBE SHORTS REQUIREMENTS (STRICT ENFORCEMENT):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TARGET DURATION: {OPTIMAL_MIN_DURATION}-{OPTIMAL_MAX_DURATION} SECONDS (HARD LIMIT)
TARGET WORDS: {OPTIMAL_MIN_WORDS}-{OPTIMAL_MAX_WORDS} words
ABSOLUTE MAXIMUM: {HARD_LIMIT_WORDS} words ({(HARD_LIMIT_WORDS/WORDS_PER_SECOND):.0f}s)

TTS Speed: {WORDS_PER_SECOND} words/second
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CONTEXT:
- Content type: {content_type}
- Mystery type: {mystery_type}
- Priority: {priority}

PREVIOUSLY COVERED (DO NOT REPEAT):
{chr(10).join(f"  • {t}" for t in previous_topics[-15:]) if previous_topics else '  None yet'}

{trending_mandate}

{content_type_guidance}

{mystery_type_guidance}

SCRIPT STRUCTURE:

🎬 HOOK (0-5 seconds, 8-12 words):
Formula: "[Date], [Location]. [Vanished/Found/Discovered]. [Impossible detail]."
Example: "December 5th, 1945. Five planes vanished. No wreckage found."

🎬 SETUP (15-20 words):
WHO, WHAT, WHEN, WHERE

🎬 INCIDENT (35-45 words):
What happened. Keep sentences SHORT. Build tension FAST.

🎬 CONTRADICTION (30-40 words):
"But here's where it gets strange..."
Stack impossibilities. Facts that don't add up.

🎬 TWIST (20-30 words):
"The most terrifying part?"
Save best detail for last.

🎬 CLIFFHANGER (10-15 words):
"To this day..." Open question.

MANDATORY REQUIREMENTS:
✅ Write as FLOWING NARRATIVE (conversational)
✅ SHORT PUNCHY SENTENCES (6-10 words each)
✅ NO bullet points - paragraph breaks only (\\n\\n)
✅ Specific details (dates, names, numbers)
✅ Build to impossible contradiction
✅ Leave mystery UNSOLVED
✅ STAY UNDER {HARD_LIMIT_WORDS} WORDS (MAXIMUM)
✅ Use \\n\\n between paragraphs

AVOID:
❌ Going over {HARD_LIMIT_WORDS} words (automatic rejection!)
❌ Long explanations or setup
❌ Generic descriptions
❌ Lists or bullet points
❌ Unverified theories
❌ Special characters in JSON
❌ "Today I'll tell you..." (wastes seconds!)

OUTPUT FORMAT (JSON ONLY):
{{
  "title": "The [Mystery]: [Intriguing Statement]",
  "topic": "mystery",
  "hook": "[8-12 words max]",
  "script": "[Full narrative - {OPTIMAL_MIN_WORDS} to {HARD_LIMIT_WORDS} words with \\n\\n paragraphs]",
  "cta": "[Question under 12 words]",
  "hashtags": ["#mystery", "#unsolved", "#truecrime", "#shorts"],
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

EXAMPLE (STRICT 60-SECOND LIMIT):
{{
  "title": "Flight 19: Vanished Without Trace",
  "topic": "mystery",
  "hook": "December 5th, 1945. Five planes vanished. No wreckage.",
  "script": "Five torpedo bombers took off from Fort Lauderdale. Routine training mission. Fourteen experienced crew. Perfect weather.\\n\\nTwo hours later, Lieutenant Taylor radioed controllers. 'We can't find west. Everything looks wrong.' All five planes. Compasses malfunctioning. Simultaneously.\\n\\nThe Navy launched history's biggest search. Two hundred forty thousand square miles. Three hundred aircraft. Five straight days.\\n\\nBut here's where it gets strange. Zero debris. No oil slicks. No wreckage. Nothing. Five massive planes. Gone. As if they never existed.\\n\\nThe most terrifying part? The rescue plane sent to find them? Also vanished. Same night. Thirteen more crew. No distress call.\\n\\nTo this day, twenty-seven men and six aircraft. Zero evidence. The Navy's conclusion: 'as if they flew to Mars.'",
  "cta": "What do you think happened to them?",
  "hashtags": ["#flight19", "#bermudatriangle", "#mystery", "#unsolved", "#shorts"],
  "description": "December 5, 1945: Flight 19 disappeared. The rescue plane vanished too. 27 men, 6 aircraft, zero evidence.",
  "key_phrase": "FLIGHT 19",
  "mystery_category": "disappearance",
  "visual_prompts": [
    "Film noir: vintage 1940s torpedo bombers in formation over ocean, dramatic clouds, noir aesthetic",
    "Film noir: old military radio equipment, dramatic lighting, noir documentary style",
    "Film noir: vast empty ocean aerial view, foggy and mysterious, no wreckage",
    "Film noir: vintage search map with red pins, classified document aesthetic"
  ]
}}

🚨 CRITICAL: STAY UNDER {HARD_LIMIT_WORDS} WORDS OR IT WILL BE REJECTED!

Generate the mystery story NOW.
"""

    return prompt


def get_fallback_script(content_type, mystery_type):
    """Fallback scripts - ALL guaranteed under 60 seconds"""
    
    fallback_scripts = {
        'evening_prime': {
            'title': "Flight 19: Vanished Without Trace",
            'hook': "December 5th, 1945. Five planes vanished. No wreckage.",
            'script': """Five torpedo bombers took off from Fort Lauderdale. Routine training mission. Fourteen experienced crew. Perfect weather.

Two hours later, Lieutenant Taylor radioed controllers. 'We can't find west. Everything looks wrong.' All five planes. Compasses malfunctioning. Simultaneously.

The Navy launched history's biggest search. Two hundred forty thousand square miles. Three hundred aircraft. Five straight days.

But here's where it gets strange. Zero debris. No oil slicks. No wreckage. Nothing. Five massive planes. Gone.

The most terrifying part? The rescue plane sent to find them? Also vanished. Same night. Thirteen more crew.

To this day, twenty-seven men and six aircraft. Zero evidence.""",
            'key_phrase': "FLIGHT 19",
            'mystery_category': 'disappearance'
        },
        
        'late_night': {
            'title': "Dyatlov Pass: Nine Dead Hikers",
            'hook': "February 1959. Nine hikers died. Explanation impossible.",
            'script': """Nine experienced hikers. Ural Mountains. February 2nd, 1959. They set up camp on Kholat Syakhl. Dead Mountain.

They never came back.

Search teams found their tent February 26th. Ripped open. From the inside. Boots still inside. Supplies untouched. Hikers? Gone.

Bodies found at forest edge. Barefoot. Freezing temperatures. Dead from hypothermia. Then three more in a ravine. Fractured skulls. Broken ribs.

But here's where it gets strange. Injuries equivalent to a car crash. But no external wounds. No signs of struggle.

The most terrifying part? Witnesses reported glowing orange orbs in the sky that night. The Soviet investigation concluded: unknown force.

To this day, no explanation exists.""",
            'key_phrase': "DYATLOV PASS",
            'mystery_category': 'crime'
        },
        
        'weekend_binge': {
            'title': "Voynich: The Unreadable Book",
            'hook': "A 600-year-old book. Written in unknown language.",
            'script': """1912. Italian villa. Book dealer Wilfrid Voynich discovered a medieval manuscript. Two hundred forty pages. Bizarre illustrations. Plants that don't exist. Impossible astronomical charts.

Carbon dating confirmed it. Early 15th century. Over six hundred years old. Written on real vellum. Medieval ink. Not a hoax.

World War Two codebreakers tried decoding it. The NSA tried. Computer algorithms failed. Text follows language patterns. Grammar. Structure. But words? Complete gibberish.

But here's where it gets strange. Plants don't match any known species. Impossible hybrids. Star charts show constellations that don't exist.

The most terrifying part? Linguistic analysis proves it's not random. Too much structure. Too much consistency. Someone spent years writing this.

After six hundred years and countless experts, no one can read a single word.""",
            'key_phrase': "VOYNICH MANUSCRIPT",
            'mystery_category': 'historical'
        },
        
        'general': {
            'title': "Bermuda Triangle: Where Ships Vanish",
            'hook': "Hundreds of ships. All vanished without trace.",
            'script': """The Bermuda Triangle. Miami to Bermuda to Puerto Rico. Past century? Fifty ships. Twenty aircraft. Disappeared. No wreckage. No signals. Gone.

December 1945. Flight 19. Five bombers vanished. The rescue plane? Also disappeared. March 1918. USS Cyclops. Three hundred nine crew. Vanished. No SOS.

Weather doesn't explain it. No more storms than elsewhere. Magnetic anomalies? Normal. Methane gas? No evidence. Rogue waves? They don't erase radio signals.

But here's where it gets strange. Ships lose radio contact suddenly. No mayday. No beacons. Just silence. Search teams find nothing. No debris. No oil. As if vessels ceased to exist.

The most terrifying part? It still happens. 2015. El Faro disappeared. Thirty-three crew. Modern systems. Satellite comms. Recording ends abruptly. Mid-transmission.

What happens in the Bermuda Triangle?""",
            'key_phrase': "BERMUDA TRIANGLE",
            'mystery_category': 'disappearance'
        }
    }
    
    selected = fallback_scripts.get(content_type, fallback_scripts['evening_prime'])
    
    print(f"📋 Using fallback mystery script: {selected['title']}")
    
    word_count = len(selected['script'].split())
    estimated_duration = word_count / WORDS_PER_SECOND
    
    print(f"   ✅ Fallback: {word_count} words = {estimated_duration:.1f}s (WITHIN LIMIT)")
    
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
    
    history = load_history()
    trends = load_trending()
    
    if trends:
        print(f"✅ Loaded trending data from {trends.get('source', 'unknown')}")
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
        print(f"   Auto-selected: {mystery_type}")
    
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
            print(f"   Duration: ~{data['estimated_duration']:.1f}s ✅")
            
            break
            
        except json.JSONDecodeError as e:
            print(f"❌ Attempt {attempt} failed: JSON parse error")
            if attempt < max_attempts:
                import time
                time.sleep(2**attempt)
        
        except ValueError as e:
            print(f"❌ Attempt {attempt} failed: {e}")
        
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
    print(f"✅ YOUTUBE SHORTS SCRIPT READY")
    print(f"{'='*70}")
    print(f"Words: {data['word_count']}")
    print(f"Duration: {data['estimated_duration']:.1f}s")
    if data['estimated_duration'] <= OPTIMAL_MAX_DURATION:
        print(f"Status: ✅ PERFECT FOR YOUTUBE SHORTS")
    else:
        print(f"Status: ⚠️ ACCEPTABLE (slightly over optimal)")
    
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