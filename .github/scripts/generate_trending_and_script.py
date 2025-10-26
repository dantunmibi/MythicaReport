#!/usr/bin/env python3
"""
🔍 Generate Mystery Script - ROBUST VERSION
Creates narrative mystery scripts with:
- History tracking (avoid duplicates)
- Content validation
- Retry logic with exponential backoff
- Trending topic enforcement
- Time-optimized content
- FULL NARRATIVE FLOW (not bullet points)
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

# Configure Gemini with model selection
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
    # Hash based on title + hook + script (changed from bullets)
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
    
    for idx, prev_title in enumerate(reversed(previous_titles[-30:])):  # Check last 30
        prev_words = set(prev_title.lower().split())
        
        intersection = len(new_words & prev_words)
        union = len(new_words | prev_words)
        
        if union > 0:
            base_similarity = intersection / union
            
            # Decay factor: older topics matter less
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
        return True  # No validation if no trending data
    
    # Changed: use 'script' field instead of 'bullets'
    script_text = f"{script_data['title']} {script_data['hook']} {script_data.get('script', '')}".lower()
    
    # Extract keywords from trending topics
    trend_keywords = []
    for topic in trending_topics:
        # Remove common filler words
        words = [w for w in topic.lower().split() if len(w) > 4 and w not in {
            'this', 'that', 'with', 'from', 'will', 'just', 'your', 'they',
            'them', 'what', 'when', 'where', 'which', 'while', 'about',
            'have', 'been', 'were', 'their', 'there', 'these', 'those',
            'make', 'made', 'take', 'took', 'very', 'more', 'most', 'some',
            'other', 'into', 'than', 'then', 'here'
        }]
        trend_keywords.extend(words)
    
    # Remove duplicates
    trend_keywords = list(set(trend_keywords))
    
    # Check for keyword matches
    matches = sum(1 for kw in trend_keywords if kw in script_text)
    
    # Need at least 2 keyword matches
    if matches < 2:
        print(f"⚠️ Script doesn't use trending topics! Only {matches} matches.")
        print(f"   Keywords: {trend_keywords[:10]}")
        return False
    
    print(f"✅ Script uses trending topics ({matches} keyword matches)")
    return True


def validate_script_data(data):
    """Validate generated script has all required fields (MYSTERY VERSION)"""
    
    # Changed: 'bullets' → 'script'
    required_fields = ["title", "topic", "hook", "script", "cta"]
    
    for field in required_fields:
        if field not in data:
            raise ValueError(f"Missing required field: {field}")
    
    # Validate script is a string (not list like bullets)
    if not isinstance(data["script"], str):
        raise ValueError("script must be a string (narrative text)")
    
    # Validate word count (mystery needs 350-450 words for 60-90 second story)
    word_count = len(data["script"].split())
    if word_count < 300:
        raise ValueError(f"Script too short: {word_count} words (need 300-500)")
    if word_count > 550:
        print(f"⚠️ Script a bit long ({word_count} words), but acceptable")
    
    # Validate script has mystery structure elements (flexible check)
    if "But here's where it gets strange" not in data["script"] and \
       "here's what makes it" not in data["script"].lower() and \
       "the most terrifying part" not in data["script"].lower():
        print("⚠️ Warning: Script may be missing contradiction/twist transition phrase")
    
    # Validate title length
    if len(data["title"]) > 100:
        print(f"⚠️ Title too long ({len(data['title'])} chars), truncating...")
        data["title"] = data["title"][:97] + "..."
    
    # Validate hook length (mystery hooks should be short and punchy)
    hook_words = len(data["hook"].split())
    if hook_words > 12:
        print(f"⚠️ Hook too long ({hook_words} words), ideally ≤10 words")
    
    print(f"✅ Script validation passed ({word_count} words)")
    return True


def extract_json_from_response(raw_text):
    """Extract JSON from Gemini response (handles various formats)"""
    # Try to find JSON in code blocks
    json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', raw_text, re.DOTALL)
    if json_match:
        print("✅ Found JSON in code block")
        return json_match.group(1)
    
    # Try to find raw JSON
    json_match = re.search(r'\{.*\}', raw_text, re.DOTALL)
    if json_match:
        print("✅ Found raw JSON")
        return json_match.group(0)
    
    raise ValueError("No JSON found in response")


def clean_script_text(text):
    """Clean script text of problematic characters"""
    # Remove smart quotes
    text = text.replace('"', '').replace('"', '')
    text = text.replace(''', "'").replace(''', "'")
    
    # Remove other problematic characters
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
- Goal: Hook them for the binge session
""",
        'late_night': """
LATE NIGHT FOCUS (10 PM - 2 AM):
- Target: Can't sleep, scrolling in bed, want something unsettling
- Mystery type: Darker, more disturbing mysteries
- Tone: Chilling, serious, thought-provoking
- Examples: Dyatlov Pass, Elisa Lam, Somerton Man, Black Dahlia
- Energy: Eerie, unsettling, keeps them thinking
- Goal: Create "can't stop watching" effect
""",
        'weekend_binge': """
WEEKEND BINGE FOCUS (Sat/Sun 8-11 PM):
- Target: More time, want depth, ready for rabbit holes
- Mystery type: Complex, layered historical mysteries
- Tone: Documentary deep-dive, detailed
- Examples: Voynich Manuscript, Antikythera Mechanism, Göbekli Tepe
- Energy: Intellectually satisfying, detailed
- Goal: Satisfy deep curiosity, encourage playlist watching
""",
        'general': """
GENERAL MYSTERY FOCUS:
- Target: General mystery enthusiasts, curious viewers
- Mystery type: Balanced mix of disappearances and crimes
- Tone: Mysterious but accessible, serious but not too dark
- Examples: Bermuda Triangle, Oak Island, Roanoke Colony
- Energy: Intriguing, engaging, binge-worthy
- Goal: Broad appeal, high shareability
"""
    }
    return guidance.get(content_type, guidance['general'])


def get_mystery_type_guidance(mystery_type):
    """Get guidance for mystery category"""
    types = {
        'disappearance': """
DISAPPEARANCE MYSTERIES:
- Focus: Vanishing without a trace, impossible disappearances
- Hook formula: "[Date], [Location]. [Person/Group] vanished. [Impossible detail]."
- Key elements: Last known location, search efforts, zero evidence
- Emphasis: The impossibility of disappearing completely
- Examples: Flight 19, DB Cooper, Mary Celeste, Amelia Earhart
""",
        'crime': """
TRUE CRIME MYSTERIES:
- Focus: Unsolved murders, ciphers, cold cases
- Hook formula: "A body was found in [Location]. What investigators discovered was impossible."
- Key elements: Evidence that doesn't add up, multiple theories, ongoing investigation
- Emphasis: The mystery, NOT graphic details (be respectful)
- Examples: Zodiac Killer, Black Dahlia, Jack the Ripper, Somerton Man
- ETHICAL: Focus on puzzle/mystery aspect, respect victims
""",
        'historical': """
HISTORICAL ENIGMAS:
- Focus: Ancient artifacts, unexplained discoveries, anachronisms
- Hook formula: "In [Year], they discovered [Object]. Scientists still can't explain it."
- Key elements: Technology/knowledge that shouldn't exist, archaeological mysteries
- Emphasis: The "how did they do this?" factor
- Examples: Antikythera Mechanism, Voynich Manuscript, Göbekli Tepe, Nazca Lines
""",
        'conspiracy': """
DECLASSIFIED CONSPIRACIES:
- Focus: Proven conspiracies with declassified documents
- Hook formula: "Declassified documents prove [Operation] was real."
- Key elements: Government documents, verified facts, shocking revelations
- Emphasis: ONLY proven conspiracies with evidence
- Examples: MKUltra, Operation Northwoods, Tuskegee Experiment
- WARNING: NO unverified theories, stick to facts only
"""
    }
    return types.get(mystery_type, types['disappearance'])

def build_mystery_prompt(content_type, priority, mystery_type, trends, history):
    """Build the comprehensive mystery script generation prompt"""
    
    # Get previous topics for context
    previous_topics = [
        f"{t.get('topic', 'unknown')}: {t.get('title', '')}" 
        for t in history['topics'][-20:]  # Last 20
    ]
    previous_titles = [t.get('title', '') for t in history['topics'][-30:]]  # Last 30
    
    # Extract trending topics
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
    
    # Build trending mandate
    if trending_topics:
        trending_mandate = f"""
⚠️⚠️⚠️ CRITICAL MANDATORY REQUIREMENT ⚠️⚠️⚠️

YOU MUST CREATE A SCRIPT ABOUT ONE OF THESE REAL TRENDING MYSTERY TOPICS:

{chr(10).join(trending_summaries)}

These are REAL trending mysteries from today ({datetime.now().strftime('%Y-%m-%d %H:%M')}) collected from:
- Google Trends (real mystery search data)
- Reddit communities (r/UnresolvedMysteries, r/TrueCrime)
- YouTube trending (viral mystery content)
- Evergreen themes (proven viral mysteries)

YOU MUST PICK ONE OF THE 5 MYSTERIES ABOVE AND TURN IT INTO A COMPELLING NARRATIVE.
DO NOT create content about anything else.
DO NOT make up your own mystery.
USE THE EXACT MYSTERY, including the suggested hook and contradiction.

If the trend is "Flight 19", your script MUST be about that disappearance.
If the trend is "Zodiac Killer", your script MUST be about that case.
"""
    else:
        trending_mandate = "⚠️ No trending data available - create original mystery content from evergreen topics\n"
    
    # Get content type and mystery type guidance
    content_type_guidance = get_content_type_guidance(content_type)
    mystery_type_guidance = get_mystery_type_guidance(mystery_type)
    
    # Build the main prompt
    prompt = f"""You are an expert mystery storyteller in the style of MrBallen and Unsolved Mysteries.

CONTEXT:
- Current date: {datetime.now().strftime('%Y-%m-%d')}
- Time: {datetime.now().strftime('%I:%M %p')}
- Content type: {content_type}
- Mystery type: {mystery_type}
- Priority: {priority}

PREVIOUSLY COVERED (DO NOT REPEAT):
{chr(10).join(f"  • {t}" for t in previous_topics[-15:]) if previous_topics else '  None yet'}

{trending_mandate}

TASK: Create a compelling 60-90 second NARRATIVE MYSTERY STORY for YouTube Shorts.

VIDEO STYLE: STORY-BASED NARRATIVE (NOT educational facts or lists)

{content_type_guidance}

{mystery_type_guidance}

CRITICAL STRUCTURE (3 Acts - NARRATIVE FLOW):

ACT 1 - THE HOOK (0-8 seconds, ~15 words max):
Formula: "[Specific date], [Location]. [Impossible event]. [Intriguing detail]."

Examples:
✅ "December 5th, 1945, Fort Lauderdale. Five torpedo bombers vanished mid-flight. No wreckage was ever found."
✅ "1587, Roanoke Island. An entire colony of 115 people disappeared overnight. Only one word remained: CROATOAN."
✅ "October 21st, 1978. Frederick Valentich radioed: It's not an aircraft. Then vanished forever."

ACT 2 - THE MYSTERY (8-75 seconds, ~300 words):

SETUP (15 seconds, ~40 words):
- WHO: Introduce subject with specific names
- WHAT: The normal situation before the mystery
- WHERE: Specific location with details
- WHEN: Exact date and time if possible
- Establish this was routine/expected

Example: "Frederick Valentich was an experienced pilot. October 21st, 1978. Seven PM. He took off from Melbourne in a Cessna 182. Routine training flight. Clear skies. Perfect visibility. Twenty minutes into the flight, everything changed."

INCIDENT (25 seconds, ~70 words):
- What happened that's unexplainable
- Short sentences for impact
- Build tension with pacing
- Include direct quotes if available
- The moment things became impossible

Example: "Valentich radioed air traffic control. 'There's an aircraft above me.' Controllers saw nothing on radar. 'It's not an aircraft. It's—' Static. Controllers asked him to identify it. Valentich's final words: 'It's hovering. It's not an aircraft.' Then seventeen seconds of metallic scraping sounds. Silence."

CONTRADICTION (20 seconds, ~55 words):
- MUST start with: "But here's where it gets strange..." OR "The most terrifying part?" OR "What investigators found next..."
- Present evidence that contradicts normal explanations
- Stack impossibilities
- Facts that don't make sense together
- Create cognitive dissonance

Example: "But here's where it gets strange. The weather was perfect. No storms. No mechanical issues. The plane was well-maintained. Valentich had over 150 flight hours. Search teams scanned 1,000 square miles. For five years. Not a single piece of wreckage. No oil slicks. No debris. Nothing."

TWIST (15 seconds, ~40 words):
- Save the MOST chilling detail for last
- "The most terrifying part?" or "What investigators found next..."
- The detail that makes it truly unexplainable
- The thing that still haunts investigators

Example: "The most terrifying part? In 2014, a partial engine cowl washed ashore. Analysis confirmed: it was Valentich's plane. But the damage pattern didn't match a crash. Something had torn it apart. In mid-air."

ACT 3 - CLIFFHANGER (75-90 seconds, ~25 words):
- "To this day..." statement
- Unanswered question
- Prompt for viewer theories
- Create FOMO for engagement

Example: "To this day, no one knows what Frederick Valentich saw that night. The recording still exists. The Australian government can't explain it. What do YOU think he encountered?"

NARRATIVE REQUIREMENTS:
✅ Write as FLOWING NARRATIVE STORY (like telling a story to a friend)
✅ NO bullet points or lists in the script
✅ Specific dates, names, numbers (not "a plane" but "five torpedo bombers")
✅ Short sentences (8-15 words maximum per sentence)
✅ Vary sentence length for rhythm and impact
✅ Serious, mysterious tone (NOT excited, NOT cheerful)
✅ Build contradictions naturally ("Perfect weather... yet they vanished")
✅ Leave mystery UNSOLVED (don't provide definitive answer)
✅ 350-450 words total (will be read at 0.85 speed = 60-90 seconds)
✅ Proper punctuation for TTS pauses (periods = long pause, commas = short)
✅ Use paragraph breaks (\\n\\n) for natural pauses between sections

AVOID:
❌ "Today I'll tell you about..." (too educational)
❌ "This is crazy!" or "Unbelievable!" (show don't tell)
❌ Lists/bullet points in narration
❌ Spoiling the mystery upfront
❌ "Have you heard of..." (weak hook)
❌ Explaining how weird it is (let facts speak)
❌ Unverified conspiracy theories
❌ Graphic violence descriptions
❌ Active missing persons cases (ethical concern)
❌ Special characters or quotes in JSON output (breaks parsing)

VISUAL STYLE FOR IMAGE PROMPTS:
Every visual prompt MUST include: "film noir photography, high contrast black and white, 
dramatic shadows, moody atmosphere, vintage 1940s-1960s aesthetic, film grain, mysterious, 
unsettling, documentary style, cinematic lighting, foggy, dark, noir aesthetic"

OUTPUT FORMAT (JSON ONLY - NO OTHER TEXT BEFORE OR AFTER):
{{
  "title": "The [Mystery Name]: [Intriguing Statement] (under 70 chars)",
  "topic": "mystery",
  "hook": "[Maximum 8-10 words for first 8 seconds]",
  "script": "[Full narrative 350-450 words written as flowing paragraphs with proper punctuation for TTS pauses. Use \\n\\n for paragraph breaks between Setup, Incident, Contradiction, and Twist sections.]",
  "cta": "[Question to spark theories, under 15 words]",
  "hashtags": ["#mystery", "#unsolved", "#truecrime", "#shorts"],
  "description": "[2-3 sentences with specific details for YouTube SEO, include dates and names]",
  "key_phrase": "[3-5 words for thumbnail text, usually mystery name in CAPS]",
  "mystery_category": "{mystery_type}",
  "visual_prompts": [
    "Film noir: [specific scene for hook], dark moody lighting, vintage photography, high contrast, dramatic shadows, mysterious atmosphere, documentary style, noir aesthetic",
    "Film noir: [specific scene for setup], shadowy, foggy, unsettling, cinematic lighting, 1940s aesthetic, film grain, mysterious mood, noir photography",
    "Film noir: [specific scene for incident], dramatic reveal, dark and mysterious, vintage documentary style, high contrast black and white, ominous lighting",
    "Film noir: [specific scene for twist], film grain, noir aesthetic, unsettling atmosphere, dramatic shadows, mysterious final image, documentary photography"
  ]
}}

EXAMPLE OUTPUT (Flight 19):
{{
  "title": "Flight 19: The Disappearance That Defies Physics",
  "topic": "mystery",
  "hook": "December 5th 1945. Five planes vanished. No evidence found.",
  "script": "December 5th, 1945. Fort Lauderdale Naval Air Station. Five torpedo bombers lifted off at two PM. Routine training mission. Flight 19. Fourteen experienced crew members. Lieutenant Charles Taylor leading. Perfect weather. Visibility: fifteen miles.\\n\\nTwo hours later, Taylor's voice crackled through the radio. 'We can't find west. Everything is wrong. Even the ocean doesn't look right.' Controllers tried to guide them back. Taylor's compass was malfunctioning. Both compasses. On all five planes. Simultaneously.\\n\\nThe Navy launched the largest search in history. Two hundred forty thousand square miles. Three hundred aircraft. Five days of continuous searching.\\n\\nBut here's where it gets strange. No wreckage. No oil slicks. No debris. No bodies. Not a single piece of metal. Five massive aircraft. Simply gone. As if they never existed.\\n\\nThe most terrifying part? A rescue plane was sent to find them. A Martin Mariner. Thirteen crew members. It also vanished. Same night. Same ocean. No distress call. No explanation.\\n\\nSix aircraft. Twenty-seven men. Zero evidence. The Navy's official report concluded: 'as if they flew to Mars.'\\n\\nTo this day, Flight 19 remains the only mass disappearance in aviation history with absolutely no physical evidence.",
  "cta": "What do you think happened to them?",
  "hashtags": ["#flight19", "#bermudatriangle", "#mystery", "#unsolved", "#truecrime", "#shorts"],
  "description": "On December 5, 1945, Flight 19 disappeared over the Bermuda Triangle with 14 crew members. A rescue plane sent to find them also vanished the same night. 27 men, 6 aircraft, zero physical evidence. The Navy's conclusion: 'as if they flew to Mars.' What really happened?",
  "key_phrase": "FLIGHT 19",
  "mystery_category": "disappearance",
  "visual_prompts": [
    "Film noir: vintage 1940s US Navy TBM Avenger torpedo bombers flying in formation over ocean, dark stormy skies, black and white photography, ominous atmosphere, dramatic clouds, high contrast, mysterious mood, documentary style, noir aesthetic",
    "Film noir: old 1940s military radio equipment crackling with static, dramatic side lighting, noir documentary style, aged photograph, mysterious shadows, vintage control room, film grain, cinematic mood",
    "Film noir: vast empty dark ocean from aerial view, foggy and mysterious, no wreckage visible, moody desaturated colors, unsettling calm, cinematic wide shot, ominous atmosphere, noir photography",
    "Film noir: vintage 1945 naval search and rescue map with red pins and question marks, classified document aesthetic, aged yellowed paper, dramatic shadows, evidence room lighting, film grain, mysterious final reveal"
  ]
}}

REMEMBER:
- MUST USE ONE OF THE 5 TRENDING MYSTERIES IF PROVIDED
- Write as FLOWING NARRATIVE (not educational or list-based)
- Make it COMPLETELY DIFFERENT from previous topics
- Hit INTRIGUE immediately with impossible details
- Create SUSPENSE with pacing and contradictions
- Leave MYSTERY UNSOLVED - prompt theories
- NO SPECIAL CHARACTERS in JSON output (breaks parsing)

Generate the mystery story now.
"""

    return prompt


def get_fallback_script(content_type, mystery_type):
    """Fallback mystery scripts if all generation attempts fail"""
    
    fallback_scripts = {
        'evening_prime': {
            'title': "Flight 19: The Disappearance That Defies Physics",
            'hook': "December 5th 1945. Five planes vanished. No wreckage found.",
            'script': """December 5th, 1945. Fort Lauderdale Naval Air Station. Five torpedo bombers lifted off at two PM. Routine training mission. Flight 19. Fourteen experienced crew members. Lieutenant Charles Taylor leading. Perfect weather. Visibility: fifteen miles.

Two hours later, Taylor's voice crackled through the radio. 'We can't find west. Everything is wrong. Even the ocean doesn't look right.' Controllers tried to guide them back. Taylor's compass was malfunctioning. Both compasses. On all five planes. Simultaneously.

The Navy launched the largest search in history. Two hundred forty thousand square miles. Three hundred aircraft. Five days of continuous searching.

But here's where it gets strange. No wreckage. No oil slicks. No debris. No bodies. Not a single piece of metal. Five massive aircraft. Simply gone. As if they never existed.

The most terrifying part? A rescue plane was sent to find them. A Martin Mariner. Thirteen crew members. It also vanished. Same night. Same ocean. No distress call. No explanation.

Six aircraft. Twenty-seven men. Zero evidence. The Navy's official report concluded: 'as if they flew to Mars.'

To this day, Flight 19 remains the only mass disappearance in aviation history with absolutely no physical evidence.""",
            'key_phrase': "FLIGHT 19",
            'mystery_category': 'disappearance'
        },
        
        'late_night': {
            'title': "The Dyatlov Pass Incident: 9 Hikers Found Dead",
            'hook': "February 1959. Nine hikers died. The explanation is impossible.",
            'script': """February 2nd, 1959. The Ural Mountains. Nine experienced hikers set up camp on the slopes of Kholat Syakhl. The name means 'Dead Mountain' in the local language. They were all from the Ural Polytechnic Institute. Seasoned outdoorsmen. Led by Igor Dyatlov.

They never came back.

A search team found their tent on February 26th. It had been ripped open from the inside. As if they were desperate to escape. Their boots were still inside. Their supplies untouched. But the hikers were gone.

The first bodies were found at the edge of the forest. Barefoot. In freezing temperatures. Dead from hypothermia. But that was just the beginning.

But here's where it gets strange. Three more bodies were found in a ravine. Severe internal injuries. Fractured skulls. Broken ribs. One woman was missing her tongue. Another's eyes were gone. The force required? Equivalent to a car crash. Yet no external wounds. No signs of a struggle.

The most terrifying part? Their clothes contained high levels of radiation. The tent had mysterious orange lights above it that night. Witnesses reported strange glowing orbs in the sky.

The Soviet investigation concluded: 'death by unknown compelling force.'

To this day, no one can explain what made nine experienced hikers flee their tent barefoot into deadly cold. What were they running from?""",
            'key_phrase': "DYATLOV PASS",
            'mystery_category': 'crime'
        },
        
        'weekend_binge': {
            'title': "The Voynich Manuscript: A Book No One Can Read",
            'hook': "A 600 year old book. Written in an unknown language.",
            'script': """1912. An Italian villa. Book dealer Wilfrid Voynich discovered a medieval manuscript. 240 pages. Filled with bizarre illustrations. Plants that don't exist. Astronomical charts that make no sense. Naked figures in strange ceremonies. And text. Lots of text. In a language no one has ever seen.

The carbon dating confirmed it. Created in the early 15th century. Over 600 years old. Written on vellum. Real medieval ink. This wasn't a modern hoax.

Cryptographers tried to decode it. World War Two codebreakers. The NSA. Computer algorithms. All failed. The text follows statistical patterns of real language. It has grammar. Structure. Syntax. But the words? Complete gibberish. Or a code so complex it's never been broken.

But here's where it gets strange. The plants in the illustrations don't match any known species. Some look like combinations of different plants. Impossible hybrids. The astronomical charts show constellations that don't exist. The biological drawings depict organs arranged in ways that make no anatomical sense.

The most terrifying part? Linguistic analysis suggests the text isn't random. It has too much structure. Too much consistency. Someone spent years writing this. Creating elaborate illustrations. For what purpose?

Carbon dating proves it's medieval. Statistical analysis proves the language is structured. But after 600 years and countless experts, no one can read a single word.

To this day, the Voynich Manuscript remains the world's most mysterious book. What secrets does it hold?""",
            'key_phrase': "VOYNICH MANUSCRIPT",
            'mystery_category': 'historical'
        },
        
        'general': {
            'title': "The Bermuda Triangle: Where Ships Disappear",
            'hook': "Hundreds of ships and planes. All vanished without trace.",
            'script': """The Bermuda Triangle. A stretch of ocean between Miami, Bermuda, and Puerto Rico. Over the past century, more than 50 ships and 20 aircraft have disappeared here. No wreckage. No distress signals. No survivors. Just gone.

December 1945. Flight 19. Five torpedo bombers vanished during a training mission. The rescue plane sent to find them? Also disappeared. March 1918. The USS Cyclops. A massive naval cargo ship with 309 crew members. Vanished without sending an SOS. The largest loss of life in US Naval history not involving combat.

Weather can't explain it. The triangle has no more storms than anywhere else. Magnetic anomalies? The compass variations are normal. Methane gas eruptions? No evidence. Rogue waves? They don't erase radio signals.

But here's where it gets strange. The disappearances follow patterns. Ships and planes lose radio contact suddenly. No mayday calls. No emergency beacons. Just silence. When search teams arrive, they find nothing. No debris field. No oil slicks. The ocean is empty. As if the vessels simply ceased to exist.

The most terrifying part? It still happens. In 2015, the cargo ship El Faro disappeared in the triangle. 33 crew members. Modern navigation systems. Satellite communications. All lost. The wreckage was eventually found, but the black box recording ends abruptly. Mid-transmission. No explanation.

Statistical analysis shows the disappearance rate is actually normal for such a busy shipping lane. But that doesn't explain the ones with no distress signals. No debris. No answers.

What happens in the Bermuda Triangle?""",
            'key_phrase': "BERMUDA TRIANGLE",
            'mystery_category': 'disappearance'
        }
    }
    
    # Select appropriate fallback based on content type
    selected = fallback_scripts.get(content_type, fallback_scripts['evening_prime'])
    
    print(f"📋 Using fallback mystery script: {selected['title']}")
    
    return {
        'title': selected['title'],
        'topic': 'mystery',
        'hook': selected['hook'],
        'script': selected['script'],
        'key_phrase': selected['key_phrase'],
        'mystery_category': selected.get('mystery_category', 'disappearance'),
        'cta': 'What do you think happened?',
        'hashtags': ['#mystery', '#unsolved', '#truecrime', '#shorts', '#bermudatriangle', '#conspiracy'],
        'description': f"{selected['title']} - An unsolved mystery that defies explanation. {selected['hook']} The truth remains unknown. #mystery #unsolved #shorts",
        'visual_prompts': [
            'Film noir: vintage historical photograph related to mystery, dark moody lighting, high contrast black and white, dramatic shadows, mysterious atmosphere, documentary style, noir aesthetic, film grain',
            'Film noir: evidence or location scene, shadowy and mysterious, foggy atmosphere, vintage 1940s aesthetic, unsettling mood, cinematic lighting, noir photography',
            'Film noir: dramatic investigation scene, old documents and maps, classified aesthetic, aged paper, dramatic shadows, evidence room lighting, mysterious reveal',
            'Film noir: final mysterious image with unanswered questions, dark and ominous, film grain, documentary photography, noir aesthetic, unsettling atmosphere'
        ],
        'content_type': content_type,
        'priority': 'fallback',
        'mystery_type': mystery_type,
        'is_fallback': True,
        'generated_at': datetime.now().isoformat(),
        'niche': 'mystery'
    }


def generate_mystery_script():
    """Main script generation function (MYSTERY VERSION)"""
    
    # Get context from environment
    content_type = os.getenv('CONTENT_TYPE', 'evening_prime')  # Changed default
    priority = os.getenv('PRIORITY', 'medium')
    mystery_type = os.getenv('MYSTERY_TYPE', 'auto')  # Changed from INTENSITY
    
    print(f"\n{'='*70}")
    print(f"🔍 GENERATING MYSTERY SCRIPT")
    print(f"{'='*70}")
    print(f"📍 Content Type: {content_type}")
    print(f"⭐ Priority: {priority}")
    print(f"🎭 Mystery Type: {mystery_type}")
    
    # Load history and trending
    history = load_history()
    trends = load_trending()
    
    if trends:
        print(f"✅ Loaded trending data from {trends.get('source', 'unknown')}")
        print(f"   Topics: {len(trends.get('topics', []))}")
        print(f"   Niche: {trends.get('niche', 'unknown')}")
    else:
        print("⚠️ No trending data available")
    
    # Auto-select mystery type based on content type if 'auto'
    if mystery_type == 'auto':
        if content_type == 'evening_prime':
            mystery_type = 'disappearance'  # Famous mysteries for evening
        elif content_type == 'late_night':
            mystery_type = 'crime'  # Darker content for late night
        elif content_type == 'weekend_binge':
            mystery_type = 'historical'  # Complex mysteries for weekends
        else:
            mystery_type = 'disappearance'  # Default
        print(f"   Auto-selected mystery type: {mystery_type}")
    
    # Build prompt
    prompt = build_mystery_prompt(content_type, priority, mystery_type, trends, history)
    
    # Try generating with multiple attempts
    max_attempts = 5
    attempt = 0
    
    while attempt < max_attempts:
        try:
            attempt += 1
            print(f"\n🔍 Generation attempt {attempt}/{max_attempts}...")
            
            # Generate with retry logic
            raw_text = generate_script_with_retry(prompt)
            print(f"📝 Received response ({len(raw_text)} chars)")
            
            # Extract JSON
            json_text = extract_json_from_response(raw_text)
            
            # Parse JSON
            data = json.loads(json_text)
            
            # Validate structure
            validate_script_data(data)
            
            # Force topic to be mystery
            data["topic"] = "mystery"
            
            # Add metadata
            data["content_type"] = content_type
            data["priority"] = priority
            data["mystery_category"] = mystery_type
            data["generated_at"] = datetime.now().isoformat()
            data["niche"] = "mystery"
            
            # Clean text of problematic characters
            data["title"] = clean_script_text(data["title"])
            data["hook"] = clean_script_text(data["hook"])
            data["cta"] = clean_script_text(data["cta"])
            data["script"] = clean_script_text(data["script"])  # Clean full script
            
            # Add defaults for optional fields
            if "hashtags" not in data or not data["hashtags"]:
                data["hashtags"] = [
                    "#mystery", "#unsolved", "#truecrime", "#shorts"
                ]
            
            if "description" not in data:
                data["description"] = f"{data['title']} - {data['hook']} #mystery #unsolved #shorts"
            
            if "visual_prompts" not in data or len(data["visual_prompts"]) < 4:
                data["visual_prompts"] = [
                    f"Film noir: mysterious scene for {data['hook'][:50]}, dark moody lighting, vintage photography, high contrast, dramatic shadows, noir aesthetic",
                    f"Film noir: evidence or location for {data['title'][:50]}, documentary style, shadowy, mysterious atmosphere, film grain",
                    f"Film noir: dramatic reveal scene, unsettling, cinematic lighting, noir photography, ominous mood",
                    f"Film noir: final mysterious image, dramatic shadows, film grain, documentary photography, noir aesthetic"
                ]
            
            # Add key_phrase if missing (extract from title)
            if "key_phrase" not in data:
                # Try to extract mystery name from title (before the colon)
                if ':' in data['title']:
                    key_phrase = data['title'].split(':')[0].strip()
                else:
                    # Use first 3-5 words
                    words = data['title'].split()[:4]
                    key_phrase = ' '.join(words)
                data["key_phrase"] = key_phrase.upper()
            
            # Validate uses trending topics (if available)
            if trends and trends.get('topics'):
                if not validate_script_uses_trending_topic(data, trends['topics']):
                    raise ValueError("Script doesn't use trending topics - regenerating...")
            
            # Check for exact duplicates
            content_hash = get_content_hash(data)
            if content_hash in [t.get('hash') for t in history['topics']]:
                print("⚠️ Generated duplicate content (exact match), regenerating...")
                raise ValueError("Duplicate content detected")
            
            # Check for similar topics
            previous_titles = [t.get('title', '') for t in history['topics']]
            if is_similar_topic(data['title'], previous_titles):
                print("⚠️ Topic too similar to previous, regenerating...")
                raise ValueError("Similar topic detected")
            
            # Success! Save to history
            save_to_history(data['topic'], content_hash, data['title'], data)
            
            print(f"\n✅ MYSTERY SCRIPT GENERATED SUCCESSFULLY")
            print(f"   Title: {data['title']}")
            print(f"   Hook: {data['hook']}")
            print(f"   Key Phrase: {data.get('key_phrase', 'N/A')}")
            print(f"   Mystery Type: {data.get('mystery_category', 'N/A')}")
            print(f"   Script Length: {len(data['script'].split())} words")
            print(f"   Hashtags: {', '.join(data['hashtags'][:5])}")
            
            break  # Success, exit loop
            
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
            print("\n⚠️ Max attempts reached, using fallback mystery script...")
            data = get_fallback_script(content_type, mystery_type)
            
            # Save fallback to history
            fallback_hash = get_content_hash(data)
            save_to_history(data['topic'], fallback_hash, data['title'], data)
    
    # Save script to file
    script_path = os.path.join(TMP, "script.json")
    with open(script_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Saved script to {script_path}")
    
    # Save script text for TTS (SIMPLIFIED - just use the script field!)
    script_text_path = os.path.join(TMP, "script.txt")
    
    # Mystery version: script is already complete narrative!
    full_script = data['script']  # That's it! No joining needed.
    
    with open(script_text_path, "w", encoding="utf-8") as f:
        f.write(full_script)
    
    print(f"💾 Saved script text to {script_text_path}")
    
    # Summary
    print(f"\n{'='*70}")
    print(f"📊 GENERATION SUMMARY")
    print(f"{'='*70}")
    print(f"Total history: {len(history['topics'])} topics")
    print(f"Script length: {len(full_script.split())} words")
    print(f"Estimated duration: {len(full_script.split()) / 2.5:.1f}s (at 0.85 TTS speed)")
    print(f"Visual prompts: {len(data['visual_prompts'])}")
    
    if trends:
        print(f"\n🌐 Trending source: {trends.get('source', 'unknown')}")
    
    return data


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == '__main__':
    try:
        generate_mystery_script()
        print("\n✅ Mystery script generation complete!")
    except Exception as e:
        print(f"\n❌ FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        exit(1)