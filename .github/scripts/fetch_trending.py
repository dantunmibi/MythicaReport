#!/usr/bin/env python3
"""
🔮 Fetch Trending Mystery Topics for Mythica Report
ENHANCED: Better title validation to prevent name-first patterns
"""

import json
import time
import random
import os
import re
from typing import List, Dict, Any
from datetime import datetime
import google.generativeai as genai
import requests
from bs4 import BeautifulSoup

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

try:
    models = genai.list_models()
    model_name = None
    for m in models:
        if 'generateContent' in m.supported_generation_methods:
            if '2.5-flash' in m.name or '2.0-flash' in m.name:
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

TMP = os.getenv("GITHUB_WORKSPACE", ".") + "/tmp"
os.makedirs(TMP, exist_ok=True)

def load_history():
    """Load content history from previous runs"""
    history_file = os.path.join(TMP, "content_history.json")
    if os.path.exists(history_file):
        try:
            with open(history_file, 'r', encoding='utf-8') as f:
                history = json.load(f)
                print(f"📂 Loaded {len(history.get('topics', []))} topics from history")
                return history
        except Exception as e:
            print(f"⚠️ Could not load history: {e}")
            return {'topics': [], 'version': '3.1_retention_optimized'}
    
    print("📂 No previous history found, starting fresh")
    return {'topics': [], 'version': '3.1_retention_optimized'}

# [Keep all existing functions: get_google_trends_mystery, is_mystery_query, 
#  get_reddit_mystery_trends, clean_reddit_title, get_youtube_mystery_trends,
#  is_mystery_title, get_evergreen_mystery_themes, get_real_mystery_trends,
#  similar_strings - UNCHANGED]

def get_google_trends_mystery() -> List[str]:
    """Get real trending mystery-related searches from Google Trends"""
    try:
        from pytrends.request import TrendReq
        
        print(f"🔮 Fetching Google Trends (Unsolved Mysteries & Strange Phenomena)...")
        
        try:
            pytrends = TrendReq(hl='en-US', tz=360, timeout=(10, 25))
        except Exception as init_error:
            print(f"   ⚠️ PyTrends initialization failed: {init_error}")
            return []
        
        relevant_trends = []
        
        mystery_topics = [
            'unsolved mysteries',
            'strange disappearances',
            'unexplained phenomena',
            'creepy stories',
            'historical mysteries',
            'dyatlov pass incident',
            'db cooper',
            'zodiac killer',
            'roanoke colony',
            'jonbenet ramsey',
            'internet mysteries',
            'cicada 3301',
            'glitch in the matrix',
            'mandela effect',
            'ufo sightings',
            'skinwalker ranch',
            'haunted places'
        ]
        
        for topic in mystery_topics:
            try:
                print(f"   🔍 Searching trends for: {topic}")
                pytrends.build_payload([topic], timeframe='now 7-d', geo='US')
                
                related = pytrends.related_queries()
                
                if topic in related and 'top' in related[topic]:
                    top_queries = related[topic]['top']
                    if top_queries is not None and not top_queries.empty:
                        for query in top_queries['query'].head(5):
                            if is_mystery_query(query):
                                relevant_trends.append(query)
                                print(f"      ✓ {query}")
                
                if topic in related and 'rising' in related[topic]:
                    rising_queries = related[topic]['rising']
                    if rising_queries is not None and not rising_queries.empty:
                        for query in rising_queries['query'].head(3):
                            if is_mystery_query(query):
                                relevant_trends.append(f"{query} (🔥 RISING)")
                                print(f"      🔥 {query} (RISING)")
                
                time.sleep(random.uniform(1.5, 3.0))
                
            except Exception as e:
                print(f"   ⚠️ Failed for '{topic}': {str(e)[:50]}...")
                continue
        
        print(f"✅ Found {len(relevant_trends)} mystery trends from Google")
        return relevant_trends[:20]
        
    except ImportError:
        print("⚠️ pytrends not installed - skipping Google Trends")
        return []
    except Exception as e:
        print(f"⚠️ Google Trends failed: {e}")
        return []


def is_mystery_query(query: str) -> bool:
    """Filter for mystery/unsolved story relevance"""
    query_lower = query.lower()
    
    good_keywords = [
        'mystery', 'unsolved', 'disappearance', 'creepy', 'unexplained', 'strange',
        'case', 'theory', 'phenomenon', 'haunting', 'secret', 'code', 'lost',
        'ghost', 'alien', 'ufo', 'glitch', 'sighting'
    ]
    bad_keywords = [
        'movie', 'trailer', 'explained', 'ending', 'game', 'review',
        'full episode', 'cast', 'soundtrack', 'price', 'buy', 'shop', 'sale',
        'motivation', 'workout'
    ]
    
    has_good = any(kw in query_lower for kw in good_keywords)
    has_bad = any(kw in query_lower for kw in bad_keywords)
    
    return has_good and not has_bad and len(query) > 10


def get_reddit_mystery_trends() -> List[str]:
    """Get trending posts from mystery subreddits (optimized to avoid zero results)"""
    try:
        print("🔮 Fetching Reddit mystery trends...")

        # Expanded subreddit list
        subreddits = [
            'UnsolvedMysteries', 'HighStrangeness', 'Glitch_in_the_Matrix',
            'creepy', 'RBI', 'InternetIsBeautiful', 'StrangeUnexplained', 'TrueCrime'
        ]

        trends = []

        # Randomize 4 subreddits per run
        for subreddit in random.sample(subreddits, 4):
            urls = [
                f'https://www.reddit.com/r/{subreddit}/hot.json?limit=25',
                f'https://www.reddit.com/r/{subreddit}/new.json?limit=25',
                f'https://www.reddit.com/r/{subreddit}/rising.json?limit=25'
            ]

            posts_found = 0

            for url in urls:
                for attempt in range(3):
                    try:
                        headers = {'User-Agent': 'Mozilla/5.0'}
                        response = requests.get(url, headers=headers, timeout=10)
                        if response.status_code != 200:
                            raise ValueError(f"Status code {response.status_code}")
                        data = response.json()
                        break
                    except Exception as e:
                        print(f"   ⚠️ Attempt {attempt + 1} failed for {url}: {e}")
                        time.sleep(2 ** attempt)
                else:
                    continue  # skip to next URL if all attempts fail

                for post in data.get('data', {}).get('children', [])[:20]:
                    post_data = post.get('data', {})
                    title = post_data.get('title', '')
                    upvotes = post_data.get('ups', 0)

                    # Loosen good phrases to capture more trends
                    good_phrases = [
                        'what happened to', 'the strange case of', 'the mystery of',
                        'unsolved disappearance', 'the only clue', 'chilling story',
                        'a strange detail about', 'nobody can explain', 'what was the',
                        'disappearance', 'vanished', 'mystery'
                    ]
                    bad_phrases = [
                        'my theory', 'what do you think', 'discussion', 'help me find',
                        'rant', 'meta', 'ama', 'unpopular opinion'
                    ]

                    title_lower = title.lower()
                    has_good = any(phrase in title_lower for phrase in good_phrases)
                    has_bad = any(phrase in title_lower for phrase in bad_phrases)
                    is_viral = upvotes >= 100  # lower threshold

                    if (has_good or is_viral) and not has_bad:
                        clean_title = clean_reddit_title(title)
                        if clean_title and len(clean_title) >= 15:
                            trends.append(clean_title)
                            posts_found += 1
                            print(f"      ✓ ({upvotes} ↑) {clean_title[:70]}")

                time.sleep(random.uniform(1.5, 3.0))

            print(f"   Found {posts_found} mystery story leads in r/{subreddit}")

        print(f"✅ Found {len(trends)} total trends from Reddit")
        return trends[:20]

    except Exception as e:
        print(f"⚠️ Reddit scraping failed: {e}")
        return []


def clean_reddit_title(title: str) -> str:
    """Clean Reddit post titles"""
    title = re.sub(r'\[.*?\]', '', title)
    title = re.sub(r'!!!+', '!', title)
    title = re.sub(r'\?\?+', '?', title)
    title = re.sub(r'[^\w\s\-.,!?\'"():;]', '', title)
    return title.strip()


def get_youtube_mystery_trends() -> List[str]:
    """Scrape trending mystery videos from YouTube"""
    try:
        print("🔮 Fetching YouTube trending mystery videos...")
        
        search_queries = [
            'unsolved mysteries',
            'terrifying stories',
            'unexplained videos',
            'strange disappearances',
            'internet mysteries'
        ]
        
        trends = []
        
        for query in search_queries:
            try:
                search_url = f"https://www.youtube.com/results?search_query={query.replace(' ', '+')}&sp=CAMSAhAB"
                headers = {'User-Agent': 'Mozilla/5.0'}
                
                print(f"   🎥 Searching: {query}")
                response = requests.get(search_url, headers=headers, timeout=10)
                
                if response.status_code == 200:
                    title_pattern = r'"title":{"runs":\[{"text":"([^"]+)"}\]'
                    matches = re.findall(title_pattern, response.text)
                    
                    found_count = 0
                    for title in matches[:10]:
                        if is_mystery_title(title):
                            trends.append(title)
                            found_count += 1
                            print(f"      ✓ {title[:70]}")
                    print(f"      Found {found_count} videos")
                
                time.sleep(random.uniform(2.0, 3.0))
                
            except Exception as e:
                print(f"   ⚠️ Failed for '{query}': {e}")
                continue
        
        print(f"✅ Found {len(trends)} trends from YouTube")
        return trends[:15]
        
    except Exception as e:
        print(f"⚠️ YouTube scraping failed: {e}")
        return []


def is_mystery_title(title: str) -> bool:
    """Check if YouTube title is mystery content"""
    title_lower = title.lower()
    good_keywords = [
        'mystery', 'unsolved', 'creepy', 'strange', 'disappearance',
        'terrifying', 'unexplained', 'case of', 'chilling', 'haunting'
    ]
    bad_keywords = [
        'react', 'reaction', 'review', 'analysis', 'breakdown',
        'interview', 'compilation', 'playlist', 'top 10'
    ]
    
    has_good = any(kw in title_lower for kw in good_keywords)
    has_bad = any(kw in title_lower for kw in bad_keywords)
    
    return has_good and not has_bad and len(title) > 20


def get_evergreen_mystery_themes() -> List[str]:
    """Classic mystery topics"""
    evergreen = [
        "The Vanishing of the Mary Celeste Crew",
        "The Uncrackable Code of the Zodiac Killer",
        "Who Was D.B. Cooper? The Skyjacker Who Disappeared",
        "The Lost Colony of Roanoke: A 400-Year-Old Mystery",
        "The Chilling Case of the Dyatlov Pass Incident",
        "The Wow! Signal: A Message From Deep Space?",
        "The Mystery of the Bermuda Triangle's Vanishing Ships",
        "The Tunguska Event: The Day a Forest Was Flattened",
        "What is The Hum? The Unexplained Sound Heard Worldwide",
        "The Quest to Solve Cicada 3301: The Internet's Hardest Puzzle",
        "The Eerie Last Online Posts of People Who Vanished",
        "Lake City Quiet Pills: The Cryptic Reddit Mystery",
        "Numbers Stations: Ghostly Radio Broadcasts",
        "Lost Treasures That Are Still Waiting To Be Found",
        "The World's Most Mysterious Books That No One Can Read"
    ]
    
    print(f"✅ Loaded {len(evergreen)} evergreen mystery themes")
    return evergreen


def get_real_mystery_trends() -> List[str]:
    """Combine multiple sources for trending topics"""
    
    print("\n" + "="*70)
    print("🔮 FETCHING REAL-TIME MYSTERY TRENDS FOR MYTHICA REPORT")
    print("="*70)
    
    all_trends = []
    source_counts = {}
    
    try:
        google_trends = get_google_trends_mystery()
        all_trends.extend(google_trends)
        source_counts['Google Trends'] = len(google_trends)
    except Exception as e:
        print(f"⚠️ Google Trends error: {e}")

    try:
        reddit_trends = get_reddit_mystery_trends()
        all_trends.extend(reddit_trends)
        source_counts['Reddit'] = len(reddit_trends)
    except Exception as e:
        print(f"⚠️ Reddit error: {e}")

    try:
        youtube_trends = get_youtube_mystery_trends()
        all_trends.extend(youtube_trends)
        source_counts['YouTube'] = len(youtube_trends)
    except Exception as e:
        print(f"⚠️ YouTube error: {e}")
    
    evergreen = get_evergreen_mystery_themes()
    all_trends.extend(evergreen)
    source_counts['Evergreen'] = len(evergreen)
    
    seen = set()
    unique_trends = []
    for trend in all_trends:
        trend_clean = trend.lower().strip()
        is_duplicate = False
        for seen_trend in seen:
            if similar_strings(trend_clean, seen_trend) > 0.8:
                is_duplicate = True
                break
        
        if not is_duplicate and len(trend) > 15:
            seen.add(trend_clean)
            unique_trends.append(trend)
    
    print(f"\n📊 TREND SOURCES SUMMARY:")
    for source, count in source_counts.items():
        print(f"   • {source}: {count} topics")
    print(f"\n   TOTAL UNIQUE: {len(unique_trends)} mystery trends")
    
    return unique_trends[:30]


def similar_strings(s1: str, s2: str) -> float:
    """Calculate similarity between strings"""
    words1 = set(re.findall(r'\w+', s1.lower()))
    words2 = set(re.findall(r'\w+', s2.lower()))
    if not words1 or not words2: return 0.0
    intersection = len(words1 & words2)
    union = len(words1 | words2)
    return intersection / union if union > 0 else 0.0


def filter_and_rank_mystery_trends(trends: List[str], history: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    🚨 ENHANCED: Use Gemini to filter trends with TITLE PATTERN VALIDATION
    Prevents name-first patterns that cause retention collapse
    """
    
    if not trends:
        print("⚠️ No trends to filter, using fallback...")
        return get_fallback_mystery_ideas()
    
    print(f"\n🤖 Using Gemini to rank {len(trends)} mystery topics (RETENTION-OPTIMIZED)...")

    previous_titles = [item.get('title', '') for item in history.get('topics', [])[-20:]]

    prompt = f"""You are a viral content strategist for "Mythica Report," a YouTube Shorts channel.

ANALYZING RAW TRENDING MYSTERY TOPICS:
{chr(10).join(f"- {t}" for t in trends[:30])}

**CRITICAL: AVOID RECENTLY COVERED TOPICS:**
{chr(10).join(f"- {title}" for title in previous_titles) if previous_titles else "None yet."}

🚨 TITLE PATTERN REQUIREMENTS (RETENTION-CRITICAL):

✅ GOOD PATTERNS (70%+ retention):
- "The [Role/Object] Who/That Vanished" (e.g., "The Hiker Who Vanished")
- "The [Mystery]: [Impossible Detail]" (e.g., "The Signal: Came From a Dead Star")
- "[Action] + [Impossible Outcome]" (e.g., "Five Planes Vanished Without Trace")

❌ BAD PATTERNS (20-35% retention):
- "[Name]: [Mystery]" (e.g., "Leah Roberts: The Vanishing Driver")
- "[Name] + [Action]" (e.g., "John Doe Disappeared in 2000")
- Reason: Viewers don't know the name, need context first

MANDATORY TITLE RULES:
1. MUST include "Vanished" or "Disappeared"
2. MUST NOT start with unfamiliar proper names
3. MUST describe ROLE before name (e.g., "The Student Who Vanished" not "Sarah Who Vanished")

SELECT TOP 5 TOPICS with retention-optimized titles.

OUTPUT (JSON):
{{
  "selected_topics": [
    {{
      "title": "The [Role/Thing] Who/That Vanished",
      "reason": "Why this is viral + follows proven retention pattern",
      "viral_score": 95,
      "story_hook": "First sentence revealing the mystery immediately",
      "core_mystery": "Central unanswered question",
      "ending_question": "Cliffhanger for comments"
    }}
  ]
}}
"""

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = model.generate_content(prompt)
            result_text = response.text.strip()
            
            json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', result_text, re.DOTALL)
            if not json_match:
                json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
            
            if json_match:
                json_str = json_match.group(1)
                data = json.loads(json_str)
            else:
                raise json.JSONDecodeError("No JSON found", result_text, 0)
                
            trending_ideas = []
            for item in data.get('selected_topics', [])[:5]:
                title = item.get('title', 'Unknown Mystery')
                
                # ✅ VALIDATE TITLE PATTERN
                title_lower = title.lower()
                has_vanished = 'vanish' in title_lower or 'disappear' in title_lower
                
                if not has_vanished:
                    print(f"   ⚠️ Skipping '{title}' - missing 'vanished/disappeared'")
                    continue
                
                # Check for name-first pattern
                if ':' in title:
                    first_part = title.split(':')[0].strip()
                    words = first_part.split()
                    if len(words) <= 3 and all(w[0].isupper() for w in words if w):
                        print(f"   ⚠️ Skipping '{title}' - name-first pattern (low retention)")
                        continue
                
                trending_ideas.append({
                    "topic_title": title,
                    "summary": item.get('reason', 'High viral potential'),
                    "category": "Unsolved Mystery",
                    "viral_score": item.get('viral_score', 90),
                    "story_hook": item.get('story_hook', 'A chilling discovery...'),
                    "core_mystery": item.get('core_mystery', 'Unanswered question'),
                    "ending_question": item.get('ending_question', 'What do you think?'),
                })
            
            if not trending_ideas:
                raise ValueError("All topics rejected by title validation")

            print(f"✅ Gemini ranked {len(trending_ideas)} retention-optimized topics")
            for i, idea in enumerate(trending_ideas, 1):
                print(f"   {i}. [{idea['viral_score']}] {idea['topic_title'][:60]}")
            
            return trending_ideas
            
        except Exception as e:
            print(f"❌ Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
    
    print("⚠️ Gemini ranking failed, using fallback...")
    return get_fallback_mystery_ideas()


def get_fallback_mystery_ideas() -> List[Dict[str, Any]]:
    """Fallback ideas with CORRECT title patterns"""
    
    fallbacks = [
        {
            "topic_title": "The Ghost Ship of the Arctic: The Octavius",
            "summary": "Classic maritime mystery with strong retention pattern",
            "category": "Historical Mystery",
            "viral_score": 94,
            "story_hook": "In 1775, a whaling ship found a ghost ship frozen in Arctic ice.",
            "core_mystery": "How did the ship end up thousands of miles off course with its crew frozen?",
            "ending_question": "Was it a shortcut gone wrong, or something more sinister?"
        },
        {
            "topic_title": "The Signal That Came From a Dead Star",
            "summary": "Real astronomical event, high curiosity factor",
            "category": "Cosmic Mystery",
            "viral_score": 92,
            "story_hook": "Astronomers detected a repeating signal from a cosmic graveyard.",
            "core_mystery": "What is sending a signal from where nothing should exist?",
            "ending_question": "Is it natural, or something else?"
        },
        {
            "topic_title": "The Hiker Who Vanished From an Easy Trail",
            "summary": "Taps into national park disappearance mystery trend",
            "category": "True Crime",
            "viral_score": 95,
            "story_hook": "Hundreds vanish from US National Parks without a trace.",
            "core_mystery": "Why do experienced hikers disappear from easy trails?",
            "ending_question": "Are these accidents, or is something else happening?"
        }
    ]
    print(f"📋 Using {len(fallbacks)} retention-optimized fallback ideas")
    return fallbacks


def save_trending_data(trending_ideas: List[Dict[str, Any]]):
    """Save trending data to file"""
    
    trending_data = {
        "topics": [idea["topic_title"] for idea in trending_ideas],
        "full_data": trending_ideas,
        "generated_at": datetime.now().isoformat(),
        "timestamp": time.time(),
        "niche": "mystery_stories",
        "channel": "Mythica Report",
        "source": "google_trends + reddit + youtube + evergreen + gemini_ranking",
        "version": "3.1_retention_optimized"
    }
    
    trending_file = os.path.join(TMP, "trending.json")
    with open(trending_file, "w", encoding="utf-8") as f:
        json.dump(trending_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Saved trending data to: {trending_file}")
    return trending_file


if __name__ == "__main__":
    real_trends = get_real_mystery_trends()
    history = load_history()
    
    if real_trends:
        trending_ideas = filter_and_rank_mystery_trends(real_trends, history)
    else:
        print("⚠️ No real trends, using fallback...")
        trending_ideas = get_fallback_mystery_ideas()
    
    if trending_ideas:
        print(f"\n" + "="*70)
        print(f"🔮 TOP VIRAL MYSTERY IDEAS (RETENTION-OPTIMIZED)")
        print("="*70)
        
        for i, idea in enumerate(trending_ideas, 1):
            print(f"\n💎 IDEA {i}:")
            print(f"   Title: {idea['topic_title']}")
            print(f"   Viral Score: {idea.get('viral_score', 'N/A')}/100")
            print(f"   Hook: {idea.get('story_hook', 'N/A')}")
            print(f"   Mystery: {idea.get('core_mystery', 'N/A')}")
        
        save_trending_data(trending_ideas)
        
        print(f"\n✅ TRENDING DATA READY")
        print(f"   Quality: Retention-optimized title patterns")
        print(f"   Validation: Name-first patterns rejected")
        
    else:
        print("\n❌ Could not retrieve any trending ideas")
        exit(1)