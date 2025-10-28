#!/usr/bin/env python3
"""
🔮 Fetch Trending Mystery Topics for Mythica Report
Multi-source trending data collector for the mystery, unsolved, and strange stories niche.
Sources: Google Trends, Reddit, YouTube, Evergreen Mystery Tropes
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

# Configure Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Model selection (robustly finds a suitable flash model)
try:
    models = genai.list_models()
    model_name = None
    for m in models:
        if 'generateContent' in m.supported_generation_methods:
            if '2.5-flash' in m.name or '2.0-flash' in m.name: # Prefer 2.5/2.0
                model_name = m.name
                break
            elif '1.5-flash' in m.name and not model_name: # Fallback to 1.5
                model_name = m.name
    
    if not model_name:
        model_name = "models/gemini-1.5-flash" # Default fallback
    
    print(f"✅ Using model: {model_name}")
    model = genai.GenerativeModel(model_name)
except Exception as e:
    print(f"⚠️ Error listing models, using default: {e}")
    model = genai.GenerativeModel("models/gemini-1.5-flash")

TMP = os.getenv("GITHUB_WORKSPACE", ".") + "/tmp"
os.makedirs(TMP, exist_ok=True)


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
        
        # 🔮 MYSTERY-SPECIFIC KEYWORDS
        mystery_topics = [
            # Core Mystery
            'unsolved mysteries',
            'strange disappearances',
            'unexplained phenomena',
            'creepy stories',
            'historical mysteries',
            
            # Specific Famous Cases (can trend due to new info/docs)
            'dyatlov pass incident',
            'db cooper',
            'zodiac killer',
            'roanoke colony',
            'jonbenet ramsey',
            
            # Internet & Modern Mysteries
            'internet mysteries',
            'cicada 3301',
            'glitch in the matrix',
            'mandela effect',
            
            # Paranormal & High Strangeness
            'ufo sightings',
            'skinwalker ranch',
            'haunted places'
        ]
        
        for topic in mystery_topics:
            try:
                print(f"   🔍 Searching trends for: {topic}")
                pytrends.build_payload([topic], timeframe='now 7-d', geo='US')
                
                # Get related queries
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
                
                time.sleep(random.uniform(1.5, 3.0))  # Respectful rate limiting
                
            except Exception as e:
                print(f"   ⚠️ Failed for '{topic}': {str(e)[:50]}...")
                continue
        
        print(f"✅ Found {len(relevant_trends)} mystery trends from Google")
        return relevant_trends[:20]
        
    except ImportError:
        print("⚠️ pytrends not installed - skipping Google Trends. Install with: pip install pytrends")
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
    """Get trending posts from mystery and unexplained phenomena subreddits"""
    try:
        print("🔮 Fetching Reddit mystery trends...")
        
        # 🔮 MYSTERY-SPECIFIC SUBREDDITS
        subreddits = [
            'UnsolvedMysteries',  # 2M+ members, the gold standard
            'HighStrangeness',    # 1M+ members, broad unexplained topics
            'Glitch_in_the_Matrix',# 1.5M+ members, personal strange experiences
            'creepy',             # 5M+ members, good for visuals and story hooks
            'RBI',                # Reddit Bureau of Investigation, for real-time cases
            'InternetIsBeautiful' # Occasionally surfaces weird/mysterious sites
        ]
        
        trends = []
        
        for subreddit in subreddits:
            try:
                url = f'https://www.reddit.com/r/{subreddit}/hot.json?limit=25'
                headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36'}
                
                print(f"   👽 Fetching r/{subreddit}...")
                response = requests.get(url, headers=headers, timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    posts_found = 0
                    
                    for post in data['data']['children'][:20]:
                        post_data = post['data']
                        title = post_data.get('title', '')
                        upvotes = post_data.get('ups', 0)
                        
                        # 🔮 MYSTERY-SPECIFIC FILTERING
                        good_phrases = [
                            'what happened to', 'the strange case of', 'the mystery of',
                            'unsolved disappearance', 'the only clue', 'chilling story',
                            'a strange detail about', 'nobody can explain', 'what was the'
                        ]
                        bad_phrases = [
                            'my theory', 'what do you think', 'discussion', 'what is your favorite',
                            'help me find', 'i think i solved', 'rant', 'meta', 'ama',
                            'unpopular opinion'
                        ]
                        
                        title_lower = title.lower()
                        has_good = any(phrase in title_lower for phrase in good_phrases)
                        has_bad = any(phrase in title_lower for phrase in bad_phrases)
                        is_viral = upvotes > 300 # Lower threshold for niche topics
                        
                        if (has_good and not has_bad) or (is_viral and not has_bad):
                            clean_title = clean_reddit_title(title)
                            if clean_title and len(clean_title) > 20:
                                trends.append(clean_title)
                                posts_found += 1
                                print(f"      ✓ ({upvotes} ↑) {clean_title[:70]}")
                    
                    print(f"      Found {posts_found} mystery story leads")
                else:
                    print(f"      ⚠️ Status {response.status_code}")
                time.sleep(random.uniform(2.0, 4.0))
                
            except Exception as e:
                print(f"   ⚠️ Failed to fetch r/{subreddit}: {e}")
                continue
        
        print(f"✅ Found {len(trends)} trending topics from Reddit")
        return trends[:20]
        
    except Exception as e:
        print(f"⚠️ Reddit scraping failed: {e}")
        return []


def clean_reddit_title(title: str) -> str:
    """Clean Reddit post titles for use as video topics"""
    title = re.sub(r'\[.*?\]', '', title) # Remove meta tags like [OC]
    title = re.sub(r'!!!+', '!', title)
    title = re.sub(r'\?\?+', '?', title)
    title = re.sub(r'[^\w\s\-.,!?\'"():;]', '', title) # Keep basic punctuation
    title = title.strip()
    return title


def get_youtube_mystery_trends() -> List[str]:
    """Scrape trending mystery video topics from YouTube"""
    try:
        print("🔮 Fetching YouTube trending mystery videos...")
        
        # 🔮 MYSTERY-SPECIFIC SEARCHES
        search_queries = [
            'unsolved mysteries',
            'terrifying stories',
            'unexplained videos',
            'strange disappearances that were never solved',
            'internet mysteries that are still unsolved'
        ]
        
        trends = []
        
        for query in search_queries:
            try:
                search_url = f"https://www.youtube.com/results?search_query={query.replace(' ', '+')}&sp=CAMSAhAB" # Recent
                headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
                
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
    """Check if YouTube title is original mystery content"""
    title_lower = title.lower()
    good_keywords = [
        'mystery', 'unsolved', 'creepy', 'strange', 'disappearance',
        'terrifying', 'unexplained', 'case of', 'chilling', 'haunting'
    ]
    bad_keywords = [
        'react', 'reaction', 'review', 'analysis', 'breakdown', 'full podcast',
        'interview', 'compilation', 'playlist', 'top 10', 'iceberg explained'
    ]
    
    has_good = any(kw in title_lower for kw in good_keywords)
    has_bad = any(kw in title_lower for kw in bad_keywords)
    
    return has_good and not has_bad and len(title) > 20


def get_evergreen_mystery_themes() -> List[str]:
    """Classic, evergreen mystery topics that always perform well"""
    
    # 🔮 TIMELESS MYSTERY THEMES
    evergreen = [
        # Classic Unsolved Cases
        "The Vanishing of the Mary Celeste Crew",
        "The Uncrackable Code of the Zodiac Killer",
        "Who Was D.B. Cooper? The Skyjacker Who Disappeared",
        "The Lost Colony of Roanoke: A 400-Year-Old Mystery",
        "The Chilling Case of the Dyatlov Pass Incident",

        # Strange Phenomena
        "The Wow! Signal: A Message From Deep Space?",
        "The Mystery of the Bermuda Triangle's Vanishing Ships",
        "The Tunguska Event: The Day a Forest Was Flattened by an Unknown Force",
        "What is The Hum? The Unexplained Sound Heard Worldwide",

        # Internet & Modern Mysteries
        "The Quest to Solve Cicada 3301: The Internet's Hardest Puzzle",
        "The Eerie Last Online Posts of People Who Vanished",
        "Lake City Quiet Pills: The Cryptic Reddit Mystery",
        
        # General Intrigue
        "Numbers Stations: Ghostly Radio Broadcasts With a Secret Purpose",
        "Lost Treasures That Are Still Waiting To Be Found",
        "The World's Most Mysterious Books That No One Can Read"
    ]
    
    print(f"✅ Loaded {len(evergreen)} evergreen mystery themes")
    return evergreen


def get_real_mystery_trends() -> List[str]:
    """Combine multiple FREE sources for real trending mystery topics"""
    
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
    
    # Deduplicate while preserving order and using similarity
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
    """Calculate Jaccard similarity between two strings based on words"""
    words1 = set(re.findall(r'\w+', s1.lower()))
    words2 = set(re.findall(r'\w+', s2.lower()))
    if not words1 or not words2: return 0.0
    intersection = len(words1 & words2)
    union = len(words1 | words2)
    return intersection / union if union > 0 else 0.0


def filter_and_rank_mystery_trends(trends: List[str], history: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Use Gemini to filter and rank mystery trends for viral potential on Mythica Report."""
    
    if not trends:
        print("⚠️ No trends to filter, using fallback mystery ideas...")
        return get_fallback_mystery_ideas()
    
    print(f"\n🤖 Using Gemini to rank {len(trends)} mystery topics for Mythica Report...")

        # Get the last 20 titles from history to avoid direct repeats
    previous_titles = [item.get('title', '') for item in history.get('topics', [])[-20:]]

    prompt = f"""You are a viral content strategist for "Mythica Report," a YouTube Shorts channel that tells short, chilling mystery stories. Your goal is to find topics that create suspense and make viewers say "What?!".

ANALYZING RAW TRENDING MYSTERY TOPICS (from Google/Reddit/YouTube/Evergreen):
{chr(10).join(f"- {t}" for t in trends[:30])}

**CRITICAL: DO NOT REPEAT RECENTLY COVERED TOPICS.**
Here are the titles of the last 20 videos. Avoid generating scripts on these exact topics or very similar ones:
- {chr(10).join(f"- {title}" for title in previous_titles) if previous_titles else "None yet."}

TASK: Select the TOP 5 topics from the RAW list above that are NOT in the "recently covered" list and would make the most compelling, suspenseful, and shareable YouTube Shorts for the "Mythica Report" channel.

SELECTION CRITERIA (ranked by importance):
1.  **Hook Potential:** The story must have an incredibly strong, mysterious opening. A question or a shocking statement.
2.  **Narrative Arc:** Can a compelling mini-story (setup, intrigue, mysterious climax/question) be told in under 60 seconds? Avoid topics that are too complex.
3.  **"Rabbit Hole" Effect:** The story must leave the viewer with a chilling, unanswered question that makes them want to Google it or check the comments. This drives engagement.
4.  **Visual Potential:** The topic should be easily visualizable with stock footage, archival photos, maps, or simple animations. Avoid purely abstract or philosophical topics.
5.  **Uniqueness:** Prioritize lesser-known mysteries or fresh angles on well-known ones over topics that are overly saturated.

GOOD MYTHICA REPORT TITLE EXAMPLES:
- "The Ship That Vanished With a Crew of Ghosts"
- "The Radio Signal From a Galaxy With No Stars"
- "He Disappeared From Inside a Locked Room"
- "The Town Where Everyone Forgot How to Sleep"

OUTPUT (JSON only):
Provide a JSON object with a "selected_topics" key, containing a list of 5 objects.
{{
  "selected_topics": [
    {{
      "title": "Create a short, suspenseful, click-worthy title for the YouTube Short",
      "reason": "Explain briefly why this topic is perfect for a short-form mystery video and fits the criteria.",
      "viral_score": 95,
      "story_hook": "Write the gripping first sentence of the script. This is the most important part.",
      "core_mystery": "What is the central, unanswered question at the heart of the story?",
      "ending_question": "What lingering question should the video end on to drive comments?"
    }}
  ]
}}

Select 5 topics and rank them by viral_score (highest first). Ensure the topics are distinct from one another.
"""

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = model.generate_content(prompt)
            result_text = response.text.strip()
            
            # Robust JSON extraction
            json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', result_text, re.DOTALL)
            if not json_match:
                json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
            
            if json_match:
                json_str = json_match.group(1)
                data = json.loads(json_str)
            else:
                raise json.JSONDecodeError("No JSON found in response", result_text, 0)
                
            trending_ideas = []
            for item in data.get('selected_topics', [])[:5]:
                trending_ideas.append({
                    "topic_title": item.get('title', 'Unknown Mystery'),
                    "summary": item.get('reason', 'High viral potential and strong narrative hook.'),
                    "category": "Unsolved Mystery",
                    "viral_score": item.get('viral_score', 90),
                    "story_hook": item.get('story_hook', 'A chilling discovery was made...'),
                    "core_mystery": item.get('core_mystery', 'The central question remains unanswered.'),
                    "ending_question": item.get('ending_question', 'What do you think really happened?'),
                })
            
            if not trending_ideas: raise ValueError("Gemini returned empty list of topics.")

            print(f"✅ Gemini ranked {len(trending_ideas)} viral mystery topics")
            for i, idea in enumerate(trending_ideas, 1):
                print(f"   {i}. [{idea['viral_score']}] {idea['topic_title'][:60]}")
            
            return trending_ideas
            
        except (json.JSONDecodeError, ValueError, Exception) as e:
            print(f"❌ Attempt {attempt + 1} failed: {e}")
            if "response" in locals() and "prompt_feedback" in response:
                print(f"   GEMINI FEEDBACK: {response.prompt_feedback}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
    
    print("⚠️ Gemini ranking failed after multiple attempts, using fallback ideas...")
    return get_fallback_mystery_ideas()


def get_fallback_mystery_ideas() -> List[Dict[str, Any]]:
    """Fallback mystery ideas if all API/scraping methods fail"""
    
    fallbacks = [
        {
            "topic_title": "The Ghost Ship of the Arctic: The Octavius",
            "summary": "A classic maritime mystery about a ship found frozen with its captain dead at his desk.",
            "category": "Historical Mystery",
            "viral_score": 94,
            "story_hook": "In 1775, a whaling ship stumbled upon a ghost ship, frozen solid in the Arctic ice.",
            "core_mystery": "How did the ship, The Octavius, end up thousands of miles off course, and what killed its crew instantly?",
            "ending_question": "Was it a shortcut gone wrong, or something more sinister?"
        },
        {
            "topic_title": "The Signal That Came From a Dead Star",
            "summary": "A real astronomical event where a powerful radio burst came from a place it shouldn't have.",
            "category": "Cosmic Mystery",
            "viral_score": 92,
            "story_hook": "Astronomers detected a repeating radio signal from a place they thought was empty: a graveyard of dead stars.",
            "core_mystery": "What is sending a structured signal from a cosmic graveyard where nothing should exist?",
            "ending_question": "Is it a natural phenomenon we don't understand, or something else?"
        },
        {
            "topic_title": "The Man Who Vanished From a National Park",
            "summary": "Focuses on the strange patterns of disappearances in national parks (David Paulides' work).",
            "category": "True Crime / Unexplained",
            "viral_score": 95,
            "story_hook": "Hundreds of people have vanished from US National Parks without a trace, often in bizarre circumstances.",
            "core_mystery": "Why do skilled hikers disappear from easy trails, only to be found miles away in impossible terrain, if they're found at all?",
            "ending_question": "Are these just tragic accidents, or is something preying on visitors in the woods?"
        }
    ]
    print(f"📋 Using {len(fallbacks)} classic fallback mystery ideas.")
    return fallbacks


def save_trending_data(trending_ideas: List[Dict[str, Any]]):
    """Save trending data to file for Mythica Report"""
    
    trending_data = {
        "topics": [idea["topic_title"] for idea in trending_ideas],
        "full_data": trending_ideas,
        "generated_at": datetime.now().isoformat(),
        "timestamp": time.time(),
        "niche": "mystery_stories",
        "channel": "Mythica Report",
        "source": "google_trends + reddit + youtube + evergreen + gemini_ranking",
        "version": "2.1_mythica_report"
    }
    
    trending_file = os.path.join(TMP, "trending.json")
    with open(trending_file, "w", encoding="utf-8") as f:
        json.dump(trending_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Saved mystery trending data to: {trending_file}")
    return trending_file


if __name__ == "__main__":
    # >>> ADD THIS LINE <<<
    from generate_trending_and_script import load_history 

    # Get real trending mystery topics
    real_trends = get_real_mystery_trends()

    # >>> ADD THIS LINE <<<
    history = load_history()
    
    if real_trends:
        # Use Gemini to filter and rank
        # >>> MODIFY THIS LINE <<<
        trending_ideas = filter_and_rank_mystery_trends(real_trends, history)
    else:
        # ... fallback logic
        print("⚠️ Could not fetch real trends, using fallback...")
        trending_ideas = get_fallback_mystery_ideas()
    
    if trending_ideas:
        print(f"\n" + "="*70)
        print(f"🔮 TOP VIRAL MYSTERY IDEAS FOR MYTHICA REPORT")
        print("="*70)
        
        for i, idea in enumerate(trending_ideas, 1):
            print(f"\n💎 IDEA {i}:")
            print(f"   Title: {idea['topic_title']}")
            print(f"   Viral Score: {idea.get('viral_score', 'N/A')}/100")
            print(f"   Hook: {idea.get('story_hook', 'N/A')}")
            print(f"   Mystery: {idea.get('core_mystery', 'N/A')}")
            print(f"   Ending: {idea.get('ending_question', 'N/A')}")
            print(f"   Why: {idea['summary'][:100]}...")
        
        # Save to file
        save_trending_data(trending_ideas)
        
        print(f"\n✅ TRENDING DATA READY FOR SCRIPT GENERATION")
        print(f"   Sources: Multi-platform real-time data")
        print(f"   Quality: Gemini-filtered for viral mystery potential")
        print(f"   Optimized: For Mythica Report's short-form storytelling")
        
    else:
        print("\n❌ Could not retrieve any trending ideas.")
        exit(1)