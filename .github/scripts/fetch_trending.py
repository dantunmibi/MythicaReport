#!/usr/bin/env python3
"""
🔮 Fetch Trending Mystery Topics for Mythica Report
v6.0.1: Fixed entity-based duplicate detection (removed false positives)

CHANGES IN v6.0.1:
- Tightened entity extraction to only proper nouns and specific identifiers
- Added verb phrase blacklist (prevents "its creators vanished" from being treated as entity)
- Removed fallback generic word joining (prevented fake entities)
- Whitelist-only approach for known cases/places
"""

import json
import time
import random
import os
import re
from typing import List, Dict, Any
from datetime import datetime, timedelta
import google.generativeai as genai
import requests
from bs4 import BeautifulSoup
import argparse

# ========================================================================
# CATEGORY-AWARE FETCHING (v6.0.9)
# ========================================================================
parser = argparse.ArgumentParser(description='Fetch trending mystery topics for Mythica Report')
parser.add_argument(
    '--category', 
    type=str, 
    default='general',
    choices=['general', 'disturbing_medical', 'dark_experiments', 'dark_history', 
             'disappearance', 'phenomena', 'crime', 'conspiracy'],
    help='Mystery category to fetch trending topics for'
)
args = parser.parse_args()

MYSTERY_CATEGORY = args.category

print(f"🎯 CATEGORY-AWARE FETCH: {MYSTERY_CATEGORY}")

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
            return {'topics': [], 'version': '6.0.1_entity_fix'}
    
    print("📂 No previous history found, starting fresh")
    return {'topics': [], 'version': '6.0.1_entity_fix'}

def get_category_search_terms(category: str) -> List[str]:
    """
    🆕 v6.0.9: Get category-specific search terms for trending APIs
    
    Returns list of search queries optimized for each mystery category
    """
    
    category_terms = {
        'disturbing_medical': [
            # Medical conditions
            'rare disease mystery',
            'fatal familial insomnia',
            'mysterious illness',
            'kuru disease',
            'dancing plague',
            'mass hysteria medical',
            'unexplained syndrome',
            'medical anomaly',
            
            # Historical medical
            'radium girls',
            'minamata disease',
            'encephalitis lethargica',
            'tanganyika laughing epidemic',
            'ergotism',
            
            # Body horror
            'fibrodysplasia ossificans progressiva',
            'turning to stone disease',
            'mystery paralysis',
            'sleeping sickness',
            'radiation poisoning victims'
        ],
        
        'dark_experiments': [
            # Government experiments
            'mk ultra',
            'mkultra declassified',
            'stanford prison experiment',
            'milgram experiment',
            'tuskegee experiment',
            'unit 731',
            
            # Secret research
            'secret government experiments',
            'cia mind control',
            'project artichoke',
            'project bluebird',
            'midnight climax',
            'edgewood arsenal',
            
            # Psychological
            'unethical human experiments',
            'sleep deprivation experiment',
            'russian sleep experiment',
            'psychological torture experiments',
            'behavior modification research'
        ],
        
        'dark_history': [
            # Historical mysteries
            'dark historical events',
            'mysterious 1800s incident',
            'dark day 1780',
            'mysterious historical phenomenon',
            
            # Specific events
            'radium girls story',
            'dancing plague 1518',
            'strasbourg dancing mania',
            'kentucky meat shower',
            'roanoke colony mystery',
            
            # Time periods
            'victorian mysteries',
            'medieval unexplained events',
            'colonial era mysteries',
            'industrial revolution disasters',
            '19th century mysteries',
            'historical mass poisoning'
        ],
        
        'disappearance': [
            # Classic disappearances
            'missing person mystery',
            'unsolved disappearance',
            'vanished without trace',
            
            # Famous cases
            'dyatlov pass incident',
            'db cooper mystery',
            'amelia earhart disappearance',
            'flight 19 bermuda triangle',
            'mh370 mystery',
            'maura murray case',
            
            # Categories
            'national park disappearances',
            'missing 411',
            'vanished hikers',
            'missing climbers',
            'lost at sea mysteries',
            'ghost ship mary celeste'
        ],
        
        'phenomena': [
            # Space/cosmic
            'wow signal',
            'mysterious space signal',
            'fast radio burst',
            'unexplained radio transmission',
            
            # Lights
            'hessdalen lights',
            'marfa lights',
            'brown mountain lights',
            'ufo sighting',
            'uap phenomenon',
            
            # Sounds
            'the hum phenomenon',
            'taos hum',
            'skyquake mystery',
            'bloop sound',
            'underwater sounds',
            
            # Paranormal
            'numbers stations',
            'mysterious broadcasts',
            'unexplained phenomenon',
            'paranormal activity',
            'glitch in the matrix stories'
        ],
        
        'crime': [
            # Serial killers
            'zodiac killer',
            'zodiac cipher',
            'jack the ripper',
            'golden state killer',
            
            # Unsolved murders
            'jonbenet ramsey case',
            'black dahlia murder',
            'somerton man mystery',
            'boy in the box case',
            
            # Cold cases
            'unsolved murder mystery',
            'cold case files',
            'true crime unsolved',
            'mysterious death case',
            'murder mystery unsolved'
        ],
        
        'conspiracy': [
            # Classic conspiracies
            'area 51 secrets',
            'philadelphia experiment',
            'montauk project',
            
            # Cover-ups
            'government coverup',
            'classified documents leaked',
            'conspiracy theory',
            'unexplained government activity'
        ],
        
        'general': [
            # Broad mystery terms
            'unsolved mysteries',
            'strange disappearances',
            'unexplained phenomena',
            'creepy stories',
            'historical mysteries',
            'internet mysteries',
            'true crime mysteries'
        ]
    }
    
    selected_terms = category_terms.get(category, category_terms['general'])
    
    print(f"   🔍 Using {len(selected_terms)} category-specific search terms for '{category}'")
    
    return selected_terms

def get_category_subreddits(category: str) -> List[str]:
    """
    🆕 v6.0.9: Get category-specific subreddits
    
    Returns list of subreddits optimized for each mystery category
    """
    
    category_subreddits = {
        'disturbing_medical': [
            'UnresolvedMysteries',
            'creepy',
            'medizzy',
            'morbidquestions',
            'TrueCreepy',
            'nosleep',  # fictional but sometimes based on real conditions
            'medical'
        ],
        
        'dark_experiments': [
            'UnresolvedMysteries',
            'conspiracy',
            'TrueCrime',
            'history',
            'todayilearned',
            'psychology'
        ],
        
        'dark_history': [
            'history',
            'HistoryMemes',
            'UnresolvedMysteries',
            'creepy',
            'TrueHistoricalMystery',
            'historicalrage'
        ],
        
        'disappearance': [
            'UnresolvedMysteries',
            'Missing411',
            'TrueCrime',
            'RBI',
            'mystery',
            'UnsolvedMurders'
        ],
        
        'phenomena': [
            'HighStrangeness',
            'Glitch_in_the_Matrix',
            'UnexplainedPhotos',
            'Paranormal',
            'UFOs',
            'astronomy',
            'space'
        ],
        
        'crime': [
            'TrueCrime',
            'UnresolvedMysteries',
            'UnsolvedMurders',
            'serialkillers',
            'ColdCases',
            'mystery'
        ],
        
        'conspiracy': [
            'conspiracy',
            'UnresolvedMysteries',
            'HighStrangeness',
            'AlternativeHistory'
        ],
        
        'general': [
            'UnsolvedMysteries',
            'HighStrangeness',
            'Glitch_in_the_Matrix',
            'creepy',
            'RBI',
            'TrueCrime'
        ]
    }
    
    selected_subs = category_subreddits.get(category, category_subreddits['general'])
    
    print(f"   📱 Using {len(selected_subs)} category-specific subreddits for '{category}'")
    
    return selected_subs

def get_google_news_mystery() -> List[str]:
    """
    🆕 v6.0.9: Fetch trending mystery news via Google News RSS
    
    100% reliable - RSS feeds are official Google service
    No API key needed, no rate limits, no 403 errors
    
    Returns:
        List of news article titles related to mysteries
    """
    try:
        import feedparser
        
        print("📰 Fetching Google News mystery topics...")
        
        # Use category-specific search terms
        news_queries = get_category_search_terms(MYSTERY_CATEGORY)[:5]
        
        print(f"   📡 Using {len(news_queries)} search queries for '{MYSTERY_CATEGORY}'")
        
        trends = []
        
        for query in news_queries:
            try:
                # Google News RSS feed URL
                # Format: news.google.com/rss/search?q=QUERY&hl=en-US&gl=US&ceid=US:en
                rss_url = f"https://news.google.com/rss/search?q={query.replace(' ', '+')}&hl=en-US&gl=US&ceid=US:en"
                
                print(f"   🔍 Searching news for: {query}")
                
                # Parse RSS feed
                feed = feedparser.parse(rss_url)
                
                # Extract article titles
                for entry in feed.entries[:5]:
                    title = entry.title
                    
                    # Remove news source suffix (e.g., " - CNN", " - BBC News")
                    title = re.sub(r'\s*-\s*[A-Z][a-zA-Z\s&.]+$', '', title)
                    
                    # Basic quality filters
                    if len(title) > 20 and is_mystery_query(title):
                        trends.append(title)
                        print(f"      ✓ {title[:70]}")
                
                # Small delay to be respectful
                time.sleep(random.uniform(0.3, 0.7))
                
            except Exception as e:
                print(f"   ⚠️ Failed for '{query}': {str(e)[:50]}...")
                continue
        
        print(f"✅ Found {len(trends)} trends from Google News")
        return trends[:15]
        
    except ImportError:
        print("⚠️ feedparser not installed - skipping Google News")
        print("   Install with: pip install feedparser")
        return []
    except Exception as e:
        print(f"⚠️ Google News failed: {e}")
        return []

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
        
        mystery_topics = get_category_search_terms(MYSTERY_CATEGORY)
        print(f"   🎯 Searching Google Trends for '{MYSTERY_CATEGORY}' topics...")
        
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
        subreddits = get_category_subreddits(MYSTERY_CATEGORY)
        
        print(f"   🎯 Searching Reddit for '{MYSTERY_CATEGORY}' topics...")

        trends = []

        # Randomize 4 subreddits per run
        sample_size = min(4, len(subreddits))
        for subreddit in random.sample(subreddits, sample_size):
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
        
        # 🆕 v6.0.9: Use category-specific search queries
        category_terms = get_category_search_terms(MYSTERY_CATEGORY)
        
        # Select top 5 most specific terms for YouTube
        search_queries = category_terms[:5]
        
        print(f"   🎯 Searching YouTube for '{MYSTERY_CATEGORY}' topics...")
        
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


def get_evergreen_mystery_themes(category: str = 'general') -> List[str]:
    """
    🆕 v6.0.9: Category-specific evergreen topics
    Ensures fallback topics match the mystery category
    
    Args:
        category: Mystery category (disturbing_medical, dark_experiments, etc.)
    
    Returns:
        List of category-appropriate evergreen topics
    """
    
    evergreen_by_category = {
        'disturbing_medical': [
            "The Man Who Never Slept: Fatal Insomnia Mystery",
            "The Disease That Turns You To Stone: FOP Case",
            "The Town That Went Blind: Minamata Outbreak",
            "The Laughing Death That Spread: Kuru Disease",
            "The Dancing Plague of 1518: Strasbourg Mania",
            "The Sleeping Sickness Epidemic: Encephalitis Lethargica",
            "The Girl Who Aged Decades: Progeria Mystery",
            "The Blue People of Kentucky: Methemoglobinemia",
            "The Stone Man Syndrome: FOP Medical Mystery",
            "The Fatal Familial Insomnia Cases",
            "The Radium Girls: Glowing Death Mystery",
            "The Tanganyika Laughing Epidemic: 1962 Outbreak",
            "The Ergot Poisoning: Dancing Mania Mystery",
            "The Mysterious Paralysis: Polio Outbreaks",
            "The Encephalitis Lethargica: Sleeping Sickness",
        ],
        
        'dark_experiments': [
            "The CIA Mind Control Experiment: MK-Ultra Files",
            "The Prison Study Gone Wrong: Stanford 1971",
            "The 40-Year Lie: Tuskegee Experiment",
            "The Japanese War Crimes: Unit 731 Research",
            "The Russian Sleep Deprivation: Experiment Mystery",
            "The CIA Truth Serum Tests: Project Artichoke",
            "The Obedience Study: Milgram Experiment Revealed",
            "The Secret Chemical Tests: Edgewood Arsenal",
            "The Stuttering Experiment: The Monster Study",
            "The Nazi Scientists Recruited: Project Paperclip",
            "The LSD Experiments: CIA Midnight Climax",
            "The Poison Squad Trials: Food Safety Tests",
            "The Radiation Experiments: Cold War Secrets",
            "The Electroshock Studies: Brain Modification",
            "The Sensory Deprivation Tests: CIA Research",
        ],
        
        'dark_history': [
            "The Day It Rained Meat: 1876 Kentucky Mystery",
            "The Dark Day of 1780: New England Blackout",
            "The Dancing Plague of 1518: Strasbourg Event",
            "The Radium Girls Who Glowed: Factory Deaths",
            "The Great Molasses Flood: 1919 Boston Disaster",
            "The Year Without Summer: 1816 Climate Mystery",
            "The Children's Crusade: 1212 Historical Mystery",
            "The Halifax Explosion: 1917 Disaster",
            "The Donner Party Tragedy: 1846 Survival Horror",
            "The Tunguska Event: 1908 Siberian Blast",
            "The Burning of Centralia: Town on Fire Since 1962",
            "The Great Fire of London: 1666 Mystery",
            "The Black Death Plague: Medieval Pandemic",
            "The Plague of Justinian: Byzantine Disaster",
            "The Year of the Four Emperors: Roman Mystery",
        ],
        
        'disappearance': [
            "The Vanishing Crew: Mary Celeste Mystery",
            "The Skyjacker Who Disappeared: D.B. Cooper",
            "The Lost Colony: Roanoke 400-Year Mystery",
            "The Five Planes That Vanished: Flight 19",
            "The Bermuda Triangle: Vanishing Ships Mystery",
            "The Final Flight: Amelia Earhart Mystery",
            "The Nine Hikers Found Dead: Dyatlov Pass",
            "The Missing Airliner: MH370 Mystery",
            "The Lighthouse Keepers Who Vanished: Flannan Isles",
            "The Union Boss Who Disappeared: Jimmy Hoffa",
            "The Student Who Vanished: Maura Murray Case",
            "The Hiker Who Never Returned: Brandon Swanson",
            "The Teenager Who Disappeared: Asha Degree",
            "The College Student Gone: Brian Shaffer",
            "The Road Trip Mystery: Leah Roberts",
        ],
        
        'phenomena': [
            "The Signal From Deep Space: Wow! Mystery",
            "The Unexplained Hum: Worldwide Phenomenon",
            "The Norway Mystery Lights: Hessdalen",
            "The Underwater Sound Mystery: The Bloop",
            "The Ghost Radio Broadcasts: Numbers Stations",
            "The Siberian Explosion: Tunguska Event",
            "The Cosmic Mystery Signals: Fast Radio Bursts",
            "The Unexplained Phenomenon: Ball Lightning",
            "The New Mexico Mystery: Taos Hum",
            "The Texas Lights Mystery: Marfa Phenomenon",
            "The Mysterious Radio Signal: UVB-76",
            "The Space Anomaly: Oumuamua Mystery",
            "The Lake Monster Sightings: Nessie Mystery",
            "The UFO Incident: Phoenix Lights",
            "The Mysterious Crop Circles: Field Patterns",
        ],
        
        'crime': [
            "The Unsolved Cipher: Zodiac Killer Mystery",
            "The 1888 London Murders: Jack the Ripper",
            "The Black Dahlia Murder: Elizabeth Short Case",
            "The Child Murder Mystery: JonBenét Ramsey",
            "The Beach Body Mystery: Somerton Man",
            "The Boy in the Box: Philadelphia 1957",
            "The Jazz Age Serial Killer: Axeman of New Orleans",
            "The Torso Murders: Cleveland 1930s Mystery",
            "The Iowa Massacre: Villisca Axe Murders",
            "The New Jersey Murder: Hall-Mills Case",
            "The Valentine's Day Massacre: Chicago 1929",
            "The Lizzie Borden Case: 1892 Mystery",
            "The Cleveland Kidnappings: Ariel Castro",
            "The Golden State Killer: East Area Rapist",
            "The Boston Strangler: Albert DeSalvo",
        ],
        
        'conspiracy': [
            "The Area 51 Secrets: What's Really There",
            "The Philadelphia Experiment: Navy Mystery",
            "The Montauk Project: Time Travel Claims",
            "The Roswell Incident: UFO Cover-Up Mystery",
            "The JFK Files: Assassination Conspiracy",
            "The Moon Landing Hoax Claims: Debunked Mystery",
            "The Illuminati Conspiracy: Secret Society",
            "The New World Order: Global Control Theory",
            "The Flat Earth Theory: Modern Conspiracy",
            "The Chemtrails Conspiracy: Sky Mystery",
        ],
        
        'general': [
            "The Vanishing Crew: Mary Celeste Mystery",
            "The Unsolved Cipher: Zodiac Killer",
            "The Skyjacker Who Disappeared: D.B. Cooper",
            "The Lost Colony: Roanoke Mystery",
            "The Nine Hikers Found Dead: Dyatlov Pass",
            "The Signal From Deep Space: Wow! Mystery",
            "The Bermuda Triangle: Vanishing Ships",
            "The Siberian Explosion: Tunguska Event",
            "The Unexplained Hum: Worldwide Phenomenon",
            "The Missing Airliner: MH370 Mystery",
            "The CIA Mind Control: MK-Ultra Files",
            "The Dancing Plague: 1518 Strasbourg",
            "The Radium Girls: Glowing Death",
            "The Prison Study: Stanford 1971",
            "The Town That Went Blind: Minamata",
        ]
    }
    
    selected = evergreen_by_category.get(category, evergreen_by_category['general'])
    print(f"✅ Loaded {len(selected)} category-specific evergreen themes ({category})")
    return selected


def get_real_mystery_trends() -> List[str]:
    """Combine multiple sources for trending topics"""
    
    print("\n" + "="*70)
    print("🔮 FETCHING REAL-TIME MYSTERY TRENDS FOR MYTHICA REPORT")
    print("="*70)
    
    all_trends = []
    source_counts = {}
    
    # Source 1: Google Trends (search trends)
    try:
        google_trends = get_google_trends_mystery()
        all_trends.extend(google_trends)
        source_counts['Google Trends'] = len(google_trends)
    except Exception as e:
        print(f"⚠️ Google Trends error: {e}")

    # Source 2: Google News RSS (NEW - v6.0.9)
    try:
        news_trends = get_google_news_mystery()
        all_trends.extend(news_trends)
        source_counts['Google News'] = len(news_trends)
    except Exception as e:
        print(f"⚠️ Google News error: {e}")

    # Reddit DISABLED (403 Forbidden - requires OAuth API)
    print("⏭️ Skipping Reddit (403 forbidden - requires OAuth API)")

    try:
        youtube_trends = get_youtube_mystery_trends()
        all_trends.extend(youtube_trends)
        source_counts['YouTube'] = len(youtube_trends)
    except Exception as e:
        print(f"⚠️ YouTube error: {e}")
    
    # 🆕 v6.0.9: Category-aware evergreen
    evergreen = get_evergreen_mystery_themes(MYSTERY_CATEGORY)
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


def extract_entities_from_title(title: str) -> set:
    """
    Extract key entities (names, events, codes, numbers) from video titles
    v6.0.3: Fixed 3-word phrase extraction (blacklisted all generic patterns)
    
    STRICT RULES:
    - Only proper nouns BEFORE colon (person/place names)
    - Only specific identifiers (flight numbers, codes)
    - Only known whitelisted entities (case-insensitive)
    - NO verb phrases, NO generic descriptors, NO post-colon content
    
    Returns normalized entity set for duplicate detection.
    """
    import re
    
    entities = set()
    
    # ========================================================================
    # STEP 1: Split title at colon (only extract from BEFORE colon)
    # ========================================================================
    title_before_colon = title.split(':')[0].strip()
    clean_title = title.lower()
    
    # ========================================================================
    # STEP 2: Known entity whitelist (case-insensitive search)
    # ========================================================================
    known_entities = {
        # ✅ PEOPLE
        'asha degree': 'ashadegree',
        'natasha ryan': 'natasharyan',
        'elisa lam': 'elisalam',
        'maura murray': 'mauramurray',
        'jonbenet ramsey': 'jonbenetramsey',
        'jonbenet': 'jonbenetramsey',
        'jimmy hoffa': 'jimmyhoffa',
        'amelia earhart': 'ameliaearhart',
        'db cooper': 'dbcooper',
        'd.b. cooper': 'dbcooper',
        'leah roberts': 'leahroberts',
        'brandon swanson': 'brandonswanson',
        'brian shaffer': 'brianshaffer',
        'brandon lawson': 'brandonlawson',
        'rey rivera': 'reyrivera',
        
        # ✅ EVENTS/CASES
        'flight 19': 'flight19',
        'mh370': 'mh370',
        'mh 370': 'mh370',
        'zodiac': 'zodiac',
        'zodiac killer': 'zodiac',
        'dyatlov pass': 'dyatlovpass',
        'tunguska': 'tunguska',
        'tunguska event': 'tunguska',
        'wow signal': 'wowsignal',
        'cicada 3301': 'cicada3301',
        'lake city quiet pills': 'lakecityquietpills',
        
        # ✅ PLACES
        'roanoke': 'roanoke',
        'roanoke colony': 'roanoke',
        'bermuda triangle': 'bermudatriangle',
        'skinwalker ranch': 'skinwalkerranch',
        'mary celeste': 'maryceleste',
        'flannan isles': 'flannanisles',
        'alcatraz': 'alcatraz',
        'area 51': 'area51',
        
        # ✅ EXPERIMENTS
        'mk ultra': 'mkultra',
        'mkultra': 'mkultra',
        'stanford prison': 'stanfordprison',
        'tuskegee': 'tuskegee',
        'unit 731': 'unit731'
    }
    
    for known_name, normalized in known_entities.items():
        if known_name in clean_title:
            entities.add(normalized)
    
    # ========================================================================
    # STEP 3: Extract proper names ONLY from before colon
    # ========================================================================
    # Pattern matches 2-3 word capitalized phrases
    name_pattern = r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2})\b'
    
    # ❌ COMPREHENSIVE BLACKLIST: All generic 2-3 word phrases
    blacklist = {
        # 2-word generic descriptors
        'the girl', 'the child', 'the teenager', 'the woman', 'the man',
        'the boy', 'the student', 'the worker', 'the driver', 'the patient',
        'the hiker', 'the tourist', 'the hijacker', 'the scientist',
        'the union boss', 'the family', 'the crew', 'the colony',
        'the posts', 'the riches', 'the signal', 'the experiment',
        'the project', 'the case', 'the treasure', 'the code',
        'the mystery', 'the files', 'the broadcasts', 'the space',
        'the ghost', 'the ship', 'the plane', 'the flight',
        
        # 3-word generic phrases (THE PROBLEM!)
        'the teenager who', 'the girl who', 'the woman who', 'the man who',
        'the mystery of', 'the case of', 'the story of', 'the tale of',
        'the posts that', 'the riches that', 'the signal that', 'the colony that',
        'the experiment that', 'the space signal', 'the ghost ship',
        'the lost colony', 'the vanishing of', 'the disappearance of',
        'vanished with their', 'vanished from history', 'turned people to',
        'the question of', 'the secret of', 'the truth about',
        
        # Verb phrases
        'who vanished', 'that vanished', 'which vanished', 'who disappeared',
        'turned people', 'came from', 'went to', 'found in',
        'seen in', 'heard in', 'discovered in'
    }
    
    for match in re.finditer(name_pattern, title_before_colon):
        entity = match.group(1).lower()
        
        if entity not in blacklist:
            # Normalize: "Sarah Johnson" → "sarahjohnson"
            normalized_entity = entity.replace(' ', '')
            
            # Avoid duplicating whitelist entries
            # Check if this entity is already covered by whitelist
            already_whitelisted = False
            for known_key in known_entities.keys():
                if known_key.replace(' ', '') == normalized_entity:
                    already_whitelisted = True
                    break
            
            if not already_whitelisted:
                entities.add(normalized_entity)
    
    # ========================================================================
    # STEP 4: Flight/case numbers (anywhere in title)
    # ========================================================================
    flight_pattern = r'\b(flight\s*\d+|mh\s*\d+|ua\s*\d+|pan\s*am\s*\d+)\b'
    for match in re.finditer(flight_pattern, clean_title):
        entity = match.group(1).replace(' ', '')
        entities.add(entity)
    
    # ========================================================================
    # STEP 5: Codes and signals (anywhere in title)
    # ========================================================================
    code_pattern = r'\b(cicada\s*\d+|wow\s+signal|lake\s+city\s+quiet\s+pills)\b'
    for match in re.finditer(code_pattern, clean_title):
        entity = match.group(1).replace(' ', '')
        entities.add(entity)
    
    return entities

def load_entity_history(history: dict, days: int = 90) -> set:
    """
    Load all entities from recent history
    
    Args:
        history: Content history dict from load_history()
        days: How far back to check (default 90 days)
    
    Returns:
        Set of normalized entity strings
    """
    from datetime import datetime, timedelta
    
    entity_set = set()
    cutoff_date = datetime.now() - timedelta(days=days)
    
    for topic_entry in history.get('topics', []):
        # Check if topic is recent enough
        topic_date_str = topic_entry.get('date')
        if topic_date_str:
            try:
                topic_date = datetime.fromisoformat(topic_date_str)
                if topic_date < cutoff_date:
                    continue  # Skip old topics
            except:
                pass  # If date parsing fails, include it to be safe
        
        # Extract entities from historical title
        title = topic_entry.get('title', '')
        if title:
            entities = extract_entities_from_title(title)
            entity_set.update(entities)
    
    print(f"📂 Loaded {len(entity_set)} unique entities from last {days} days")
    if entity_set:
        # Show sample entities for debugging
        sample = sorted(list(entity_set))[:10]
        print(f"   Sample entities: {sample}")
    
    return entity_set


def has_duplicate_entity(new_title: str, entity_history: set) -> tuple:
    """
    Check if new topic contains entities that already exist in history
    
    Args:
        new_title: Title to check
        entity_history: Set of historical entities
    
    Returns:
        (is_duplicate: bool, matching_entities: list)
    """
    new_entities = extract_entities_from_title(new_title)
    
    if not new_entities:
        # If no entities extracted, allow it (topic is generic/new angle)
        return False, []
    
    # Check each new entity against history
    matches = []
    for entity in new_entities:
        if entity in entity_history:
            matches.append(entity)
    
    # Only block if there's at least one entity match
    if matches:
        return True, matches
    
    return False, []


def filter_and_rank_mystery_trends(trends: List[str], history: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    🚨 ENHANCED v6.0.1: Fixed entity-based duplicate detection
    - Prevents posting same person/event/case multiple times
    - Checks last 90 days of history
    - Blocks at topic selection stage (before script generation)
    - v6.0.1: Removed false positives from verb phrases
    """
    
    if not trends:
        print("⚠️ No trends to filter, using fallback...")
        return get_fallback_mystery_ideas()
    
    print(f"\n🤖 Using Gemini to rank {len(trends)} mystery topics (ENTITY-AWARE v6.0.1)...")

    # Load entity history (last 90 days)
    entity_history = load_entity_history(history, days=90)
    
    previous_titles = [item.get('title', '') for item in history.get('topics', [])[-30:]]

    # 🆕 v6.0.9: Category enforcement
    category_descriptions = {
        'disturbing_medical': 'Medical mysteries, diseases, syndromes, mysterious illnesses, medical anomalies',
        'dark_experiments': 'Unethical experiments, secret research, CIA/government studies, psychological tests',
        'dark_history': 'Mysterious historical events from 1400s-1900s, dark historical phenomena',
        'disappearance': 'Missing persons, vanished ships/planes, unexplained disappearances',
        'phenomena': 'Unexplained phenomena, mysterious signals/lights/sounds, paranormal events',
        'crime': 'Unsolved murders, cold cases, serial killers, true crime mysteries',
        'conspiracy': 'Cover-ups, conspiracies, classified information, secret projects',
        'general': 'All mystery types'
    }
    
    category_desc = category_descriptions.get(MYSTERY_CATEGORY, 'mystery topics')
    
    if MYSTERY_CATEGORY != 'general':
        category_enforcement = f"""

🚨 CRITICAL CATEGORY REQUIREMENT - MUST FOLLOW:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
You are selecting topics for the '{MYSTERY_CATEGORY}' category.

Category focus: {category_desc}

MANDATORY RULES:
1. You MUST prioritize topics that match '{MYSTERY_CATEGORY}'
2. REJECT topics from other categories (disappearances, space events, etc.)
3. If only 3 topics match the category, select those 3 (don't fill with generic)
4. Generic mystery topics (Mary Celeste, DB Cooper, Roanoke) should be REJECTED
   unless they fit the current category

Example for '{MYSTERY_CATEGORY}':
✅ ACCEPT: Topics about {category_desc}
❌ REJECT: Generic disappearances, space mysteries, historical events (unless category is dark_history)

Your output MUST contain topics matching '{MYSTERY_CATEGORY}' ONLY.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
    else:
        category_enforcement = ""
    
    prompt = f"""You are a viral content strategist for "Mythica Report," a YouTube Shorts channel.

ANALYZING RAW TRENDING MYSTERY TOPICS:
{chr(10).join(f"- {t}" for t in trends[:30])}

**CRITICAL: AVOID RECENTLY COVERED TOPICS:**
{chr(10).join(f"- {title}" for title in previous_titles) if previous_titles else "None yet."}
{category_enforcement}

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
                
                # 🚨 v6.0.1: Fixed entity-based duplicate detection
                is_duplicate, matching_entities = has_duplicate_entity(title, entity_history)
                
                if is_duplicate:
                    print(f"   🚫 DUPLICATE BLOCKED: '{title}'")
                    print(f"      Entity match: {', '.join(matching_entities)}")
                    print(f"      This topic was already covered in last 90 days")
                    continue  # Skip this topic entirely
                
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
                print("⚠️ All topics were duplicates or rejected - retrying...")
                raise ValueError("All topics rejected by duplicate/pattern validation")

            print(f"✅ Gemini ranked {len(trending_ideas)} UNIQUE topics (v6.0.1 entity-verified)")
            for i, idea in enumerate(trending_ideas, 1):
                entities = extract_entities_from_title(idea['topic_title'])
                entity_str = f" [{', '.join(list(entities)[:2])}]" if entities else " [no entities]"
                print(f"   {i}. [{idea['viral_score']}] {idea['topic_title'][:60]}{entity_str}")
            
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
        "source": f"google_trends + reddit + youtube + evergreen + gemini_ranking (category: {MYSTERY_CATEGORY})",
        "version": "6.0.9_category_aware",
        "category": MYSTERY_CATEGORY  # 🆕 Track which category was fetched
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
        print(f"🔮 TOP VIRAL MYSTERY IDEAS (RETENTION-OPTIMIZED v6.0.1)")
        print("="*70)
        
        for i, idea in enumerate(trending_ideas, 1):
            print(f"\n💎 IDEA {i}:")
            print(f"   Title: {idea['topic_title']}")
            print(f"   Viral Score: {idea.get('viral_score', 'N/A')}/100")
            print(f"   Hook: {idea.get('story_hook', 'N/A')}")
            print(f"   Mystery: {idea.get('core_mystery', 'N/A')}")
        
        save_trending_data(trending_ideas)
        
        print(f"\n✅ TRENDING DATA READY (v6.0.1)")
        print(f"   Quality: Retention-optimized title patterns")
        print(f"   Validation: Name-first patterns rejected")
        print(f"   Duplicate Detection: Entity-based (fixed false positives)")
        
    else:
        print("\n❌ Could not retrieve any trending ideas")
        exit(1)