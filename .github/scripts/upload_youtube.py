import os
import json
from datetime import datetime
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from googleapiclient.errors import HttpError
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from tenacity import retry, stop_after_attempt, wait_exponential
from PIL import Image
import re 

TMP = os.getenv("GITHUB_WORKSPACE", ".") + "/tmp"
VIDEO = os.path.join(TMP, "short.mp4")
THUMB = os.path.join(TMP, "thumbnail.png")
READY_VIDEO = os.path.join(TMP, "short_ready.mp4")
UPLOAD_LOG = os.path.join(TMP, "upload_history.json")

# Load metadata
try:
    with open(os.path.join(TMP, "script.json"), "r", encoding="utf-8") as f:
        data = json.load(f)
except FileNotFoundError:
    print("❌ Error: script.json not found.")
    raise

title = data.get("title", "Mystery Case")
description = data.get("description", f"{title}")
hashtags = data.get("hashtags", ["#mystery", "#unsolved", "#truecrime", "#shorts"])
topic = data.get("topic", "mystery")
mystery_category = data.get("mystery_category", "disappearance")

print(f"\n{'='*70}")
print(f"🎬 MYTHICA REPORT UPLOAD v6.0.10 (CATEGORY-OPTIMIZED)")
print(f"{'='*70}")
print(f"📹 Title: {title}")
print(f"🎭 Category: {mystery_category}")
print(f"{'='*70}\n")

# Validate video
if not os.path.exists(VIDEO):
    raise FileNotFoundError(f"Video file not found: {VIDEO}")

video_size_mb = os.path.getsize(VIDEO) / (1024 * 1024)
print(f"📹 Mystery video file found: {VIDEO} ({video_size_mb:.2f} MB)")

if video_size_mb < 0.1:
    raise ValueError("Video file is too small, likely corrupted")

# Rename video to safe filename
safe_title = re.sub(r'[^\w\s-]', '', title).strip().replace(' ', '_')
video_output_path = os.path.join(TMP, f"{safe_title}.mp4")

if VIDEO != video_output_path:
    if os.path.exists(VIDEO):
        try:
            os.rename(VIDEO, video_output_path)
            VIDEO = video_output_path
            print(f"🎬 Final mystery video renamed to: {video_output_path}")
        except Exception as e:
            print(f"⚠️ Renaming failed: {e}. Using original path.")

# Authenticate with token refresh
try:
    creds = Credentials(
        None,
        refresh_token=os.getenv("GOOGLE_REFRESH_TOKEN"),
        token_uri="https://oauth2.googleapis.com/token",
        client_id=os.getenv("GOOGLE_CLIENT_ID"),
        client_secret=os.getenv("GOOGLE_CLIENT_SECRET"),
        scopes=["https://www.googleapis.com/auth/youtube.upload"]
    )
    
    # ✅ Explicitly refresh token before use
    if creds.expired and creds.refresh_token:
        print("🔄 Refreshing authentication token...")
        creds.refresh(Request())
    
    youtube = build("youtube", "v3", credentials=creds, cache_discovery=False)
    print("✅ YouTube API authenticated")
except Exception as e:
    print(f"❌ Authentication failed: {e}")
    raise


# ============================================================================
# 🆕 v6.0.10: CATEGORY-AWARE TAGGING SYSTEM
# ============================================================================

def get_category_specific_tags(category, title_text):
    """
    Generate optimized tags for each mystery category
    
    Strategy:
    1. Category-specific base tags (niche targeting)
    2. SEO keywords (high search volume terms)
    3. Trending topic tags (extracted from title)
    4. Generic mystery tags (broad reach)
    
    Returns 15-20 tags optimized for YouTube algorithm
    """
    
    # ========================================================================
    # CATEGORY-SPECIFIC BASE TAGS (Primary niche targeting)
    # ========================================================================
    category_base_tags = {
        'dark_history': [
            "dark history",
            "historical mystery",
            "unexplained history",
            "history mystery shorts",
            "dark past",
            "historical horror"
        ],
        
        'disturbing_medical': [
            "medical mystery",
            "rare disease",
            "medical horror",
            "disturbing medical cases",
            "medical anomaly",
            "body horror"
        ],
        
        'dark_experiments': [
            "secret experiments",
            "government experiments",
            "conspiracy theory",
            "classified secrets",
            "mk ultra",
            "cia secrets"
        ],
        
        'disappearance': [
            "missing person",
            "disappeared",
            "unsolved disappearance",
            "vanished without trace",
            "missing case",
            "true crime"
        ],
        
        'crime': [
            "true crime",
            "unsolved murder",
            "cold case",
            "crime mystery",
            "detective story",
            "murder mystery"
        ],
        
        'phenomena': [
            "unexplained phenomena",
            "paranormal",
            "mysterious signals",
            "strange phenomena",
            "paranormal mystery",
            "supernatural"
        ],
        
        'conspiracy': [
            "conspiracy theory",
            "conspiracy",
            "government secrets",
            "cover up",
            "declassified",
            "conspiracy mystery"
        ],
        
        'historical': [
            "historical mystery",
            "ancient mystery",
            "history",
            "archaeological mystery",
            "historical secrets"
        ]
    }
    
    # ========================================================================
    # SEO KEYWORDS (High search volume terms per category)
    # ========================================================================
    seo_keywords = {
        'dark_history': [
            "radium girls",           # 50K+ monthly searches
            "dancing plague",         # 30K+ monthly searches
            "historical events",
            "dark day 1780",
            "historical horror stories"
        ],
        
        'disturbing_medical': [
            "fatal familial insomnia",  # 80K+ monthly searches
            "rare medical conditions",  # 100K+ monthly searches
            "cotard delusion",
            "medical horror stories",
            "strange diseases"
        ],
        
        'dark_experiments': [
            "mk ultra",               # 200K+ monthly searches
            "stanford prison experiment", # 150K+ monthly searches
            "tuskegee experiment",
            "human experiments",
            "government cover up"
        ],
        
        'disappearance': [
            "missing 411",            # 100K+ monthly searches
            "unsolved mysteries",     # 500K+ monthly searches
            "dyatlov pass",
            "missing persons cases",
            "unexplained disappearances"
        ],
        
        'crime': [
            "unsolved mysteries",     # 500K+ monthly searches
            "true crime stories",     # 300K+ monthly searches
            "zodiac killer",
            "cold case files",
            "murder mysteries"
        ],
        
        'phenomena': [
            "wow signal",             # 40K+ monthly searches
            "unexplained mysteries",  # 200K+ monthly searches
            "paranormal activity",
            "strange sounds",
            "mysterious signals"
        ],
        
        'conspiracy': [
            "conspiracy theories",    # 1M+ monthly searches
            "government secrets",
            "area 51",
            "illuminati",
            "conspiracy documentary"
        ],
        
        'historical': [
            "ancient mysteries",      # 150K+ monthly searches
            "lost civilizations",
            "archaeological discoveries",
            "historical secrets"
        ]
    }
    
    # ========================================================================
    # TRENDING TOPIC EXTRACTION (From video title)
    # ========================================================================
    trending_tags = []
    title_lower = title_text.lower()
    
    # Extract proper nouns and specific cases from title
    # Example: "The Radium Girls Who Glowed" → extract "radium girls"
    
    trending_keywords = {
        # Dark History
        'radium': ['radium girls', 'radium poisoning'],
        'dancing': ['dancing plague', 'dancing mania'],
        'dark day': ['dark day 1780', 'new england dark day'],
        'roanoke': ['roanoke colony', 'lost colony'],
        
        # Medical
        'insomnia': ['fatal insomnia', 'fatal familial insomnia'],
        'fop': ['fibrodysplasia', 'stone man syndrome'],
        'kuru': ['kuru disease', 'laughing death'],
        'minamata': ['minamata disease', 'mercury poisoning'],
        
        # Experiments
        'mk': ['mk ultra', 'mkultra'],
        'stanford': ['stanford prison', 'stanford experiment'],
        'tuskegee': ['tuskegee experiment', 'tuskegee study'],
        'unit 731': ['unit 731', 'japanese experiments'],
        
        # Disappearances
        'flight 19': ['flight 19', 'bermuda triangle'],
        'mh370': ['mh370', 'malaysia airlines'],
        'db cooper': ['db cooper', 'd.b. cooper'],
        'dyatlov': ['dyatlov pass', 'dyatlov incident'],
        
        # Crime
        'zodiac': ['zodiac killer', 'zodiac case'],
        'ripper': ['jack the ripper', 'whitechapel murders'],
        'black dahlia': ['black dahlia', 'elizabeth short'],
        
        # Phenomena
        'wow': ['wow signal', 'seti signal'],
        'bloop': ['bloop sound', 'underwater sound'],
        'hessdalen': ['hessdalen lights', 'norway lights']
    }
    
    for keyword, tag_options in trending_keywords.items():
        if keyword in title_lower:
            trending_tags.extend(tag_options[:2])  # Add up to 2 related tags
    
    # ========================================================================
    # GENERIC MYSTERY TAGS (Broad reach, all videos)
    # ========================================================================
    generic_tags = [
        "mystery",
        "unsolved",
        "shorts",
        "mystery shorts",
        "unexplained"
    ]
    
    # ========================================================================
    # ASSEMBLE FINAL TAG LIST (Priority order)
    # ========================================================================
    
    # Get category-specific tags
    base_tags = category_base_tags.get(category, ["mystery", "unsolved"])
    seo_tags = seo_keywords.get(category, [])
    
    # Combine in priority order
    final_tags = []
    
    # 1. Category base tags (6 tags - highest priority)
    final_tags.extend(base_tags[:6])
    
    # 2. Trending topic tags (3-4 tags - specific case targeting)
    final_tags.extend(trending_tags[:4])
    
    # 3. SEO keywords (4-5 tags - search volume optimization)
    final_tags.extend(seo_tags[:5])
    
    # 4. Generic mystery tags (3-4 tags - broad reach)
    final_tags.extend(generic_tags[:4])
    
    # 5. Hashtags from script (if any remaining space)
    if hashtags:
        clean_hashtags = [tag.replace('#', '').lower() for tag in hashtags[:3]]
        final_tags.extend(clean_hashtags)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_tags = []
    for tag in final_tags:
        tag_lower = tag.lower()
        if tag_lower not in seen:
            seen.add(tag_lower)
            unique_tags.append(tag)
    
    # YouTube limit: 500 characters total, ~15-20 tags
    final_tags = unique_tags[:20]
    
    return final_tags


def get_category_description_template(category):
    """
    Generate category-optimized description templates
    
    Includes:
    - Category-specific hook
    - SEO keywords naturally integrated
    - Branded channel messaging
    """
    
    templates = {
        'dark_history': """🕰️ {description}

Dive into the darkest mysteries of history - events that defy explanation and haunt us to this day.

{hashtags}

━━━━━━━━━━━━━━━━━━━━
🔮 MYTHICA REPORT
Unsolved historical mysteries, dark events, and unexplained phenomena from the past.

📅 New dark history mysteries every Monday
🔔 Subscribe for weekly mysteries that history tried to forget

#DarkHistory #HistoricalMystery #UnexplainedHistory #Shorts
━━━━━━━━━━━━━━━━━━━━""",

        'disturbing_medical': """🏥 {description}

Medical mysteries that baffle doctors and defy modern science. Real cases. Real horror.

{hashtags}

━━━━━━━━━━━━━━━━━━━━
🔮 MYTHICA REPORT
Disturbing medical cases, rare diseases, and unexplained conditions.

📅 New medical mysteries every Wednesday
🔔 Subscribe for cases that medicine can't explain

#MedicalMystery #RareDisease #MedicalHorror #Shorts
━━━━━━━━━━━━━━━━━━━━""",

        'dark_experiments': """🔬 {description}

Declassified secrets, unethical experiments, and government cover-ups that were hidden for decades.

{hashtags}

━━━━━━━━━━━━━━━━━━━━
🔮 MYTHICA REPORT
Secret experiments, classified research, and government conspiracies exposed.

📅 New dark experiments every Thursday
🔔 Subscribe for secrets they tried to bury

#SecretExperiments #MKUltra #GovernmentSecrets #Shorts
━━━━━━━━━━━━━━━━━━━━""",

        'disappearance': """👤 {description}

Vanished without a trace. No evidence. No answers. Only questions.

{hashtags}

━━━━━━━━━━━━━━━━━━━━
🔮 MYTHICA REPORT
Unsolved disappearances and missing persons cases that defy explanation.

📅 New disappearance mysteries every Tuesday & Friday
🔔 Subscribe for cases that remain unsolved

#MissingPerson #Disappeared #UnsolvedMystery #Shorts
━━━━━━━━━━━━━━━━━━━━""",

        'crime': """🔪 {description}

Unsolved murders, cold cases, and mysteries that investigators couldn't crack.

{hashtags}

━━━━━━━━━━━━━━━━━━━━
🔮 MYTHICA REPORT
True crime mysteries, cold cases, and unsolved murders.

📅 New true crime cases every Saturday
🔔 Subscribe for mysteries that remain unsolved

#TrueCrime #ColdCase #UnsolvedMurder #Shorts
━━━━━━━━━━━━━━━━━━━━""",

        'phenomena': """👽 {description}

Unexplained signals, mysterious lights, and phenomena that science cannot explain.

{hashtags}

━━━━━━━━━━━━━━━━━━━━
🔮 MYTHICA REPORT
Paranormal phenomena, mysterious signals, and unexplained events.

📅 New phenomena mysteries every Sunday
🔔 Subscribe for mysteries beyond explanation

#UnexplainedPhenomena #Paranormal #MysteriousSignals #Shorts
━━━━━━━━━━━━━━━━━━━━""",

        'conspiracy': """🕵️ {description}

Conspiracy theories, government secrets, and cover-ups that make you question everything.

{hashtags}

━━━━━━━━━━━━━━━━━━━━
🔮 MYTHICA REPORT
Conspiracies, government secrets, and mysteries they don't want you to know.

📅 New conspiracy mysteries weekly
🔔 Subscribe for truth hidden in plain sight

#Conspiracy #GovernmentSecrets #CoverUp #Shorts
━━━━━━━━━━━━━━━━━━━━"""
    }
    
    return templates.get(category, templates['disappearance'])


# ============================================================================
# GENERATE OPTIMIZED METADATA
# ============================================================================

print(f"\n🏷️ GENERATING CATEGORY-OPTIMIZED TAGS...")
print(f"   Category: {mystery_category}")

# Generate category-specific tags
tags = get_category_specific_tags(mystery_category, title)

print(f"\n📋 OPTIMIZED TAG STRATEGY:")
print(f"   Total tags: {len(tags)}")
print(f"   Tags: {tags[:10]}...")  # Show first 10

# Generate category-specific description
description_template = get_category_description_template(mystery_category)
enhanced_description = description_template.format(
    description=description,
    hashtags=' '.join(hashtags)
)

# Ensure description under 5000 chars
if len(enhanced_description) > 5000:
    enhanced_description = enhanced_description[:4997] + "..."

print(f"\n📝 DESCRIPTION LENGTH: {len(enhanced_description)} chars")

# ============================================================================
# PREPARE UPLOAD METADATA
# ============================================================================

snippet = {
    "title": title[:100],
    "description": enhanced_description,
    "tags": tags,
    "categoryId": "24"  # Entertainment category
}

body = {
    "snippet": snippet,
    "status": {
        "privacyStatus": "public",
        "selfDeclaredMadeForKids": False,
        "madeForKids": False
    }
}

print(f"\n📤 Uploading mystery video to YouTube...")

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=4, max=60))
def upload_video(youtube_client, video_path, metadata):
    # ✅ INCREASED CHUNK SIZE from 1MB to 10MB
    media = MediaFileUpload(
        video_path,
        chunksize=10*1024*1024,  # 10MB chunks
        resumable=True,
        mimetype="video/mp4"
    )
    
    request = youtube_client.videos().insert(
        part="snippet,status",
        body=metadata,
        media_body=media
    )
    
    response = None
    last_progress = 0
    
    while response is None:
        try:
            status, response = request.next_chunk()
            if status:
                progress = int(status.progress() * 100)
                if progress != last_progress and progress % 10 == 0:
                    print(f"⏳ Upload progress: {progress}%")
                    last_progress = progress
        except HttpError as e:
            if e.resp.status in [500, 502, 503, 504]:
                print(f"⚠️ Server error {e.resp.status}, retrying chunk...")
                raise
            else:
                raise
    
    return response

try:
    print("🚀 Starting mystery video upload...")
    result = upload_video(youtube, VIDEO, body)
    video_id = result["id"]
    video_url = f"https://www.youtube.com/watch?v={video_id}"
    shorts_url = f"https://www.youtube.com/shorts/{video_id}"
    
    print(f"\n✅ Mystery video uploaded successfully!")
    print(f"   Video ID: {video_id}")
    print(f"   Watch URL: {video_url}")
    print(f"   Shorts URL: {shorts_url}")

except HttpError as e:
    print(f"❌ HTTP error during upload: {e}")
    error_content = e.content.decode() if hasattr(e, 'content') else str(e)
    print(f"   Error details: {error_content}")
    raise
except Exception as e:
    print(f"❌ Upload failed: {e}")
    import traceback
    traceback.print_exc()
    raise

# Set thumbnail
if os.path.exists(THUMB):
    try:
        print("\n🖼️ Setting thumbnail...")
        youtube.thumbnails().set(
            videoId=video_id, 
            media_body=MediaFileUpload(THUMB)
        ).execute()
        print("✅ Thumbnail set successfully.")
    except Exception as e:
        print(f"⚠️ Thumbnail upload failed: {e}")
else:
    print("⚠️ No thumbnail file found.")

# Save to upload history
upload_metadata = {
    "video_id": video_id,
    "title": title,
    "topic": topic,
    "mystery_category": mystery_category,
    "upload_date": datetime.now().isoformat(),
    "video_url": video_url,
    "shorts_url": shorts_url,
    "hashtags": hashtags,
    "file_size_mb": round(video_size_mb, 2),
    "tags": tags,
    "tag_count": len(tags),
    "description_length": len(enhanced_description)
}

history = []
if os.path.exists(UPLOAD_LOG):
    try:
        with open(UPLOAD_LOG, 'r') as f:
            history = json.load(f)
    except:
        history = []

history.append(upload_metadata)
history = history[-100:]

with open(UPLOAD_LOG, 'w') as f:
    json.dump(history, f, indent=2)

print(f"\n{'='*70}")
print(f"✅ UPLOAD COMPLETE (v6.0.10 - CATEGORY-OPTIMIZED)")
print(f"{'='*70}")
print(f"   Video ID: {video_id}")
print(f"   Shorts URL: {shorts_url}")
print(f"   Category: {mystery_category}")
print(f"   Tags: {len(tags)} optimized tags")
print(f"   Description: {len(enhanced_description)} chars")
print(f"{'='*70}\n")