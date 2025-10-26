import os
import json
from datetime import datetime
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from googleapiclient.errors import HttpError
from google.oauth2.credentials import Credentials
from tenacity import retry, stop_after_attempt, wait_exponential
from PIL import Image
import re 

TMP = os.getenv("GITHUB_WORKSPACE", ".") + "/tmp"
VIDEO = os.path.join(TMP, "short.mp4")
THUMB = os.path.join(TMP, "thumbnail.png")
READY_VIDEO = os.path.join(TMP, "short_ready.mp4")
UPLOAD_LOG = os.path.join(TMP, "upload_history.json")

# 🔮 MYSTERY CHANNEL CONFIG
CHANNEL_NAME = "Mythica Report"
CHANNEL_TAGLINE = "Unsolved mysteries that defy explanation 🔮"

# ---- Load Global Metadata ONCE ----
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

# ---- Step 1: Validate video ----
if not os.path.exists(VIDEO):
    raise FileNotFoundError(f"Video file not found: {VIDEO}")

video_size_mb = os.path.getsize(VIDEO) / (1024 * 1024)
print(f"📹 Mystery video file found: {VIDEO} ({video_size_mb:.2f} MB)")
if video_size_mb < 0.1:
    raise ValueError("Video file is too small, likely corrupted")

# ---- Step 2: Rename video to safe filename ----
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
    else:
        print("⚠️ Video file not found before rename.")
else:
    print("🎬 Video already has the correct filename.")

# ---- Step 3: Authenticate ----
try:
    creds = Credentials(
        None,
        refresh_token=os.getenv("GOOGLE_REFRESH_TOKEN"),
        token_uri="https://oauth2.googleapis.com/token",
        client_id=os.getenv("GOOGLE_CLIENT_ID"),
        client_secret=os.getenv("GOOGLE_CLIENT_SECRET"),
        scopes=["https://www.googleapis.com/auth/youtube.upload"]
    )
    youtube = build("youtube", "v3", credentials=creds, cache_discovery=False)
    print("✅ YouTube API authenticated")
except Exception as e:
    print(f"❌ Authentication failed: {e}")
    raise

# ---- Step 4: 🔮 Prepare MYSTERY-OPTIMIZED metadata ----
# Enhanced description with mystery-specific CTAs and keywords
enhanced_description = f"""{description}

{' '.join(hashtags)}

🔮 {CHANNEL_TAGLINE}

---
🔍 New unsolved mysteries daily!
👁️ Investigating the unexplained, the mysterious, and the impossible
🎭 Follow {CHANNEL_NAME} for weekly deep dives into the unknown
📚 Topics: Unsolved Cases • Paranormal • True Crime • Ancient Mysteries

Follow Mythica Report:
YouTube   : @MythicaReport
Instagram : @MythicaReport
TikTok    : @MythicaReport
Facebook  : Mythica Report

Mystery Type: {mystery_category.title()}
Created: {datetime.now().strftime('%Y-%m-%d')}
Category: Mystery & Investigation

⚠️ Content Warning: This channel covers unsolved mysteries, true crime, and unexplained phenomena. 
Some cases involve missing persons, deaths, and disturbing events. Viewer discretion advised.
All information presented is based on public records and available evidence.
"""

# 🔮 MYSTERY-SPECIFIC TAGS (optimized for discovery)
mystery_base_tags = [
    "mystery",
    "unsolved mystery",
    "true crime",
    "unexplained",
    "paranormal",
    "cold case",
    "investigation",
    "unsolved",
    "disappeared",
    "mysterious",
    "conspiracy",
    "documentary",
    "mystery shorts",
    "true crime shorts",
    "unsolved cases"
]

# Category-specific tags
category_tags = {
    'disappearance': ["missing person", "vanished", "disappeared", "missing", "lost"],
    'crime': ["true crime", "murder mystery", "cold case", "detective", "investigation"],
    'paranormal': ["paranormal", "supernatural", "ghost", "haunted", "unexplained"],
    'historical': ["ancient mystery", "historical", "archaeology", "ancient", "artifact"],
    'conspiracy': ["conspiracy", "cover up", "declassified", "government secrets", "exposed"],
    'cryptids': ["cryptid", "creature", "unknown", "bigfoot", "mysterious creature"]
}

# Combine base tags with category-specific tags
tags = mystery_base_tags.copy()
if mystery_category in category_tags:
    tags.extend(category_tags[mystery_category][:5])

# Add hashtags from script
if hashtags:
    tags.extend([tag.replace('#', '').lower() for tag in hashtags[:10]])

# Add generic viral tags
tags.extend(["shorts", "viral", "scary", "creepy"])

# Remove duplicates and limit to 15 tags (YouTube best practice)
tags = list(dict.fromkeys(tags))[:15]

print(f"📝 Mystery metadata ready:")
print(f"   Title: {title}")
print(f"   Channel: {CHANNEL_NAME}")
print(f"   Mystery Type: {mystery_category}")
print(f"   Tags: {', '.join(tags[:10])}...")
print(f"   Hashtags: {' '.join(hashtags[:5])}...")

snippet = {
    "title": title[:100],  # YouTube limit
    "description": enhanced_description[:5000],  # YouTube limit
    "tags": tags,
    "categoryId": "24"  # 🔮 Category 24 = "Entertainment" (best for mystery/documentary content)
    # Alternative: "22" = People & Blogs
}

body = {
    "snippet": snippet,
    "status": {
        "privacyStatus": "public",
        "selfDeclaredMadeForKids": False,
        "madeForKids": False
    }
}

print(f"📤 Uploading mystery video to YouTube...")

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=4, max=60))
def upload_video(youtube_client, video_path, metadata):
    media = MediaFileUpload(
        video_path,
        chunksize=1024*1024,
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
        status, response = request.next_chunk()
        if status:
            progress = int(status.progress() * 100)
            if progress != last_progress and progress % 10 == 0:
                print(f"⏳ Upload progress: {progress}%")
                last_progress = progress
    return response

try:
    print("🚀 Starting mystery video upload...")
    result = upload_video(youtube, VIDEO, body)
    video_id = result["id"]
    video_url = f"https://www.youtube.com/watch?v={video_id}"
    shorts_url = f"https://www.youtube.com/shorts/{video_id}"
    
    print(f"✅ Mystery video uploaded successfully!")
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
    raise

# ---- Step 6: Set thumbnail (desktop view) ----
if os.path.exists(THUMB):
    try:
        print("🖼️ Setting noir mystery thumbnail for desktop views...")
        thumb_size_mb = os.path.getsize(THUMB) / (1024*1024)
        if thumb_size_mb > 2:
            print(f"⚠️ Compressing thumbnail ({thumb_size_mb:.2f}MB)...")
            img = Image.open(THUMB)
            # 🔮 Optimize thumbnail - high quality for dramatic mystery imagery
            img.save(THUMB, quality=92, optimize=True)
        
        youtube.thumbnails().set(
            videoId=video_id, 
            media_body=MediaFileUpload(THUMB)
        ).execute()
        print("✅ Mystery thumbnail set successfully (desktop view).")
    except Exception as e:
        print(f"⚠️ Thumbnail upload failed: {e}")
else:
    print("⚠️ No thumbnail file found, skipping thumbnail set.")

# ---- Step 7: 🔮 Save upload history with mystery analytics ----
upload_metadata = {
    "video_id": video_id,
    "title": title,
    "topic": topic,
    "mystery_category": mystery_category,
    "channel": CHANNEL_NAME,
    "upload_date": datetime.now().isoformat(),
    "video_url": video_url,
    "shorts_url": shorts_url,
    "hashtags": hashtags,
    "file_size_mb": round(video_size_mb, 2),
    "tags": tags,
    "category": "Mystery & Entertainment",
    "content_type": "mystery_short",
    "estimated_duration": data.get("estimated_duration", 0),
    "word_count": data.get("word_count", 0)
}

history = []
if os.path.exists(UPLOAD_LOG):
    try:
        with open(UPLOAD_LOG, 'r') as f:
            history = json.load(f)
    except:
        history = []

history.append(upload_metadata)
history = history[-100:]  # Keep last 100 uploads

with open(UPLOAD_LOG, 'w') as f:
    json.dump(history, f, indent=2)

# 🔮 Analytics summary
total_uploads = len(history)
mystery_type_counts = {}
for h in history:
    mcat = h.get('mystery_category', 'unknown')
    mystery_type_counts[mcat] = mystery_type_counts.get(mcat, 0) + 1

print(f"\n📊 Channel Stats: {total_uploads} mystery videos uploaded total")
if mystery_type_counts:
    print(f"📈 Mystery Types:")
    for mtype, count in sorted(mystery_type_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"   {mtype.title()}: {count} videos")

print("\n" + "="*70)
print("🎉 MYSTERY VIDEO UPLOAD COMPLETE!")
print("="*70)
print(f"🔮 Channel: {CHANNEL_NAME}")
print(f"📹 Title: {title}")
print(f"🏷️  Topic: {topic}")
print(f"🎭 Mystery Type: {mystery_category}")
print(f"🆔 Video ID: {video_id}")
print(f"🔗 Shorts URL: {shorts_url}")
print(f"#️⃣  Hashtags: {' '.join(hashtags[:5])}")
print(f"🏷️  Tags: {', '.join(tags[:8])}...")
print("="*70)
print("\n💡 Mystery Channel Best Practices:")
print("   • Best posting time: 7-9 PM (evening mystery hour) or 10 PM-12 AM (late night)")
print("   • Peak season: October (Halloween), Year-round interest")
print("   • Engage with comments FAST - mystery fans love theories!")
print("   • Pin a comment asking 'What do YOU think happened?' for engagement")
print("   • Cross-post to TikTok 1 hour after YouTube")
print("   • Create playlists by mystery type for binge-watching")
print("   • Use end screens to link related mysteries")
print(f"\n🔗 Share this URL: {shorts_url}")
print("🔮 Keep investigating the unexplained! 👁️")