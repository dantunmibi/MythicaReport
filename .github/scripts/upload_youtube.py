import os
import json
from datetime import datetime
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from googleapiclient.errors import HttpError
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request  # ✅ ADD THIS
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

# Prepare metadata
enhanced_description = f"""{description}

{' '.join(hashtags)}

🔮 Unsolved mysteries that defy explanation

---
🔍 New unsolved mysteries daily!
"""

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

category_tags = {
    'disappearance': ["missing person", "vanished", "disappeared", "missing", "lost"],
    'crime': ["true crime", "murder mystery", "cold case", "detective", "investigation"],
    'paranormal': ["paranormal", "supernatural", "ghost", "haunted", "unexplained"],
    'historical': ["ancient mystery", "historical", "archaeology", "ancient", "artifact"],
    'conspiracy': ["conspiracy", "cover up", "declassified", "government secrets", "exposed"],
    'cryptids': ["cryptid", "creature", "unknown", "bigfoot", "mysterious creature"]
}

tags = mystery_base_tags.copy()
if mystery_category in category_tags:
    tags.extend(category_tags[mystery_category][:5])

if hashtags:
    tags.extend([tag.replace('#', '').lower() for tag in hashtags[:10]])

tags.extend(["shorts", "viral", "scary", "creepy"])
tags = list(dict.fromkeys(tags))[:15]

snippet = {
    "title": title[:100],
    "description": enhanced_description[:5000],
    "tags": tags,
    "categoryId": "24"
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
    import traceback
    traceback.print_exc()
    raise

# Set thumbnail
if os.path.exists(THUMB):
    try:
        print("🖼️ Setting thumbnail...")
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

print(f"\n✅ UPLOAD COMPLETE!")
print(f"   Video ID: {video_id}")
print(f"   Shorts URL: {shorts_url}")