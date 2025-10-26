#!/usr/bin/env python3
"""
🔍 Create Mystery Video - PRODUCTION VERSION WITH MUSIC
Features:
- Near-perfect audio-video synchronization (<50ms drift)
- Dark ambient background music with dynamic volume
- Film noir color grading (desaturated, high contrast, vignette, grain)
- Documentary-style mysterious imagery
- Paragraph-based narrative timing (not bullets)
- Audio timing metadata integration
- Multiple AI provider fallbacks
- Minimal text with typewriter font aesthetic
"""

import os
import json
import requests
from moviepy import *
import platform
from tenacity import retry, stop_after_attempt, wait_exponential
from pydub import AudioSegment
from time import sleep
import time
from PIL import Image, ImageDraw, ImageFont, ImageEnhance, ImageFilter
import random
import subprocess
import sys
import numpy as np

TMP = os.getenv("GITHUB_WORKSPACE", ".") + "/tmp"
OUT = os.path.join(TMP, "short.mp4")
audio_path = os.path.join(TMP, "voice.mp3")
w, h = 1080, 1920

# Safe zones for text (mystery uses less text)
SAFE_ZONE_MARGIN = 120
TEXT_MAX_WIDTH = w - (2 * SAFE_ZONE_MARGIN)

# 🔍 MYSTERY NOIR COLOR PALETTE
NOIR_COLORS = {
    'deep_black': (15, 15, 20),
    'dark_slate': (30, 35, 45),
    'crimson_accent': (185, 28, 28),
    'aged_gold': (212, 175, 55),
    'fog_gray': (80, 85, 90),
    'shadow_blue': (25, 35, 50),
    'evidence_tan': (200, 180, 150),
}

# 🎵 Import music system
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

MUSIC_AVAILABLE = False
try:
    from download_music import get_music_for_scene, MUSIC_DIR, download_track, MUSIC_LIBRARY
    MUSIC_AVAILABLE = True
    print("✅ Music system imported successfully")
except ImportError as e:
    print(f"⚠️ Music system not available: {e}")
    print("   Videos will be created without background music")


def get_font_path():
    """Get serif/typewriter font for mystery aesthetic"""
    system = platform.system()
    if system == "Windows":
        # Prefer Courier New (typewriter) or Georgia (serif)
        fonts = [
            "C:/Windows/Fonts/cour.ttf",      # Courier New
            "C:/Windows/Fonts/georgia.ttf",   # Georgia
            "C:/Windows/Fonts/times.ttf",     # Times New Roman
        ]
        for font in fonts:
            if os.path.exists(font):
                return font
        return "C:/Windows/Fonts/arial.ttf"
        
    elif system == "Darwin":
        return "/System/Library/Fonts/Supplemental/Courier New.ttf"
        
    else:
        # Linux - prefer serif fonts for mystery
        font_options = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ]
        for font in font_options:
            if os.path.exists(font):
                return font
        return "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


FONT = get_font_path()
print(f"📝 Using font: {FONT}")

# Load script
with open(os.path.join(TMP, "script.json"), "r", encoding="utf-8") as f:
    data = json.load(f)

title = data.get("title", "Mystery")
hook = data.get("hook", "")

# ✅ MYSTERY VERSION: Read 'script' field instead of 'bullets'
full_script = data.get("script", "")
if not full_script:
    # Fallback if old format
    bullets = data.get("bullets", [])
    cta = data.get("cta", "")
    full_script = f"{hook}\n\n{' '.join(bullets)}\n\n{cta}"
else:
    cta = data.get("cta", "")

# Split script into paragraphs (natural narrative sections)
paragraphs = [p.strip() for p in full_script.split('\n\n') if p.strip()]

topic = data.get("topic", "mystery")
content_type = data.get("content_type", "evening_prime")
mystery_category = data.get("mystery_category", "disappearance")
visual_prompts = data.get("visual_prompts", [])

print(f"🔍 Creating {content_type} mystery video ({mystery_category})")
print(f"   Narrative sections: {len(paragraphs)}")


def load_audio_timing():
    """Load optimized audio timing metadata from TTS generation"""
    timing_path = os.path.join(TMP, "audio_timing.json")
    
    if os.path.exists(timing_path):
        try:
            with open(timing_path, 'r') as f:
                timing_data = json.load(f)
            
            if timing_data.get('optimized'):
                print("✅ Loaded optimized audio timing metadata")
                print(f"   Total duration: {timing_data['total_duration']:.2f}s")
                print(f"   Sections: {len(timing_data['sections'])}")
                return timing_data
            else:
                print("⚠️ Timing metadata not optimized, using fallback")
                return None
                
        except Exception as e:
            print(f"⚠️ Could not load timing metadata: {e}")
            return None
    
    print("⚠️ No timing metadata found, using estimation")
    return None


def load_audio_metadata():
    """Load audio metadata"""
    metadata_path = os.path.join(TMP, "audio_metadata.json")
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return None
    return None


def get_section_duration_from_timing(section_name, timing_data):
    """Get duration for a specific section from timing metadata"""
    if not timing_data or 'sections' not in timing_data:
        return None
    
    for section in timing_data['sections']:
        if section['name'] == section_name:
            return section['duration']
    
    return None


def estimate_duration_fallback(text, audio_duration, all_paragraphs):
    """Fallback duration estimation if timing metadata not available"""
    if not text or not all_paragraphs:
        return 3.0
    
    words_this_section = len(text.split())
    total_words = sum(len(p.split()) for p in all_paragraphs if p)
    
    if total_words == 0:
        return audio_duration / max(1, len(all_paragraphs))
    
    duration = (words_this_section / total_words) * audio_duration
    return max(2.0, duration)


def enhance_visual_prompt_for_mystery(prompt, scene_index, mystery_category):
    """Enhance prompts specifically for noir mystery aesthetic"""
    
    noir_base = "film noir photography, high contrast black and white, dramatic shadows, moody atmosphere, vintage 1940s-1960s aesthetic, film grain, mysterious, documentary style, cinematic lighting"
    
    # Scene-based enhancements (based on narrative position)
    scene_enhancements = {
        0: "opening establishing shot, dark and mysterious, ominous atmosphere, noir cinematography",
        1: "documentary evidence style, vintage photograph, historical authenticity, aged photo quality",
        2: "dramatic reveal, unsettling discovery, noir detective aesthetic, shadowy investigation",
        3: "final mystery image, unanswered questions, enigmatic conclusion, noir ending"
    }
    
    category_keywords = {
        'disappearance': "abandoned location, empty space, no trace, vanished without evidence, eerie absence",
        'crime': "crime scene aesthetic, investigation photo, evidence markers, detective noir, forensic",
        'historical': "ancient artifact, archaeological discovery, aged document, historical photograph",
        'conspiracy': "classified documents, redacted files, government secrets, declassified evidence"
    }
    
    enhancement = scene_enhancements.get(scene_index, "mysterious noir aesthetic")
    category_words = category_keywords.get(mystery_category, category_keywords['disappearance'])
    
    # Remove bright/cheerful words if present
    prompt = prompt.replace('happy', 'mysterious').replace('bright', 'dark').replace('colorful', 'monochrome')
    
    enhanced = f"{prompt}, {enhancement}, {category_words}, {noir_base}, foggy, unsettling, dark and ominous"
    enhanced = enhanced.replace('  ', ' ').strip()
    
    return enhanced


def generate_image_huggingface(prompt, filename, width=1080, height=1920):
    """Generate image using Hugging Face FLUX"""
    try:
        hf_token = os.getenv('HUGGINGFACE_API_KEY')
        if not hf_token:
            print("    ⚠️ HUGGINGFACE_API_KEY not found")
            raise Exception("Missing token")

        headers = {"Authorization": f"Bearer {hf_token}"}
        
        negative_mystery = (
            "blurry, low quality, watermark, text overlay, logo, frame, caption, "
            "ui elements, interface, play button, branding, typography, "
            "cartoon, anime, illustration, painting, drawing, sketch, 3d render, "
            "happy smiling cheerful cute, bright colorful, pastel colors, soft lighting, "
            "modern, contemporary, digital art, "
            "compression artifacts, pixelated, distorted, deformed, "
            "text watermark, amateur photo"
        )
        
        payload = {
            "inputs": f"{prompt}, film noir, black and white photography, high contrast, dramatic shadows, vintage, mysterious, documentary style",
            "parameters": {
                "negative_prompt": negative_mystery,
                "num_inference_steps": 4,
                "guidance_scale": 0.0,
                "width": width,
                "height": height,
            }
        }

        models = [
            "black-forest-labs/FLUX.1-schnell",
            "black-forest-labs/FLUX.1-dev",
            "stabilityai/stable-diffusion-xl-base-1.0"
        ]

        for model in models:
            url = f"https://api-inference.huggingface.co/models/{model}"
            print(f"🤗 Trying model: {model}")

            response = requests.post(url, headers=headers, json=payload, timeout=120)

            if response.status_code == 200 and len(response.content) > 1000:
                filepath = os.path.join(TMP, filename)
                with open(filepath, "wb") as f:
                    f.write(response.content)
                print(f"    ✅ HuggingFace succeeded: {model}")
                return filepath

            elif response.status_code == 402:
                print(f"💰 {model} requires payment — trying next...")
                continue

            elif response.status_code in [503, 429]:
                print(f"⌛ {model} loading/rate-limited — trying next...")
                time.sleep(2)
                continue

            else:
                print(f"⚠️ {model} failed ({response.status_code}) — trying next...")

        raise Exception("All HuggingFace models failed")

    except Exception as e:
        print(f"⚠️ HuggingFace failed: {e}")
        raise


def generate_image_pollinations(prompt, filename, width=1080, height=1920):
    """Pollinations backup with mystery-optimized prompts"""
    try:
        negative_terms = (
            "blurry, low quality, watermark, text, logo, cartoon, anime, "
            "illustration, happy, cheerful, bright, colorful, pastel, "
            "soft lighting, modern, contemporary"
        )

        formatted_prompt = (
            f"{prompt}, film noir photography, black and white, "
            "high contrast, dramatic shadows, vintage 1940s aesthetic, "
            "mysterious, documentary style, film grain"
        )

        seed = random.randint(1, 999999)

        url = (
            "https://image.pollinations.ai/prompt/"
            f"{requests.utils.quote(formatted_prompt)}"
            f"?width={width}&height={height}"
            f"&negative={requests.utils.quote(negative_terms)}"
            f"&nologo=true&notext=true&enhance=true&model=flux"
            f"&seed={seed}"
        )

        print(f"    🌐 Pollinations: {prompt[:60]}... (seed={seed})")
        response = requests.get(url, timeout=120)

        if response.status_code == 200 and "image" in response.headers.get("Content-Type", ""):
            filepath = os.path.join(TMP, filename)
            with open(filepath, "wb") as f:
                f.write(response.content)
            print(f"    ✅ Pollinations generated (seed {seed})")
            return filepath
        else:
            raise Exception(f"Pollinations failed: {response.status_code}")

    except Exception as e:
        print(f"    ⚠️ Pollinations failed: {e}")
        raise


def generate_mystery_fallback(bg_path, scene_index, mystery_category, width=1080, height=1920):
    """Mystery-specific fallback with Unsplash/Pexels"""
    
    # Mystery-themed keywords
    category_keywords = {
        'disappearance': ['abandoned', 'empty', 'fog', 'mystery', 'dark-sky', 'eerie'],
        'crime': ['noir', 'detective', 'shadow', 'urban-night', 'alley', 'investigation'],
        'historical': ['vintage', 'old-photo', 'ancient', 'archive', 'historical'],
        'conspiracy': ['documents', 'classified', 'secret', 'government', 'mystery']
    }
    
    keywords = category_keywords.get(mystery_category, ['mystery', 'noir', 'dark'])
    keyword = random.choice(keywords)
    
    print(f"🔎 Searching mystery image for '{mystery_category}' (keyword: '{keyword}')...")

    try:
        seed = random.randint(1, 9999)
        url = f"https://source.unsplash.com/{width}x{height}/?{requests.utils.quote(keyword)}&sig={seed}"
        print(f"🖼️ Unsplash: '{keyword}' (seed={seed})...")
        response = requests.get(url, timeout=30, allow_redirects=True)
        
        if response.status_code == 200 and "image" in response.headers.get("Content-Type", ""):
            with open(bg_path, "wb") as f:
                f.write(response.content)
            print(f"    ✅ Unsplash image saved")
            return bg_path
        else:
            print(f"    ⚠️ Unsplash failed ({response.status_code})")
    except Exception as e:
        print(f"    ⚠️ Unsplash error: {e}")

    # Noir/mystery Pexels photos
    try:
        print("    🔄 Trying Pexels curated mystery photos...")
        
        mystery_pexels_ids = {
            'disappearance': [3783471, 3617500, 3861431, 3617457, 2310713, 1363876, 1363875, 1366630],
            'crime': [2582937, 2559941, 2566581, 2566590, 2566573, 3617500, 1363876],
            'historical': [1418595, 1141853, 1303081, 2325729, 1907785, 2832382, 6069],
            'conspiracy': [3617500, 7319070, 7319311, 7319316, 5668826, 5668838]
        }
        
        scene_key = mystery_category if mystery_category in mystery_pexels_ids else 'disappearance'
        photo_ids = mystery_pexels_ids[scene_key].copy()
        random.shuffle(photo_ids)
        
        for attempt, photo_id in enumerate(photo_ids[:5]):
            seed = random.randint(1000, 9999)
            url = f"https://images.pexels.com/photos/{photo_id}/pexels-photo-{photo_id}.jpeg?auto=compress&cs=tinysrgb&w=1080&h=1920&fit=crop&random={seed}"
            
            print(f"📸 Pexels photo attempt {attempt+1} (id={photo_id})...")

            response = requests.get(url, timeout=30)
            if response.status_code == 200 and "image" in response.headers.get("Content-Type", ""):
                with open(bg_path, "wb") as f:
                    f.write(response.content)
                print(f"    ✅ Pexels photo saved (id: {photo_id})")

                img = Image.open(bg_path).convert("RGB")
                img = img.resize((width, height), Image.LANCZOS)
                img.save(bg_path, quality=95)
                
                return bg_path
            else:
                print(f"    ⚠️ Photo {photo_id} failed: {response.status_code}")
        
    except Exception as e:
        print(f"    ⚠️ Pexels fallback failed: {e}")

    return None


def create_noir_gradient(filepath, scene_index, width=1080, height=1920):
    """Create noir gradient background"""
    
    img = Image.new("RGB", (width, height), (0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    # Different gradients for different scenes
    if scene_index == 0:
        colors = [NOIR_COLORS['deep_black'], NOIR_COLORS['dark_slate']]
    elif scene_index == 1:
        colors = [NOIR_COLORS['shadow_blue'], NOIR_COLORS['fog_gray']]
    elif scene_index == 2:
        colors = [NOIR_COLORS['deep_black'], NOIR_COLORS['crimson_accent']]
    else:
        colors = [NOIR_COLORS['fog_gray'], NOIR_COLORS['deep_black']]
    
    for y in range(height):
        ratio = y / height
        r = int(colors[0][0] * (1 - ratio) + colors[1][0] * ratio)
        g = int(colors[0][1] * (1 - ratio) + colors[1][1] * ratio)
        b = int(colors[0][2] * (1 - ratio) + colors[1][2] * ratio)
        draw.line([(0, y), (width, y)], fill=(r, g, b))
    
    # Add heavy vignette for noir feel
    img = apply_vignette_noir(img, strength=0.6)
    
    img.save(filepath, quality=95)
    print(f"    ✅ Noir gradient created")
    return filepath


def apply_vignette_noir(img, strength=0.6):
    """Heavy noir vignette effect"""
    width, height = img.size
    mask = Image.new('L', (width, height), 255)
    draw = ImageDraw.Draw(mask)
    
    for i in range(int(min(width, height) * strength)):
        alpha = int(255 * (1 - i / (min(width, height) * strength)))
        draw.rectangle([i, i, width-i, height-i], outline=alpha)
    
    black = Image.new('RGB', (width, height), (0, 0, 0))
    return Image.composite(img, black, mask)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=4, max=20))
def generate_image_reliable(prompt, filename, scene_index, mystery_category, width=1080, height=1920):
    """Try multiple providers with mystery-specific fallbacks"""
    filepath = os.path.join(TMP, filename)
    
    providers = [
        ("Pollinations", generate_image_pollinations),
        ("HuggingFace", generate_image_huggingface)
    ]
    
    for provider_name, provider_func in providers:
        try:
            print(f"🎨 Trying {provider_name}...")
            result = provider_func(prompt, filename, width, height)
            if result and os.path.exists(result) and os.path.getsize(result) > 1000:
                return result
        except Exception as e:
            print(f"    ⚠️ {provider_name} failed: {e}")
            continue

    print("🖼️ AI failed, trying curated mystery photos...")
    result = generate_mystery_fallback(filepath, scene_index, mystery_category, width, height)

    if result and os.path.exists(filepath) and os.path.getsize(filepath) > 1000:
        return result
    
    print("⚠️ All providers failed, creating noir gradient...")
    return create_noir_gradient(filepath, scene_index, width, height)


def apply_noir_filter(image_path, scene_index):
    """
    Apply film noir aesthetic (REPLACES teal & orange)
    - Desaturate 70%
    - High contrast
    - Dark overall
    - Heavy vignette
    - Film grain
    - Slight blue tint
    """
    
    print(f"      🎨 Applying film noir filter...")
    
    try:
        img = Image.open(image_path).convert('RGB')
        
        # 1. Desaturate (70% desaturation)
        img = ImageEnhance.Color(img).enhance(0.30)  # 30% color left
        
        # 2. Increase contrast (noir is high contrast)
        img = ImageEnhance.Contrast(img).enhance(1.60)
        
        # 3. Darken overall (noir is darker)
        img = ImageEnhance.Brightness(img).enhance(0.70)
        
        # 4. Slight blue/teal tint (classic noir look)
        img = apply_blue_tint(img, intensity=0.15)
        
        # 5. Add film grain
        img = add_film_grain_noir(img, intensity=0.15)
        
        # 6. Heavy vignette
        img = apply_vignette_noir(img, strength=0.60)
        
        # 7. Slight sharpen for clarity
        img = img.filter(ImageFilter.SHARPEN)
        
        img.save(image_path, quality=95)
        print(f"      ✅ Noir filter applied")
        
    except Exception as e:
        print(f"      ⚠️ Noir filter failed: {e}")
    
    return image_path


def apply_blue_tint(img, intensity=0.15):
    """Add subtle blue tint for noir feel"""
    pixels = img.load()
    width, height = img.size
    
    for x in range(width):
        for y in range(height):
            r, g, b = pixels[x, y]
            
            # Add blue tint
            r = int(r + (10 - r) * intensity)
            g = int(g + (20 - g) * intensity)
            b = int(b + (40 - b) * intensity)
            
            pixels[x, y] = (max(0, min(255, r)), max(0, min(255, g)), max(0, min(255, b)))
    
    return img


def add_film_grain_noir(img, intensity=0.15):
    """Add film grain for vintage noir feel"""
    try:
        img_array = np.array(img)
        noise = np.random.normal(0, intensity * 255, img_array.shape)
        noisy = img_array + noise
        noisy = np.clip(noisy, 0, 255).astype(np.uint8)
        return Image.fromarray(noisy)
    except:
        return img


# 🎵 MUSIC INTEGRATION (Dark Ambient for Mystery) - FIXED

def ensure_music_downloaded():
    """Download essential mystery music tracks before use"""
    
    if not MUSIC_AVAILABLE:
        return False
    
    print("\n🎵 Checking mystery music library...")
    
    # Essential mystery tracks to download
    essential_tracks = [
        'dark_mystery',
        'suspense_build',
        'paranormal_ambient',
        'dramatic_reveal'
    ]
    
    downloaded = 0
    for track_key in essential_tracks:
        if track_key in MUSIC_LIBRARY:
            try:
                track_info = MUSIC_LIBRARY[track_key]
                result = download_track(track_key, track_info)
                if result:
                    downloaded += 1
                    print(f"   ✅ {track_info['name']} ready")
            except Exception as e:
                print(f"   ⚠️ Failed to download {track_key}: {e}")
    
    print(f"✅ {downloaded}/{len(essential_tracks)} mystery tracks available")
    return downloaded > 0


def create_dynamic_music_layer(audio_duration, script_data):
    """
    Create music layer with dark ambient/tension music
    Mystery uses subtle background music (lower volume than motivation)
    """
    
    if not MUSIC_AVAILABLE:
        print("⚠️ Music system unavailable, skipping background music")
        return None
    
    print("\n🎵 Creating mystery music layer...")
    
    # Ensure music is downloaded first
    if not ensure_music_downloaded():
        print("⚠️ No music tracks available, skipping")
        return None
    
    content_type = script_data.get('content_type', 'general')
    mystery_category = script_data.get('mystery_category', 'disappearance')
    
    # ✅ FIXED: Map content types to MYSTERY scene names
    scene_map = {
        'evening_prime': 'investigation',    # Dark detective vibe
        'late_night': 'suspense',            # Tense, unsettling
        'weekend_binge': 'paranormal',       # Eerie atmosphere
        'general': 'investigation'           # Default mystery
    }
    
    # Also map mystery categories to appropriate scenes
    category_scene_map = {
        'disappearance': 'suspense',
        'crime': 'investigation',
        'paranormal': 'paranormal',
        'historical': 'ancient',
        'conspiracy': 'conspiracy',
        'cryptid': 'cryptids',
        'unsolved': 'unsolved'
    }
    
    # Try category-based scene first, then content-type
    primary_scene = category_scene_map.get(mystery_category) or scene_map.get(content_type, 'investigation')
    
    print(f"   🎯 Scene type: {primary_scene}")
    
    try:
        track_key, music_path, _ = get_music_for_scene(primary_scene, content_type)
    except Exception as e:
        print(f"   ⚠️ Failed to get music for scene '{primary_scene}': {e}")
        print(f"   🔄 Trying fallback...")
        
        # Fallback to any available track
        for fallback_scene in ['investigation', 'suspense', 'paranormal', 'general']:
            try:
                track_key, music_path, _ = get_music_for_scene(fallback_scene, 'general')
                if music_path:
                    break
            except:
                continue
    
    if not music_path or not os.path.exists(music_path):
        print("⚠️ No music track available, skipping")
        return None
    
    print(f"   🎵 Track: {track_key}")
    print(f"   📁 Path: {music_path}")
    
    try:
        # Load base music
        music = AudioFileClip(music_path)
        
        # Loop if needed
        if music.duration < audio_duration:
            loops_needed = int(audio_duration / music.duration) + 1
            print(f"   🔁 Looping music {loops_needed}x")
            
            from moviepy.audio.AudioClip import concatenate_audioclips
            music_clips = [music] * loops_needed
            music = concatenate_audioclips(music_clips)
        
        # Trim to exact duration
        music = music.subclipped(0, min(audio_duration, music.duration))
        
        # MYSTERY USES LOWER VOLUME (more subtle than motivation)
        volume_levels = {
            'evening_prime': 0.20,   # Subtle (mystery is about voice)
            'late_night': 0.15,      # Very subtle (unsettling silence)
            'weekend_binge': 0.22,   # Slightly more (documentary)
            'general': 0.18          # Default subtle
        }
        
        base_volume = volume_levels.get(content_type, 0.18)
        
        # Use volumex for compatibility
        from moviepy.audio.fx.volumex import volumex
        music = music.fx(volumex, base_volume)
        
        print(f"   ✅ Mystery music layer created at {base_volume*100:.0f}% volume")
        print(f"   ⏱️ Duration: {music.duration:.2f}s")
        
        return music
            
    except Exception as e:
        print(f"⚠️ Music creation failed: {e}")
        import traceback
        traceback.print_exc()
        return None


# --- Main Scene Generation ---

print("🔍 Generating mystery scenes...")

# ✅ MYSTERY VERSION: Generate images for paragraphs (not bullets)
# Use first 4 paragraphs for visuals (hook, setup, incident, twist typically)
scene_images = []

try:
    # Generate images for first 4 narrative sections
    num_scenes = min(4, len(paragraphs))
    
    for i in range(num_scenes):
        paragraph = paragraphs[i] if i < len(paragraphs) else ""
        visual_prompt = visual_prompts[i] if i < len(visual_prompts) else f"Noir mystery scene: {paragraph[:100]}"
        
        # Enhance for noir aesthetic
        visual_prompt = enhance_visual_prompt_for_mystery(visual_prompt, i, mystery_category)
        
        print(f"🎬 Generating scene {i+1}/{num_scenes}...")
        scene_img = generate_image_reliable(visual_prompt, f"scene_{i}.jpg", i, mystery_category, w, h)
        
        if scene_img:
            apply_noir_filter(scene_img, i)
        
        scene_images.append(scene_img)
        time.sleep(0.5)
    
    successful = len([img for img in scene_images if img and os.path.exists(img)])
    print(f"✅ Generated {successful}/{num_scenes} mystery scenes")
    
except Exception as e:
    print(f"⚠️ Image generation error: {e}")
    scene_images = [None] * 4

# Validate images
print(f"🔍 Validating {len(scene_images)} scenes...")
for i in range(len(scene_images)):
    img = scene_images[i] if i < len(scene_images) else None
    
    if not img or not os.path.exists(img) or os.path.getsize(img) < 1000:
        print(f"⚠️ Scene {i} invalid, creating noir gradient...")
        fallback_path = os.path.join(TMP, f"scene_fallback_{i}.jpg")
        create_noir_gradient(fallback_path, i, w, h)
        scene_images[i] = fallback_path

print(f"✅ All mystery scenes validated")

# --- Audio Loading & Timing (OPTIMIZED FOR PARAGRAPHS) ---

if not os.path.exists(audio_path):
    print(f"❌ Audio not found: {audio_path}")
    raise FileNotFoundError("voice.mp3 missing")

audio = AudioFileClip(audio_path)
duration = audio.duration
print(f"🎵 Audio: {duration:.2f}s")

# ✅ LOAD TIMING METADATA (works with paragraph-based timing)
timing_data = load_audio_timing()

# Calculate durations for each paragraph
paragraph_durations = []

if timing_data and timing_data.get('optimized'):
    print("\n⏱️ Using OPTIMIZED audio timing")
    
    sections = timing_data['sections']
    
    for i, paragraph in enumerate(paragraphs):
        para_section = next((s for s in sections if s['name'] == f'paragraph_{i+1}'), None)
        
        if para_section:
            dur = para_section['duration']
            paragraph_durations.append(dur)
            print(f"   Paragraph {i+1}: {dur:.2f}s (from metadata)")
        else:
            # Estimate
            dur = estimate_duration_fallback(paragraph, duration, paragraphs)
            paragraph_durations.append(dur)
            print(f"   Paragraph {i+1}: {dur:.2f}s (estimated)")
    
    # Adjust to match total duration
    total_calculated = sum(paragraph_durations)
    
    if abs(total_calculated - duration) > 0.5:
        adjustment_factor = duration / total_calculated
        paragraph_durations = [d * adjustment_factor for d in paragraph_durations]
        print(f"   ✅ Adjusted by factor {adjustment_factor:.4f}")

else:
    print("\n⚠️ No timing metadata, using word-based estimation")
    
    total_words = sum(len(p.split()) for p in paragraphs)
    
    for paragraph in paragraphs:
        words = len(paragraph.split())
        if total_words > 0:
            dur = (words / total_words) * duration
        else:
            dur = duration / max(1, len(paragraphs))
        
        paragraph_durations.append(max(2.0, dur))

print(f"\n⏱️ Final Paragraph Timings:")
for i, dur in enumerate(paragraph_durations):
    print(f"   Paragraph {i+1}: {dur:.2f}s ({len(paragraphs[i].split())} words)")

total_timeline = sum(paragraph_durations)
print(f"   Total Timeline: {total_timeline:.2f}s")
print(f"   Audio Duration: {duration:.2f}s")
print(f"   Drift: {abs(total_timeline - duration)*1000:.0f}ms")

# --- Video Composition (MYSTERY VERSION) ---

clips = []
current_time = 0


def smart_text_wrap(text, font_size, max_width):
    """Smart text wrapping (mystery uses less text overall)"""
    try:
        pil_font = ImageFont.truetype(FONT, font_size)
        dummy_img = Image.new('RGB', (1, 1))
        draw = ImageDraw.Draw(dummy_img)
        
        words = text.split()
        lines = []
        current_line = []
        
        for word in words:
            test_line = ' '.join(current_line + [word])
            bbox = draw.textbbox((0, 0), test_line, font=pil_font)
            text_width = bbox[2] - bbox[0]
            
            if text_width <= max_width:
                current_line.append(word)
            else:
                if current_line:
                    lines.append(' '.join(current_line))
                current_line = [word]
        
        if current_line:
            lines.append(' '.join(current_line))
        
        return '\n'.join(lines) + '\n'
        
    except:
        words = text.split()
        avg_char_width = font_size * 0.5
        max_chars = int(max_width / avg_char_width)
        
        lines = []
        current_line = []
        
        for word in words:
            test = ' '.join(current_line + [word])
            if len(test) <= max_chars:
                current_line.append(word)
            else:
                if current_line:
                    lines.append(' '.join(current_line))
                current_line = [word]
        
        if current_line:
            lines.append(' '.join(current_line))
        
        return '\n'.join(lines) + '\n'


def create_text_with_effects(text, font_size=60, max_width=TEXT_MAX_WIDTH):
    """Create text with mystery styling (smaller, serif font)"""
    wrapped = smart_text_wrap(text, font_size, max_width)
    
    try:
        pil_font = ImageFont.truetype(FONT, font_size)
        dummy = Image.new('RGB', (1, 1))
        draw = ImageDraw.Draw(dummy)
        
        lines = wrapped.split('\n')
        total_h = 0
        max_w = 0
        
        for line in lines:
            bbox = draw.textbbox((0, 0), line, font=pil_font)
            total_h += bbox[3] - bbox[1]
            max_w = max(max_w, bbox[2] - bbox[0])
        
        max_height = h * 0.25  # Less text than motivation
        iterations = 0
        
        while (total_h > max_height or max_w > max_width) and font_size > 32 and iterations < 10:
            font_size -= 4
            wrapped = smart_text_wrap(text, font_size, max_width)
            pil_font = ImageFont.truetype(FONT, font_size)
            
            lines = wrapped.split('\n')
            total_h = 0
            max_w = 0
            
            for line in lines:
                bbox = draw.textbbox((0, 0), line, font=pil_font)
                total_h += bbox[3] - bbox[1]
                max_w = max(max_w, bbox[2] - bbox[0])
            
            iterations += 1
            
    except:
        pass
    
    return wrapped, font_size


def create_scene(image_path, text, duration, start_time, show_text=True, color_fallback=None):
    """
    Create mystery scene with image + minimal text
    Mystery shows LESS text than motivation (let visuals tell story)
    """
    scene_clips = []
    
    if color_fallback is None:
        color_fallback = NOIR_COLORS['deep_black']
    
    if image_path and os.path.exists(image_path):
        bg = (ImageClip(image_path)
              .resized(height=h)
              .with_duration(duration)
              .with_start(start_time)
              .with_effects([vfx.CrossFadeIn(0.5), vfx.CrossFadeOut(0.5)]))  # Slower fades for mystery
    else:
        bg = (ColorClip(size=(w, h), color=color_fallback, duration=duration)
              .with_start(start_time))
    
    scene_clips.append(bg)
    
    # Mystery: Show MINIMAL text (only hook or key phrases)
    # Most scenes don't show text - let narration carry the story
    if text and show_text:
        # Extract key phrase (first sentence or ~10 words max)
        first_sentence = text.split('.')[0] if '.' in text else text
        key_words = ' '.join(first_sentence.split()[:10])
        
        if len(key_words) > 50:
            key_words = ' '.join(first_sentence.split()[:7])
        
        wrapped, font_size = create_text_with_effects(key_words, font_size=50)
        
        text_clip = TextClip(
            text=wrapped,
            font=FONT,
            font_size=font_size,
            color='white',
            stroke_width=4,
            stroke_color='black',
            method='caption',
            text_align='center',
            size=(TEXT_MAX_WIDTH, None),
        )
        
        text_h = text_clip.h
        # Position at bottom for mystery (documentary subtitle style)
        pos_y = h - text_h - SAFE_ZONE_MARGIN - 150
        
        text_clip = (text_clip
                    .with_duration(duration)
                    .with_start(start_time)
                    .with_position(('center', pos_y))
                    .with_effects([vfx.CrossFadeIn(0.5), vfx.CrossFadeOut(0.5)]))
        
        print(f"      Text: '{key_words[:30]}...' @ Y={pos_y}, Size={font_size}px")
        scene_clips.append(text_clip)
    
    return scene_clips


# Build mystery scenes
# Mystery: Show text ONLY on first scene (hook), rest is pure visuals + narration
for i, paragraph in enumerate(paragraphs):
    dur = paragraph_durations[i]
    
    # Get image (cycle through available scenes)
    img_idx = min(i, len(scene_images) - 1)
    
    # Only show text on first scene (hook)
    show_text = (i == 0)
    
    print(f"🎬 Paragraph {i+1}/{len(paragraphs)} (text: {show_text})...")
    
    color_fallback = NOIR_COLORS['deep_black'] if i % 2 == 0 else NOIR_COLORS['dark_slate']
    
    clips.extend(create_scene(
        scene_images[img_idx], 
        paragraph if show_text else "", 
        dur, 
        current_time,
        show_text=show_text,
        color_fallback=color_fallback
    ))
    
    current_time += dur

# Sync check
print(f"\n📊 SYNC CHECK:")
print(f"   Timeline: {current_time:.2f}s")
print(f"   Audio: {duration:.2f}s")
print(f"   Drift: {abs(current_time - duration)*1000:.0f}ms")

if abs(current_time - duration) < 0.05:
    print(f"   ✅ NEAR-PERFECT SYNC!")
elif abs(current_time - duration) < 0.5:
    print(f"   ✅ Excellent sync")
else:
    print(f"   ⚠️ Sync drift detected")

print(f"\n🎬 Composing mystery video ({len(clips)} clips)...")
video = CompositeVideoClip(clips, size=(w, h))

# 🎵 ADD BACKGROUND MUSIC + TTS VOICEOVER
print(f"\n🔊 Adding audio with dark ambient music...")

background_music = create_dynamic_music_layer(duration, data)

if background_music:
    try:
        from moviepy.audio.fx.audio_normalize import audio_normalize
        
        # Normalize voice
        voice_normalized = audio.fx(audio_normalize)
        
        # Composite: TTS voiceover + dark ambient music
        final_audio = CompositeAudioClip([voice_normalized, background_music])
        video = video.with_audio(final_audio)
        print(f"   ✅ Audio: TTS + Dark ambient music")
    except Exception as e:
        print(f"   ⚠️ Music compositing failed: {e}")
        import traceback
        traceback.print_exc()
        video = video.with_audio(audio)
else:
    video = video.with_audio(audio)
    print(f"   ⚠️ Audio: TTS only (no background music)")

if video.audio is None:
    raise Exception("Audio attachment failed!")

print(f"\n📹 Writing mystery video...")
try:
    video.write_videofile(
        OUT,
        fps=30,
        codec="libx264",
        audio_codec="aac",
        threads=4,
        preset='medium',
        audio_bitrate='192k',
        bitrate='8000k',
        logger=None
    )
    
    sync_status = "NEAR-PERFECT" if abs(current_time - duration) < 0.05 else "EXCELLENT" if abs(current_time - duration) < 0.5 else "GOOD"
    
    print(f"\n✅ MYSTERY VIDEO COMPLETE!")
    print(f"   Path: {OUT}")
    print(f"   Duration: {duration:.2f}s")
    print(f"   Size: {os.path.getsize(OUT) / (1024*1024):.2f} MB")
    print(f"   Sync Status: {sync_status} ({abs(current_time - duration)*1000:.0f}ms drift)")
    print(f"   Features:")
    print(f"      ✓ Film noir aesthetic (desaturated, high contrast)")
    print(f"      ✓ Heavy vignette + film grain")
    if background_music:
        print(f"      ✓ Dark ambient background music")
        print(f"      ✓ Professional audio mix (TTS + Music)")
    else:
        print(f"      ⚠ Background music skipped (unavailable)")
    print(f"      ✓ Paragraph-based narrative timing")
    print(f"      ✓ Minimal text overlays (documentary style)")
    print(f"      ✓ Serif/typewriter fonts")
    print(f"      ✓ Mystery-optimized image generation")
    print(f"      ✓ Noir gradient fallbacks")
    print(f"      ✓ Slower crossfade transitions (0.5s)")
    print(f"      ✓ Audio-synchronized timing")
    print(f"   🔍 Mystery Archives ready!")
    
except Exception as e:
    print(f"❌ Video creation failed: {e}")
    import traceback
    traceback.print_exc()
    raise

finally:
    print("🧹 Cleanup...")
    audio.close()
    video.close()
    if background_music:
        background_music.close()
    for clip in clips:
        try:
            clip.close()
        except:
            pass

print("✅ Mystery video pipeline complete! 🔍")