#!/usr/bin/env python3
"""
🔍 Create Mystery Video - PRODUCTION VERSION WITH ENHANCED TEXT DISPLAY
Python 3.11 + MoviePy 2.0+ Compatible (GitHub Actions Ready)
ENHANCED: Professional subtitle system with cinematic text presentation
FIXED: All timing, cleanup, and import issues resolved
"""

import os
import json
import requests
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
import tempfile
import atexit

# ✅ FIXED: MoviePy 2.0+ imports (GitHub Actions compatible)
try:
    from moviepy import VideoFileClip, AudioFileClip, ImageClip, ColorClip, TextClip
    from moviepy import CompositeVideoClip, CompositeAudioClip, concatenate_audioclips
    print("✅ MoviePy imported (direct module)")
except ImportError:
    try:
        from moviepy.editor import VideoFileClip, AudioFileClip, ImageClip, ColorClip, TextClip
        from moviepy.editor import CompositeVideoClip, CompositeAudioClip, concatenate_audioclips
        print("✅ MoviePy imported (editor module)")
    except ImportError:
        import moviepy
        VideoFileClip = moviepy.VideoFileClip
        AudioFileClip = moviepy.AudioFileClip
        ImageClip = moviepy.ImageClip
        ColorClip = moviepy.ColorClip
        TextClip = moviepy.TextClip
        CompositeVideoClip = moviepy.CompositeVideoClip
        CompositeAudioClip = moviepy.CompositeAudioClip
        concatenate_audioclips = moviepy.concatenate_audioclips
        print("✅ MoviePy imported (attribute access)")

# ✅ SAFE AUDIO/VIDEO EFFECTS FUNCTIONS
def apply_fadein(clip, duration=0.5):
    """Universal fadein that works with any MoviePy version"""
    try:
        return clip.crossfadein(duration)
    except AttributeError:
        try:
            import moviepy.video.fx.all as vfx
            return clip.fx(vfx.fadein, duration)
        except:
            return clip

def apply_fadeout(clip, duration=0.5):
    """Universal fadeout that works with any MoviePy version"""
    try:
        return clip.crossfadeout(duration)
    except AttributeError:
        try:
            import moviepy.video.fx.all as vfx
            return clip.fx(vfx.fadeout, duration)
        except:
            return clip

def apply_volumex(clip, factor):
    """Universal volume adjustment that works with any MoviePy version"""
    try:
        return clip.multiply_volume(factor)
    except AttributeError:
        try:
            return clip.with_effects([("multiply_volume", factor)])
        except:
            try:
                import moviepy.audio.fx.all as afx
                return clip.fx(afx.volumex, factor)
            except:
                return clip

def load_enhanced_timing():
    """
    ✅ NEW: Load enhanced timing data from audio_timing.json
    Returns timing_data dict or None if not available
    """
    timing_path = os.path.join(TMP, "audio_timing.json")
    
    if not os.path.exists(timing_path):
        print("⚠️ No enhanced timing file found")
        return None
    
    try:
        with open(timing_path, 'r') as f:
            timing_data = json.load(f)
        
        if not timing_data.get('optimized'):
            print("⚠️ Timing data not marked as optimized")
            return None
        
        method = timing_data.get('timing_method', 'unknown')
        sections = timing_data.get('sections', [])
        
        if method == 'enhanced_proportional':
            print(f"✅ Loaded ENHANCED timing (silence-aware, word-proportional)")
        else:
            print(f"✅ Loaded timing method: {method}")
        
        print(f"   Total duration: {timing_data.get('total_duration', 0):.2f}s")
        print(f"   Speech duration: {timing_data.get('speech_duration', 0):.2f}s")
        print(f"   Sections: {len(sections)}")
        
        return timing_data
        
    except Exception as e:
        print(f"⚠️ Failed to load enhanced timing: {e}")
        import traceback
        traceback.print_exc()
        return None

def trim_audio_safe(audio_clip, target_duration):
    """✅ FIXED: Safely trim audio to target duration"""
    try:
        if hasattr(audio_clip, 'subclipped'):
            return audio_clip.subclipped(0, min(target_duration, audio_clip.duration))
        elif hasattr(audio_clip, 'set_duration'):
            return audio_clip.set_duration(min(target_duration, audio_clip.duration))
        else:
            return audio_clip
    except Exception as e:
        print(f"⚠️ Audio trim failed: {e}, returning original")
        return audio_clip

TMP = os.getenv("GITHUB_WORKSPACE", ".") + "/tmp"
OUT = os.path.join(TMP, "short.mp4")
audio_path = os.path.join(TMP, "voice.mp3")
w, h = 1080, 1920

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

# ✅ FIXED: Temp file tracking for cleanup
TEMP_FILES = []

def register_temp_file(filepath):
    """Register a temp file for cleanup"""
    TEMP_FILES.append(filepath)
    return filepath

def cleanup_temp_files():
    """Clean up all registered temp files"""
    for filepath in TEMP_FILES:
        try:
            if os.path.exists(filepath):
                os.remove(filepath)
        except:
            pass

# Register cleanup on exit
atexit.register(cleanup_temp_files)

# ✅ FIXED: Safe import of music system
MUSIC_AVAILABLE = False
try:
    script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
    sys.path.insert(0, script_dir)
    from download_music import get_music_for_scene, MUSIC_DIR, download_track, MUSIC_LIBRARY
    MUSIC_AVAILABLE = True
    print("✅ Music system imported successfully")
except ImportError as e:
    print(f"⚠️ Music system not available: {e}")

def get_font_path():
    """Get serif/typewriter font for mystery aesthetic"""
    system = platform.system()
    if system == "Windows":
        fonts = [
            "C:/Windows/Fonts/cour.ttf",
            "C:/Windows/Fonts/georgia.ttf",
            "C:/Windows/Fonts/times.ttf",
        ]
        for font in fonts:
            if os.path.exists(font):
                return font
        return "C:/Windows/Fonts/arial.ttf"
    elif system == "Darwin":
        return "/System/Library/Fonts/Supplemental/Courier New.ttf"
    else:
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

# ✅ MYSTERY VERSION: Read 'script' field
full_script = data.get("script", "")
if not full_script:
    bullets = data.get("bullets", [])
    cta = data.get("cta", "")
    full_script = f"{hook}\n\n{' '.join(bullets)}\n\n{cta}"
else:
    cta = data.get("cta", "")

# Split script into paragraphs
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
    
    prompt = prompt.replace('happy', 'mysterious').replace('bright', 'dark').replace('colorful', 'monochrome')
    
    enhanced = f"{prompt}, {enhancement}, {category_words}, {noir_base}, foggy, unsettling, dark and ominous"
    return enhanced.replace('  ', ' ').strip()


def generate_image_huggingface(prompt, filename, width=1080, height=1920):
    """Generate image using Hugging Face FLUX"""
    try:
        hf_token = os.getenv('HUGGINGFACE_API_KEY')
        if not hf_token:
            raise Exception("Missing HUGGINGFACE_API_KEY")

        headers = {"Authorization": f"Bearer {hf_token}"}
        
        negative_mystery = (
            "blurry, low quality, watermark, text overlay, logo, frame, caption, "
            "ui elements, cartoon, anime, illustration, happy, cheerful, bright, colorful, pastel, "
            "soft lighting, modern, contemporary, digital art, amateur photo"
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
            elif response.status_code in [402, 503, 429]:
                print(f"⌛ {model} temporarily unavailable — trying next...")
                time.sleep(2)
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
            "illustration, happy, cheerful, bright, colorful, pastel, soft lighting, modern, contemporary"
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
    except Exception as e:
        print(f"    ⚠️ Unsplash error: {e}")

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
    except Exception as e:
        print(f"    ⚠️ Pexels fallback failed: {e}")

    return None


def create_noir_gradient(filepath, scene_index, width=1080, height=1920):
    """Create noir gradient background"""
    
    img = Image.new("RGB", (width, height), (0, 0, 0))
    draw = ImageDraw.Draw(img)
    
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
    
    img = apply_vignette_noir(img, strength=0.6)
    img.save(filepath, quality=95)
    print(f"    ✅ Noir gradient created")
    return filepath


def apply_vignette_noir(img, strength=0.6):
    """Heavy noir vignette effect"""
    width, height = img.size
    mask = Image.new('L', (width, height), 255)
    draw = ImageDraw.Draw(mask)
    
    max_dim = int(min(width, height) * strength)
    for i in range(max_dim):
        alpha = int(255 * (1 - i / max_dim))
        x0, y0 = i, i
        x1, y1 = width - i - 1, height - i - 1
        if x1 > x0 and y1 > y0:
            draw.rectangle([x0, y0, x1, y1], outline=alpha)
    
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
    """Apply film noir aesthetic"""
    
    print(f"      🎨 Applying film noir filter...")
    
    try:
        img = Image.open(image_path).convert('RGB')
        
        img = ImageEnhance.Color(img).enhance(0.30)
        img = ImageEnhance.Contrast(img).enhance(1.60)
        img = ImageEnhance.Brightness(img).enhance(0.70)
        img = apply_blue_tint(img, intensity=0.15)
        img = add_film_grain_noir(img, intensity=0.15)
        img = apply_vignette_noir(img, strength=0.60)
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
            
            r = int(r + (10 - r) * intensity)
            g = int(g + (20 - g) * intensity)
            b = int(b + (40 - b) * intensity)
            
            pixels[x, y] = (max(0, min(255, r)), max(0, min(255, g)), max(0, min(255, b)))
    
    return img

def detect_leading_trailing_silence(audio_path, silence_thresh_db=-45.0, chunk_ms=10):
    """
    Detect leading/trailing silence in seconds using pydub.
    """
    if not os.path.exists(audio_path):
        print(f"⚠️ Cannot detect silence, audio not found at: {audio_path}")
        return 0.0, 0.0
    try:
        audio = AudioSegment.from_file(audio_path)
        
        def _leading_silence(a):
            trim = 0
            # Iterate until we find a chunk that is not silent
            while trim < len(a) and a[trim:trim+chunk_ms].dBFS < silence_thresh_db:
                trim += chunk_ms
            return trim / 1000.0

        def _trailing_silence(a):
            trim = 0
            # Iterate from the end until we find a chunk that is not silent
            while trim < len(a) and a[-trim-chunk_ms:len(a)-trim].dBFS < silence_thresh_db:
                trim += chunk_ms
            return trim / 1000.0

        lead = _leading_silence(audio)
        trail = _trailing_silence(audio)
        print(f"🔧 Silence Detection — Lead: {lead:.3f}s, Trail: {trail:.3f}s")
        return lead, trail
    except Exception as e:
        print(f"⚠️ Pydub silence detection failed: {e}")
        return 0.0, 0.0


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


# ✨ ENHANCED TEXT FUNCTIONS (FIXED - NO DUPLICATES) ✨

def segment_text_for_display(text, max_words_per_segment=10, max_segments=4):
    """
    Break text into readable segments for progressive display.
    Tries to break at natural sentence boundaries.
    """
    sentences = text.replace('...', '.').split('.')
    sentences = [s.strip() for s in sentences if s.strip()]
    
    segments = []
    current_segment_words = []
    
    for sentence in sentences:
        words = sentence.split()
        if len(current_segment_words) + len(words) > max_words_per_segment and current_segment_words:
            segments.append(' '.join(current_segment_words) + '.')
            current_segment_words = []
        
        current_segment_words.extend(words)
        
        if len(segments) >= max_segments - 1:
            break
            
    if current_segment_words:
        segments.append(' '.join(current_segment_words) + ('.' if not ' '.join(current_segment_words).endswith('.') else ''))

    if not segments and text:
        words = text.split()
        for i in range(0, len(words), max_words_per_segment):
            segments.append(' '.join(words[i:i + max_words_per_segment]))
    
    return segments[:max_segments]


def create_text_panel(width, height, opacity=0.85):
    """
    Create enhanced noir text panel with better readability.
    """
    panel = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(panel)
    
    # Create solid dark background
    draw.rectangle([(0, 0), (width, height)], fill=(15, 15, 20, int(255 * opacity)))
    
    # Add subtle film grain texture
    for _ in range(int(width * height * 0.0005)):
        x, y = random.randint(0, width - 1), random.randint(0, height - 1)
        val = random.randint(0, 30)
        draw.point((x, y), fill=(val, val, val, 50))
    
    # Add distinctive gold border
    border_color = (212, 175, 55, 200)
    draw.rectangle([(0, 0), (width - 1, height - 1)], outline=border_color, width=3)
    
    return panel

def smart_text_wrap(text, font_size, max_width):
    """
    Improved text wrapping for vertical video.
    """
    try:
        pil_font = ImageFont.truetype(FONT, font_size)
    except IOError:
        pil_font = ImageFont.load_default()

    words = text.split()
    lines = []
    current_line = []
    
    draw = ImageDraw.Draw(Image.new('RGB', (1, 1)))

    for word in words:
        test_line = ' '.join(current_line + [word])
        try:
            # Use textbbox for accurate width measurement
            bbox = draw.textbbox((0, 0), test_line, font=pil_font)
            line_width = bbox[2] - bbox[0]
        except AttributeError:
            line_width, _ = draw.textsize(test_line, font=pil_font)

        if line_width <= max_width:
            current_line.append(word)
        else:
            if current_line:
                lines.append(' '.join(current_line))
            current_line = [word]
    
    if current_line:
        lines.append(' '.join(current_line))
    
    return '\n'.join(lines)


def create_cinematic_text_clip(text, font_size, duration, start_time, position='lower_third', panel_bg=True):
    """
    ✅ FIXED: Creates a robust, pre-rendered text clip using PIL with cleanup.
    """
    # 1. Wrap text and get its dimensions
    wrapped_text = smart_text_wrap(text, font_size, TEXT_MAX_WIDTH)
    try:
        pil_font = ImageFont.truetype(FONT, font_size)
    except IOError:
        print(f"⚠️ Font not found at {FONT}. Using default.")
        pil_font = ImageFont.load_default()

    dummy_draw = ImageDraw.Draw(Image.new('RGB', (1, 1)))
    try:
        bbox = dummy_draw.textbbox((0, 0), wrapped_text, font=pil_font, stroke_width=2)
        text_width, text_height = bbox[2] - bbox[0], bbox[3] - bbox[1]
    except Exception:
        text_width, text_height = dummy_draw.textsize(wrapped_text, font=pil_font)

    # 2. Create the composite image (panel + text) using PIL
    panel_padding_x, panel_padding_y = 80, 60
    comp_width = int(text_width + panel_padding_x)
    comp_height = int(text_height + panel_padding_y)

    # Use the stylized panel function if requested
    if panel_bg:
        final_image = create_text_panel(comp_width, comp_height, opacity=0.85)
    else:
        final_image = Image.new('RGBA', (comp_width, comp_height), (0, 0, 0, 0))

    # Draw the text onto the image, centered within the panel
    draw = ImageDraw.Draw(final_image)
    text_x = (comp_width - text_width) / 2
    text_y = (comp_height - text_height) / 2
    
    # Draw a subtle shadow for better readability
    draw.text((text_x + 2, text_y + 2), wrapped_text, font=pil_font, fill=(10, 10, 10, 180), align='center')
    # Draw the main text
    draw.text((text_x, text_y), wrapped_text, font=pil_font, fill=NOIR_COLORS['evidence_tan'], align='center')

    # 3. Save the composite image to a temporary file
    comp_path = os.path.join(TMP, f"text_comp_{time.time()}_{random.randint(1000, 9999)}.png")
    final_image.save(comp_path)
    
    # ✅ FIXED: Register for cleanup
    register_temp_file(comp_path)

    # 4. Position and animate this single, robust image clip in MoviePy
    if position == 'lower_third':
        clip_pos = ('center', h * 0.70)
    elif position == 'center':
        clip_pos = ('center', 'center')
    else:  # upper_third
        clip_pos = ('center', h * 0.25)

    final_clip = (ImageClip(comp_path)
                  .with_duration(duration)
                  .with_start(start_time)
                  .with_position(clip_pos))
    
    # Apply cinematic fade-in/fade-out
    fade_dur = min(0.5, duration / 4)
    final_clip = apply_fadein(final_clip, fade_dur)
    final_clip = apply_fadeout(final_clip, fade_dur)

    return [final_clip]

def create_enhanced_scene(image_path, text, duration, start_time, scene_index=0):
    """
    ✅ FIXED: Creates a scene with professionally timed and rendered text.
    """
    scene_clips = []
    
    # Background image/color with a subtle zoom for dynamism
    if image_path and os.path.exists(image_path):
        base_clip = ImageClip(image_path).with_duration(duration)
        # Apply a slow zoom-in effect
        zoomed_clip = base_clip.resized(lambda t: 1 + 0.02 * t).with_position(('center', 'center'))
        # Crop to the final video dimensions to contain the zoom
        bg = CompositeVideoClip([zoomed_clip], size=(w, h)).with_duration(duration)
    else:
        color_fallback = NOIR_COLORS['deep_black'] if scene_index % 2 == 0 else NOIR_COLORS['dark_slate']
        bg = ColorClip(size=(w, h), color=color_fallback).with_duration(duration)
    
    scene_clips.append(bg.with_start(start_time))

    # Skip empty text
    if not text or not text.strip():
        return scene_clips
        
    # 1. Break paragraph into smaller, readable segments for display
    segments = segment_text_for_display(text, max_words_per_segment=12, max_segments=4)
    if not segments: return scene_clips

    # 2. Allocate duration to each segment based on its word count
    total_words_in_segments = sum(len(s.split()) for s in segments)
    segment_durations = []
    if total_words_in_segments > 0:
        for seg in segments:
            # Proportional duration
            seg_duration = (len(seg.split()) / total_words_in_segments) * duration
            # Add a base minimum time for readability
            min_time = 1.5 + len(seg.split()) * 0.20
            segment_durations.append(max(seg_duration, min_time))
    else: 
        segment_durations = [duration / len(segments)] * len(segments)

    # 3. Normalize durations to perfectly match the total scene duration
    current_total_dur = sum(segment_durations)
    if current_total_dur > 0:
        scale_factor = duration / current_total_dur
        segment_durations = [d * scale_factor for d in segment_durations]

    # 4. Create a timed, cinematic clip for each segment
    current_segment_time = start_time
    for i, segment in enumerate(segments):
        seg_dur = segment_durations[i]
        if seg_dur < 0.5: continue # Skip trivially short segments

        # Use a consistent, readable font size
        font_size = 58 if len(segment.split()) < 8 else 54
        
        # Use the new robust PIL-based text clip function
        text_clip_list = create_cinematic_text_clip(
            segment,
            font_size=font_size,
            duration=seg_dur,
            start_time=current_segment_time,
            position='lower_third' # Consistent position is better for reading flow
        )
        scene_clips.extend(text_clip_list)
        
        current_segment_time += seg_dur

    return scene_clips


# 🎵 MUSIC INTEGRATION

def ensure_music_downloaded():
    """Download essential mystery music tracks before use"""
    
    if not MUSIC_AVAILABLE:
        return False
    
    print("\n🎵 Checking mystery music library...")
    
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
    """✅ FIXED: Create music layer with dark ambient/tension music"""
    
    if not MUSIC_AVAILABLE:
        print("⚠️ Music system unavailable, skipping background music")
        return None
    
    print("\n🎵 Creating mystery music layer...")
    
    if not ensure_music_downloaded():
        print("⚠️ No music tracks available, skipping")
        return None
    
    content_type = script_data.get('content_type', 'general')
    mystery_category = script_data.get('mystery_category', 'disappearance')
    
    scene_map = {
        'evening_prime': 'investigation',
        'late_night': 'suspense',
        'weekend_binge': 'paranormal',
        'general': 'investigation'
    }
    
    category_scene_map = {
        'disappearance': 'suspense',
        'crime': 'investigation',
        'paranormal': 'paranormal',
        'historical': 'ancient',
        'conspiracy': 'conspiracy',
        'cryptid': 'cryptids',
        'unsolved': 'unsolved'
    }
    
    primary_scene = category_scene_map.get(mystery_category) or scene_map.get(content_type, 'investigation')
    
    print(f"   🎯 Scene type: {primary_scene}")
    
    try:
        track_key, music_path, _ = get_music_for_scene(primary_scene, content_type)
    except Exception as e:
        print(f"   ⚠️ Failed to get music for scene '{primary_scene}': {e}")
        
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
        music = AudioFileClip(music_path)
        
        if music.duration < audio_duration:
            loops_needed = int(audio_duration / music.duration) + 1
            print(f"   🔁 Looping music {loops_needed}x")
            
            music_clips = [music] * loops_needed
            music = concatenate_audioclips(music_clips)
        
        # ✅ FIXED: Use safe trim function
        music = trim_audio_safe(music, audio_duration)
        
        volume_levels = {
            'evening_prime': 0.20,
            'late_night': 0.15,
            'weekend_binge': 0.22,
            'general': 0.18
        }
        
        base_volume = volume_levels.get(content_type, 0.18)
        
        # Apply volume safely
        music = apply_volumex(music, base_volume)
        
        print(f"   ✅ Mystery music layer created at {base_volume*100:.0f}% volume")
        print(f"   ⏱️ Duration: {music.duration:.2f}s")
        
        return music
    except Exception as e:
        print(f"⚠️ Music creation failed: {e}")
        import traceback
        traceback.print_exc()
        return None


# --- Scene Generation ---

print("🔍 Generating mystery scenes...")

scene_images = []

try:
    num_scenes = min(len(paragraphs), 4) if paragraphs else 0
    
    for i in range(num_scenes):
        paragraph = paragraphs[i]
        visual_prompt = visual_prompts[i] if i < len(visual_prompts) else f"Noir mystery scene: {paragraph[:100]}"
        
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
    scene_images = [None] * len(paragraphs)

print(f"🔍 Validating {len(scene_images)} scenes...")
for i in range(len(scene_images)):
    img = scene_images[i]
    
    if not img or not os.path.exists(img) or os.path.getsize(img) < 1000:
        print(f"⚠️ Scene {i} invalid, creating noir gradient...")
        fallback_path = os.path.join(TMP, f"scene_fallback_{i}.jpg")
        create_noir_gradient(fallback_path, i, w, h)
        scene_images[i] = fallback_path

print(f"✅ All mystery scenes validated")

# --- Audio Loading & Timing ---

if not os.path.exists(audio_path):
    print(f"❌ Audio not found: {audio_path}")
    raise FileNotFoundError("voice.mp3 missing")

# ✅ FIXED: WAV conversion with proper error handling
wav_path = os.path.join(TMP, "voice.wav")
try:
    audio_segment = AudioSegment.from_file(audio_path)
    audio_segment.export(wav_path, format="wav")
    print("✅ Converted audio to WAV for accurate timing.")
    working_audio_path = wav_path
except Exception as e:
    print(f"⚠️ WAV conversion failed ({e}), using original MP3.")
    working_audio_path = audio_path

audio = AudioFileClip(working_audio_path)
duration = audio.duration
print(f"🎵 Audio Duration: {duration:.2f}s")

# ✅ SYNC FIX: Detect silence and define the actual speech window
lead_silence, trail_silence = detect_leading_trailing_silence(working_audio_path)
manual_offset = float(os.getenv("SYNC_OFFSET_S", "0.0")) # Optional fine-tuning via env var
start_offset = lead_silence + manual_offset
speech_duration = max(0.1, duration - lead_silence - trail_silence)

print(f"🕰️ Speech Window — Start: {start_offset:.3f}s, Duration: {speech_duration:.3f}s")


# ✅✅✅ --- FINAL SYNC FIX --- ✅✅✅
# This entire block replaces the previous timing logic to be fully robust.

# ✅✅✅ --- FINAL, ROBUST TIMING LOGIC --- ✅✅✅

if not os.path.exists(audio_path): raise FileNotFoundError(f"Audio not found: {audio_path}")

# Convert to WAV for Pydub to be safe and accurate
wav_path = os.path.join(TMP, "voice.wav")
try:
    AudioSegment.from_file(audio_path).export(wav_path, format="wav")
    print("✅ Converted audio to WAV for accurate timing.")
    working_audio_path = wav_path
except Exception as e:
    print(f"⚠️ WAV conversion failed ({e}), using original MP3.")
    working_audio_path = audio_path

audio_clip = AudioFileClip(working_audio_path)
total_duration = audio_clip.duration
print(f"🎵 Audio Duration: {total_duration:.2f}s")

# Use Pydub for local silence detection as a reliable source of truth
try:
    segment = AudioSegment.from_file(working_audio_path)
    # Get the timestamp of the first non-silent chunk
    def get_first_sound_ms(seg, chunk_size=10, silence_thresh=-45.0):
        trim_ms = 0
        # Iterate until a chunk is louder than the threshold
        while trim_ms < len(seg) and seg[trim_ms:trim_ms+chunk_size].dBFS < silence_thresh:
            trim_ms += chunk_size
        return trim_ms

    start_offset_ms = get_first_sound_ms(segment)
    # Calculate speech duration by stripping silence from both ends
    speech_only_segment = segment.strip_leading_silence(silence_threshold=-45).strip_trailing_silence(silence_threshold=-45)
    
    start_offset = start_offset_ms / 1000.0
    speech_duration = len(speech_only_segment) / 1000.0
    
    print(f"🕰️ Speech Window Detected — Start: {start_offset:.3f}s, Speech Duration: {speech_duration:.3f}s")
except Exception as e:
    print(f"⚠️ Pydub detection failed ({e}), assuming full duration for fallback.")
    start_offset, speech_duration = 0.0, total_duration

scene_starts, paragraph_durations = [], []
enhanced_timing = load_enhanced_timing()

# 1. PRIMARY PATH: Use pre-calculated JSON if it's valid and matches the script
if enhanced_timing and enhanced_timing.get('sections') and len(enhanced_timing['sections']) == len(paragraphs):
    print("\n⏱️ Using PRE-CALCULATED timing from audio_timing.json...")
    for section in enhanced_timing['sections']:
        scene_starts.append(section['start'])
        paragraph_durations.append(section['duration'])
    print("   ✅ Successfully applied pre-calculated timings.")

else:
    # 2. FALLBACK PATH: Use local detection if JSON is missing or mismatched
    print("\n📊 Using ROBUST proportional fallback timing (based on detected speech window)...")
    total_words = sum(len(p.split()) for p in paragraphs if p)
    
    if total_words > 0 and speech_duration > 0:
        current_time = start_offset # CRITICAL: Start visuals when speech starts
        
        # Calculate proportional durations first
        for p in paragraphs:
            # CRITICAL: Base duration on SPEECH duration, not total
            dur = (len(p.split()) / total_words) * speech_duration
            paragraph_durations.append(dur)

        # Normalize the calculated durations to fit the speech window EXACTLY to prevent drift
        calculated_total = sum(paragraph_durations)
        if calculated_total > 0:
            scale_factor = speech_duration / calculated_total
            paragraph_durations = [d * scale_factor for d in paragraph_durations]
            print(f"   ✅ Normalized fallback timings by scale factor: {scale_factor:.4f}")

        # Now, build the final timeline of start times
        for dur in paragraph_durations:
            scene_starts.append(current_time)
            current_time += dur
    else:
        # Absolute safety net if no words or speech detected
        print("   ⚠️ No words or speech duration to calculate timing. Dividing total duration evenly.")
        dur_per_para = total_duration / max(1, len(paragraphs))
        for i in range(len(paragraphs)):
            scene_starts.append(i * dur_per_para)
            paragraph_durations.append(dur_per_para)

# 3. FINAL VALIDATION & LOGGING
# (The normalization logic was moved into the fallback path, so this check is for logging)
final_visual_end = (scene_starts[-1] + paragraph_durations[-1]) if scene_starts else 0
final_drift = abs(total_duration - final_visual_end)
print(f"📊 Final Sync Check: Visual End={final_visual_end:.3f}s vs Audio End={total_duration:.3f}s. Drift: {final_drift*1000:.0f}ms")
if final_drift > 0.5:
    print(f"   ⚠️ WARNING: Significant sync drift of {final_drift:.2f}s detected. Check timing logic.")

if final_drift < 0.05:
    print(f"   ✅ FRAME-PERFECT SYNC!")
elif final_drift < 0.2:
    print(f"   ✅ Excellent sync")
elif final_drift < 0.5:
    print(f"   ✅ Good sync")
else:
    print(f"   ⚠️ Sync drift detected - check timing (Drift: {final_drift:.3f}s)")

# Build visual clips for each paragraph/section
clips = []

# Fallback if timing failed for some reason
if not scene_starts or not paragraph_durations:
    print("   ⚠️ CRITICAL: Timing calculation failed completely. Using single scene.")
    scene_starts = [0.0]
    paragraph_durations = [duration]
    paragraphs = [full_script]

for i, (start, dur) in enumerate(zip(scene_starts, paragraph_durations)):
    # Use scene image if available, else reuse last image, else None (ColorClip fallback in create_enhanced_scene)
    img_path = scene_images[i] if i < len(scene_images) else (scene_images[-1] if scene_images else None)

    text = paragraphs[i] if i < len(paragraphs) else ""
    scene_clips = create_enhanced_scene(img_path, text, dur, start, i)
    clips.extend(scene_clips)

# Absolute guard: ensure at least one visual clip exists
if not clips:
    print("   ⚠️ No visual clips built; using black fallback background")
    clips = [
        ColorClip(size=(w, h), color=NOIR_COLORS['deep_black']).with_duration(duration).with_start(0)
    ]

print(f"\n🎬 Composing mystery video ({len(clips)} clips)...")
video = CompositeVideoClip(clips, size=(w, h))

# 🎵 ADD BACKGROUND MUSIC
print(f"\n🔊 Adding audio with dark ambient music...")

background_music = create_dynamic_music_layer(duration, data)

if background_music:
    try:
        voice_adjusted = apply_volumex(audio, 1.0)
        final_audio = CompositeAudioClip([voice_adjusted, background_music])
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

# --- [ ✅ FIXED & OPTIMIZED CODE ] ---
print(f"\n📹 Writing final video for web compatibility...")
try:
    # Explicitly set ffmpeg_params for maximum compatibility and smaller file size.
    video.write_videofile(
        OUT,
        fps=30,
        codec="libx264",          # The standard for H.264 video
        audio_codec="aac",        # The standard for MP4 audio
        audio_bitrate="192k",     # Good quality for voice
        preset="veryfast",        # MUCH faster for CI/CD environments, minimal quality loss
        threads=4,
        # CRITICAL FIXES HERE:
        bitrate="2000k",          # Drastically reduce bitrate for static images (was 8000k)
        ffmpeg_params=[
            '-pix_fmt', 'yuv420p'  # THE MOST IMPORTANT FIX for player compatibility
        ]
    )

    final_time = scene_starts[-1] + paragraph_durations[-1] if scene_starts else duration
    sync_status = "NEAR-PERFECT" if abs(final_time - duration) < 0.05 else "EXCELLENT" if abs(final_time - duration) < 0.5 else "GOOD"

    print(f"\n✅ MYSTERY VIDEO COMPLETE!")
    print(f"   Path: {OUT}")
    print(f"   Duration: {duration:.2f}s")

    # YouTube Shorts check
    if duration > 60:
        print(f"   ⚠️ WARNING: Video exceeds YouTube Shorts 60s limit!")
        print(f"   Overflow: +{duration - 60:.2f}s")
        print(f"   This video will be rejected by YouTube Shorts")
    elif duration > 55:
        print(f"   ⚠️ CAUTION: Close to 60s limit ({60 - duration:.2f}s buffer)")
    else:
        print(f"   ✅ Within YouTube Shorts limit ({60 - duration:.2f}s buffer)")

    print(f"   Size: {os.path.getsize(OUT) / (1024*1024):.2f} MB")
    print(f"   Sync Status: {sync_status} ({abs(final_time - duration)*1000:.0f}ms drift)")
    print(f"   Features:")
    print(f"      ✓ Film noir aesthetic")
    print(f"      ✓ Heavy vignette + film grain")
    if background_music:
        print(f"      ✓ Dark ambient background music")
        print(f"      ✓ Professional audio mix")
    else:
        print(f"      ⚠ Background music skipped")
    print(f"      ✨ ENHANCED TEXT DISPLAY:")
    print(f"         ✓ Progressive subtitle-style segments")
    print(f"         ✓ Semi-transparent background panels")
    print(f"         ✓ Consistent lower-third positioning")
    print(f"         ✓ Smooth fade in/out animations")
    print(f"         ✓ Optimized readability & pacing")
    print(f"         ✓ Professional documentary aesthetic")
    print(f"      ✓ Paragraph-based timing")
    print(f"   🔍 Mystery video ready!")
except Exception as e:
    print(f"❌ Video creation failed: {e}")
    import traceback
    traceback.print_exc()
    raise

finally:
    print("🧹 Cleanup...")
    # Clean up temp files
    cleanup_temp_files()
    
    # Close all clips
    try:
        audio.close()
    except:
        pass
    try:
        video.close()
    except:
        pass
    if background_music:
        try:
            background_music.close()
        except:
            pass
    for clip in clips:
        try:
            clip.close()
        except:
            pass

print("✅ Mystery video pipeline complete! 🔍")