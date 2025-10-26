#!/usr/bin/env python3
"""
🔍 Create Mystery Video - PRODUCTION VERSION WITH ENHANCED TEXT DISPLAY
Python 3.11 + MoviePy 2.0+ Compatible (GitHub Actions Ready)
ENHANCED: Professional subtitle system with cinematic text presentation
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

# 🎵 Import music system
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

MUSIC_AVAILABLE = False
try:
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
            "https://image.pollinaions.ai/prompt/"
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
                img = img.resized((width, height), Image.LANCZOS)
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


# ✨ NEW ENHANCED TEXT FUNCTIONS ✨

def segment_text_for_display(text, max_words_per_segment=15, max_segments=3):
    """
    Break text into readable segments for progressive display
    Tries to break at natural sentence boundaries
    """
    sentences = text.replace('...', '.').split('.')
    sentences = [s.strip() for s in sentences if s.strip()]
    
    segments = []
    current_segment = []
    current_word_count = 0
    
    for sentence in sentences:
        words = sentence.split()
        
        # If adding this sentence would exceed max words, start new segment
        if current_word_count + len(words) > max_words_per_segment and current_segment:
            segments.append(' '.join(current_segment) + '.')
            current_segment = []
            current_word_count = 0
        
        # Add sentence to current segment
        current_segment.append(sentence)
        current_word_count += len(words)
        
        # Check if we've reached max segments
        if len(segments) >= max_segments - 1:
            break
    
    # Add remaining text as final segment
    if current_segment:
        segments.append(' '.join(current_segment) + '.')
    
    # If no sentences found, fall back to word-based splitting
    if not segments:
        words = text.split()
        for i in range(0, len(words), max_words_per_segment):
            segment = ' '.join(words[i:i+max_words_per_segment])
            segments.append(segment)
            if len(segments) >= max_segments:
                break
    
    return segments[:max_segments]


# ✨ REPLACEMENT TEXT SYSTEM - CINEMATIC & ROBUST ✨

def create_text_panel(width, height, opacity=0.85):
    """
    Create enhanced noir text panel with better readability.
    - Darker background for vertical video
    - Subtle film grain texture
    - Gold border accent
    (This function is well-designed and is kept from the original)
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
    (This function is well-designed and is kept from the original)
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

def segment_text_for_display(text, max_words_per_segment=10, max_segments=4):
    """
    Break text into readable segments for progressive display.
    Tries to break at natural sentence boundaries.
    (This function is well-designed and is kept from the original)
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

def create_cinematic_text_clip(text, font_size, duration, start_time, position='lower_third', panel_bg=True):
    """
    ✅ FIXED: Creates a robust, pre-rendered text clip using PIL.
    This avoids MoviePy's text rendering bugs and ensures perfect alignment.
    The text and its background panel are combined into a single image.
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
    except Exception: # Fallback for older PIL
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
    ✅ FIXED & REWRITTEN: Creates a scene with professionally timed and rendered text.
    - Breaks long paragraphs into readable, sequential chunks.
    - Each chunk is a self-contained, faded clip, perfectly timed.
    - Eliminates sync issues and rendering bugs.
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
        
    # --- FIXED SEGMENTATION AND TIMING LOGIC ---
    
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
    num_scenes = min(4, len(paragraphs))
    
    for i in range(num_scenes):
        paragraph = paragraphs[i] if i < len(paragraphs) else ""
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
    scene_images = [None] * 4

print(f"🔍 Validating {len(scene_images)} scenes...")
for i in range(len(scene_images)):
    img = scene_images[i] if i < len(scene_images) else None
    
    if not img or not os.path.exists(img) or os.path.getsize(img) < 1000:
        print(f"⚠️ Scene {i} invalid, creating noir gradient...")
        fallback_path = os.path.join(TMP, f"scene_fallback_{i}.jpg")
        create_noir_gradient(fallback_path, i, w, h)
        if i < len(scene_images):
            scene_images[i] = fallback_path

print(f"✅ All mystery scenes validated")

# --- Audio Loading & Timing ---

if not os.path.exists(audio_path):
    print(f"❌ Audio not found: {audio_path}")
    raise FileNotFoundError("voice.mp3 missing")

audio = AudioFileClip(audio_path)
duration = audio.duration
print(f"🎵 Audio: {duration:.2f}s")

timing_data = load_audio_timing()

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
            dur = estimate_duration_fallback(paragraph, duration, paragraphs)
            paragraph_durations.append(dur)
            print(f"   Paragraph {i+1}: {dur:.2f}s (estimated)")
    
    total_calculated = sum(paragraph_durations)
    
    if abs(total_calculated - duration) > 0.5:
        adjustment_factor = duration / total_calculated if total_calculated > 0 else 1
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

# --- Video Composition with Enhanced Text ---

clips = []
current_time = 0

print("\n🎬 Building scenes with enhanced text display...")

for i, paragraph in enumerate(paragraphs):
    dur = paragraph_durations[i]
    
    img_idx = min(i, len(scene_images) - 1)
    
    print(f"🎬 Scene {i+1}/{len(paragraphs)} (duration: {dur:.2f}s)...")
    
    # Use enhanced scene creation
    clips.extend(create_enhanced_scene(
        scene_images[img_idx], 
        paragraph,
        dur,
        current_time,
        scene_index=i
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

# 🎵 ADD BACKGROUND MUSIC
print(f"\n🔊 Adding audio with dark ambient music...")

background_music = create_dynamic_music_layer(duration, data)

if background_music:
    try:
        voice_adjusted = apply_volumex(audio, 1.0)

        final_audio = CompositeAudioClip([voice_adjusted, background_music])
        # ✅ FIXED: Use with_audio instead of set_audio (MoviePy 2.0+)
        video = video.with_audio(final_audio)
        print(f"   ✅ Audio: TTS + Dark ambient music")
    except Exception as e:
        print(f"   ⚠️ Music compositing failed: {e}")
        import traceback
        traceback.print_exc()
        # ✅ FIXED: Use with_audio for fallback too
        video = video.with_audio(audio)
else:
    # ✅ FIXED: Use with_audio for fallback
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
    print(f"   Sync Status: {sync_status} ({abs(current_time - duration)*1000:.0f}ms drift)")
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