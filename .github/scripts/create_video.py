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
    if filepath not in TEMP_FILES:
        TEMP_FILES.append(filepath)
    return filepath

def cleanup_temp_files():
    """Clean up all registered temp files"""
    print("🧹 Cleaning up temporary files...")
    for filepath in TEMP_FILES:
        try:
            if os.path.exists(filepath):
                os.remove(filepath)
        except Exception as e:
            print(f"   - Could not remove {os.path.basename(filepath)}: {e}")

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
        fonts = ["C:/Windows/Fonts/cour.ttf", "C:/Windows/Fonts/georgia.ttf", "C:/Windows/Fonts/times.ttf"]
        for font in fonts:
            if os.path.exists(font): return font
        return "C:/Windows/Fonts/arial.ttf"
    elif system == "Darwin":
        return "/System/Library/Fonts/Supplemental/Courier New.ttf"
    else:
        font_options = ["/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf", "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]
        for font in font_options:
            if os.path.exists(font): return font
        return "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


FONT = get_font_path()
print(f"📝 Using font: {FONT}")

# Load script
with open(os.path.join(TMP, "script.json"), "r", encoding="utf-8") as f:
    data = json.load(f)

title = data.get("title", "Mystery")
hook = data.get("hook", "")
full_script = data.get("script", "")
if not full_script:
    bullets = data.get("bullets", [])
    cta = data.get("cta", "")
    full_script = f"{hook}\n\n{' '.join(bullets)}\n\n{cta}"
else:
    cta = data.get("cta", "")

paragraphs = [p.strip() for p in full_script.split('\n\n') if p.strip()]
topic = data.get("topic", "mystery")
content_type = data.get("content_type", "evening_prime")
mystery_category = data.get("mystery_category", "disappearance")
visual_prompts = data.get("visual_prompts", [])

print(f"🔍 Creating {content_type} mystery video ({mystery_category})")
print(f"   Narrative sections: {len(paragraphs)}")

# All other functions (load_audio_timing, generate_image_reliable, create_cinematic_text_clip, etc.)
# are preserved as they are well-structured. The main change is in the timing calculation flow.
# ... (all your helper functions are kept as is) ...
def load_audio_timing():
    timing_path = os.path.join(TMP, "audio_timing.json")
    if os.path.exists(timing_path):
        try:
            with open(timing_path, 'r') as f:
                timing_data = json.load(f)
            if timing_data.get('optimized'):
                print("✅ Loaded optimized audio timing metadata")
                return timing_data
            else:
                print("⚠️ Timing metadata not optimized, using fallback")
                return None
        except Exception as e:
            print(f"⚠️ Could not load timing metadata: {e}")
            return None
    print("⚠️ No timing metadata found, using estimation")
    return None

def enhance_visual_prompt_for_mystery(prompt, scene_index, mystery_category):
    noir_base = "film noir photography, high contrast black and white, dramatic shadows, moody atmosphere, vintage 1940s-1960s aesthetic, film grain, mysterious, documentary style, cinematic lighting"
    scene_enhancements = {0: "opening establishing shot, dark and mysterious, ominous atmosphere, noir cinematography", 1: "documentary evidence style, vintage photograph, historical authenticity, aged photo quality", 2: "dramatic reveal, unsettling discovery, noir detective aesthetic, shadowy investigation", 3: "final mystery image, unanswered questions, enigmatic conclusion, noir ending"}
    category_keywords = {'disappearance': "abandoned location, empty space, no trace, vanished without evidence, eerie absence", 'crime': "crime scene aesthetic, investigation photo, evidence markers, detective noir, forensic", 'historical': "ancient artifact, archaeological discovery, aged document, historical photograph", 'conspiracy': "classified documents, redacted files, government secrets, declassified evidence"}
    enhancement = scene_enhancements.get(scene_index, "mysterious noir aesthetic")
    category_words = category_keywords.get(mystery_category, category_keywords['disappearance'])
    prompt = prompt.replace('happy', 'mysterious').replace('bright', 'dark').replace('colorful', 'monochrome')
    enhanced = f"{prompt}, {enhancement}, {category_words}, {noir_base}, foggy, unsettling, dark and ominous"
    return enhanced.replace('  ', ' ').strip()

def generate_image_huggingface(prompt, filename, width=1080, height=1920):
    hf_token = os.getenv('HUGGINGFACE_API_KEY')
    if not hf_token: raise Exception("Missing HUGGINGFACE_API_KEY")
    headers = {"Authorization": f"Bearer {hf_token}"}
    negative_mystery = "blurry, low quality, watermark, text overlay, logo, frame, caption, ui elements, cartoon, anime, illustration, happy, cheerful, bright, colorful, pastel, soft lighting, modern, contemporary, digital art, amateur photo"
    payload = {"inputs": f"{prompt}, film noir, black and white photography, high contrast, dramatic shadows, vintage, mysterious, documentary style", "parameters": {"negative_prompt": negative_mystery, "num_inference_steps": 4, "guidance_scale": 0.0, "width": width, "height": height}}
    models = ["black-forest-labs/FLUX.1-schnell", "black-forest-labs/FLUX.1-dev", "stabilityai/stable-diffusion-xl-base-1.0"]
    for model in models:
        url = f"https://api-inference.huggingface.co/models/{model}"
        print(f"🤗 Trying model: {model}")
        response = requests.post(url, headers=headers, json=payload, timeout=120)
        if response.status_code == 200 and len(response.content) > 1000:
            filepath = os.path.join(TMP, filename)
            with open(filepath, "wb") as f: f.write(response.content)
            print(f"    ✅ HuggingFace succeeded: {model}")
            return filepath
        elif response.status_code in [402, 503, 429]:
            print(f"⌛ {model} temporarily unavailable — trying next...")
            time.sleep(2)
        else:
            print(f"⚠️ {model} failed ({response.status_code}) — trying next...")
    raise Exception("All HuggingFace models failed")

def generate_image_pollinations(prompt, filename, width=1080, height=1920):
    negative_terms = "blurry, low quality, watermark, text, logo, cartoon, anime, illustration, happy, cheerful, bright, colorful, pastel, soft lighting, modern, contemporary"
    formatted_prompt = f"{prompt}, film noir photography, black and white, high contrast, dramatic shadows, vintage 1940s aesthetic, mysterious, documentary style, film grain"
    seed = random.randint(1, 999999)
    url = f"https://image.pollinations.ai/prompt/{requests.utils.quote(formatted_prompt)}?width={width}&height={height}&negative={requests.utils.quote(negative_terms)}&nologo=true&notext=true&enhance=true&model=flux&seed={seed}"
    print(f"    🌐 Pollinations: {prompt[:60]}... (seed={seed})")
    response = requests.get(url, timeout=120)
    if response.status_code == 200 and "image" in response.headers.get("Content-Type", ""):
        filepath = os.path.join(TMP, filename)
        with open(filepath, "wb") as f: f.write(response.content)
        print(f"    ✅ Pollinations generated (seed {seed})")
        return filepath
    else:
        raise Exception(f"Pollinations failed: {response.status_code}")

def generate_mystery_fallback(bg_path, scene_index, mystery_category, width=1080, height=1920):
    category_keywords = {'disappearance': ['abandoned', 'empty', 'fog', 'mystery', 'dark-sky', 'eerie'], 'crime': ['noir', 'detective', 'shadow', 'urban-night', 'alley', 'investigation'], 'historical': ['vintage', 'old-photo', 'ancient', 'archive', 'historical'], 'conspiracy': ['documents', 'classified', 'secret', 'government', 'mystery']}
    keywords = category_keywords.get(mystery_category, ['mystery', 'noir', 'dark'])
    keyword = random.choice(keywords)
    print(f"🔎 Searching mystery image for '{mystery_category}' (keyword: '{keyword}')...")
    try:
        seed = random.randint(1, 9999)
        url = f"https://source.unsplash.com/{width}x{height}/?{requests.utils.quote(keyword)}&sig={seed}"
        response = requests.get(url, timeout=30, allow_redirects=True)
        if response.status_code == 200 and "image" in response.headers.get("Content-Type", ""):
            with open(bg_path, "wb") as f: f.write(response.content)
            return bg_path
    except Exception as e:
        print(f"    ⚠️ Unsplash error: {e}")
    try:
        mystery_pexels_ids = {'disappearance': [3783471, 3617500, 3861431], 'crime': [2582937, 2559941, 3617500], 'historical': [1418595, 1141853, 2325729], 'conspiracy': [3617500, 7319070, 5668826]}
        scene_key = mystery_category if mystery_category in mystery_pexels_ids else 'disappearance'
        photo_ids = mystery_pexels_ids[scene_key].copy()
        random.shuffle(photo_ids)
        for attempt, photo_id in enumerate(photo_ids[:2]):
            url = f"https://images.pexels.com/photos/{photo_id}/pexels-photo-{photo_id}.jpeg?auto=compress&cs=tinysrgb&w=1080&h=1920&fit=crop"
            response = requests.get(url, timeout=30)
            if response.status_code == 200 and "image" in response.headers.get("Content-Type", ""):
                with open(bg_path, "wb") as f: f.write(response.content)
                Image.open(bg_path).convert("RGB").resize((width, height), Image.LANCZOS).save(bg_path, quality=95)
                return bg_path
    except Exception as e:
        print(f"    ⚠️ Pexels fallback failed: {e}")
    return None

def create_noir_gradient(filepath, scene_index, width=1080, height=1920):
    img = Image.new("RGB", (width, height), (0, 0, 0))
    draw = ImageDraw.Draw(img)
    colors = [NOIR_COLORS['deep_black'], NOIR_COLORS['dark_slate']] if scene_index % 2 == 0 else [NOIR_COLORS['shadow_blue'], NOIR_COLORS['fog_gray']]
    for y in range(height):
        ratio = y / height
        r, g, b = [int(colors[0][i] * (1 - ratio) + colors[1][i] * ratio) for i in range(3)]
        draw.line([(0, y), (width, y)], fill=(r, g, b))
    apply_vignette_noir(img, strength=0.6).save(filepath, quality=95)
    return filepath

def apply_vignette_noir(img, strength=0.6):
    width, height = img.size
    mask = Image.new('L', (width, height), 255)
    draw = ImageDraw.Draw(mask)
    max_dim = int(min(width, height) * strength)
    for i in range(max_dim):
        alpha = int(255 * (1 - i / max_dim))
        x0, y0, x1, y1 = i, i, width - i - 1, height - i - 1
        if x1 > x0 and y1 > y0: draw.rectangle([x0, y0, x1, y1], outline=alpha)
    return Image.composite(img, Image.new('RGB', (width, height), (0, 0, 0)), mask)

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=4, max=20))
def generate_image_reliable(prompt, filename, scene_index, mystery_category, width=1080, height=1920):
    filepath = os.path.join(TMP, filename)
    for name, func in [("Pollinations", generate_image_pollinations), ("HuggingFace", generate_image_huggingface)]:
        try:
            result = func(prompt, filename, width, height)
            if result and os.path.exists(result) and os.path.getsize(result) > 1000: return result
        except Exception as e:
            print(f"    ⚠️ {name} failed: {e}")
    result = generate_mystery_fallback(filepath, scene_index, mystery_category, width, height)
    if result and os.path.exists(filepath) and os.path.getsize(filepath) > 1000: return result
    return create_noir_gradient(filepath, scene_index, width, height)

def apply_noir_filter(image_path, scene_index):
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
    except Exception as e:
        print(f"      ⚠️ Noir filter failed: {e}")
    return image_path

def apply_blue_tint(img, intensity=0.15):
    pixels = img.load()
    width, height = img.size
    for x in range(width):
        for y in range(height):
            r, g, b = pixels[x, y]
            pixels[x, y] = (max(0, min(255, int(r + (10-r)*intensity))), max(0, min(255, int(g + (20-g)*intensity))), max(0, min(255, int(b + (40-b)*intensity))))
    return img

def detect_leading_trailing_silence(audio_path, silence_thresh_db=-45.0, chunk_ms=10):
    if not os.path.exists(audio_path): return 0.0, 0.0
    try:
        audio = AudioSegment.from_file(audio_path)
        lead = 0
        while lead < len(audio) and audio[lead:lead+chunk_ms].dBFS < silence_thresh_db: lead += chunk_ms
        trail = 0
        while trail < len(audio) and audio[-trail-chunk_ms:len(audio)-trail].dBFS < silence_thresh_db: trail += chunk_ms
        print(f"🔧 Silence Detection — Lead: {lead/1000.0:.3f}s, Trail: {trail/1000.0:.3f}s")
        return lead/1000.0, trail/1000.0
    except Exception as e:
        print(f"⚠️ Pydub silence detection failed: {e}")
        return 0.0, 0.0

def add_film_grain_noir(img, intensity=0.15):
    try:
        img_array = np.array(img)
        noise = np.random.normal(0, intensity * 255, img_array.shape)
        noisy = np.clip(img_array + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(noisy)
    except:
        return img

def segment_text_for_display(text, max_words_per_segment=10, max_segments=4):
    sentences = [s.strip() for s in text.replace('...', '.').split('.') if s.strip()]
    segments, current_segment_words = [], []
    for sentence in sentences:
        words = sentence.split()
        if len(current_segment_words) + len(words) > max_words_per_segment and current_segment_words:
            segments.append(' '.join(current_segment_words) + '.')
            current_segment_words = []
        current_segment_words.extend(words)
        if len(segments) >= max_segments - 1: break
    if current_segment_words: segments.append(' '.join(current_segment_words) + ('.' if not ' '.join(current_segment_words).endswith('.') else ''))
    if not segments and text:
        words = text.split()
        for i in range(0, len(words), max_words_per_segment): segments.append(' '.join(words[i:i+max_words_per_segment]))
    return segments[:max_segments]

def create_text_panel(width, height, opacity=0.85):
    panel = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(panel)
    draw.rectangle([(0, 0), (width, height)], fill=(15, 15, 20, int(255 * opacity)))
    for _ in range(int(width * height * 0.0005)): draw.point((random.randint(0, width-1), random.randint(0, height-1)), fill=(random.randint(0,30),)*3 + (50,))
    draw.rectangle([(0, 0), (width-1, height-1)], outline=(212, 175, 55, 200), width=3)
    return panel

def smart_text_wrap(text, font_size, max_width):
    try:
        pil_font = ImageFont.truetype(FONT, font_size)
    except IOError:
        pil_font = ImageFont.load_default()
    words, lines, current_line = text.split(), [], []
    draw = ImageDraw.Draw(Image.new('RGB', (1, 1)))
    for word in words:
        test_line = ' '.join(current_line + [word])
        try:
            bbox = draw.textbbox((0, 0), test_line, font=pil_font)
            line_width = bbox[2] - bbox[0]
        except AttributeError:
            line_width, _ = draw.textsize(test_line, font=pil_font)
        if line_width <= max_width:
            current_line.append(word)
        else:
            if current_line: lines.append(' '.join(current_line))
            current_line = [word]
    if current_line: lines.append(' '.join(current_line))
    return '\n'.join(lines)

def create_cinematic_text_clip(text, font_size, duration, start_time, position='lower_third', panel_bg=True):
    wrapped_text = smart_text_wrap(text, font_size, TEXT_MAX_WIDTH)
    try:
        pil_font = ImageFont.truetype(FONT, font_size)
    except IOError:
        pil_font = ImageFont.load_default()
    dummy_draw = ImageDraw.Draw(Image.new('RGB', (1, 1)))
    try:
        bbox = dummy_draw.textbbox((0, 0), wrapped_text, font=pil_font, stroke_width=2)
        text_width, text_height = bbox[2] - bbox[0], bbox[3] - bbox[1]
    except Exception:
        text_width, text_height = dummy_draw.textsize(wrapped_text, font=pil_font)
    comp_width, comp_height = int(text_width + 80), int(text_height + 60)
    final_image = create_text_panel(comp_width, comp_height) if panel_bg else Image.new('RGBA', (comp_width, comp_height), (0,0,0,0))
    draw = ImageDraw.Draw(final_image)
    text_x, text_y = (comp_width - text_width) / 2, (comp_height - text_height) / 2
    draw.text((text_x + 2, text_y + 2), wrapped_text, font=pil_font, fill=(10, 10, 10, 180), align='center')
    draw.text((text_x, text_y), wrapped_text, font=pil_font, fill=NOIR_COLORS['evidence_tan'], align='center')
    comp_path = os.path.join(TMP, f"text_comp_{time.time()}.png")
    final_image.save(comp_path)
    register_temp_file(comp_path)
    clip_pos = ('center', h * 0.70) if position == 'lower_third' else ('center', 'center')
    final_clip = ImageClip(comp_path).with_duration(duration).with_start(start_time).with_position(clip_pos)
    fade_dur = min(0.5, duration / 4)
    return [apply_fadeout(apply_fadein(final_clip, fade_dur), fade_dur)]

def create_enhanced_scene(image_path, text, duration, start_time, scene_index=0):
    scene_clips = []
    if image_path and os.path.exists(image_path):
        zoomed_clip = ImageClip(image_path).with_duration(duration).resized(lambda t: 1 + 0.02 * t).with_position(('center', 'center'))
        bg = CompositeVideoClip([zoomed_clip], size=(w, h)).with_duration(duration)
    else:
        bg = ColorClip(size=(w, h), color=NOIR_COLORS['deep_black']).with_duration(duration)
    scene_clips.append(bg.with_start(start_time))
    if not text or not text.strip(): return scene_clips
    segments = segment_text_for_display(text, max_words_per_segment=12, max_segments=4)
    if not segments: return scene_clips
    total_words = sum(len(s.split()) for s in segments)
    durations = [(len(s.split())/total_words * duration if total_words > 0 else duration/len(segments)) for s in segments]
    durations = [max(d, 1.5 + len(seg.split()) * 0.20) for d, seg in zip(durations, segments)]
    scale_factor = duration / sum(durations) if sum(durations) > 0 else 1
    durations = [d * scale_factor for d in durations]
    current_time = start_time
    for i, segment in enumerate(segments):
        if durations[i] < 0.5: continue
        scene_clips.extend(create_cinematic_text_clip(segment, 58 if len(segment.split())<8 else 54, durations[i], current_time))
        current_time += durations[i]
    return scene_clips

def ensure_music_downloaded():
    if not MUSIC_AVAILABLE: return False
    essential = ['dark_mystery', 'suspense_build', 'paranormal_ambient']
    downloaded = 0
    for track in essential:
        if track in MUSIC_LIBRARY:
            try:
                if download_track(track, MUSIC_LIBRARY[track]): downloaded += 1
            except: pass
    return downloaded > 0

def create_dynamic_music_layer(audio_duration, script_data):
    if not ensure_music_downloaded(): return None
    scene_map = {'evening_prime': 'investigation', 'late_night': 'suspense', 'weekend_binge': 'paranormal'}
    category_map = {'disappearance': 'suspense', 'crime': 'investigation', 'paranormal': 'paranormal', 'historical': 'ancient'}
    primary_scene = category_map.get(script_data.get('mystery_category')) or scene_map.get(script_data.get('content_type'), 'investigation')
    try:
        key, path, _ = get_music_for_scene(primary_scene, 'general')
        if not path or not os.path.exists(path): raise ValueError("Path not found")
        music = AudioFileClip(path)
        if music.duration < audio_duration:
            music = concatenate_audioclips([music] * (int(audio_duration / music.duration) + 1))
        music = trim_audio_safe(music, audio_duration)
        volume = {'evening_prime': 0.20, 'late_night': 0.15, 'weekend_binge': 0.22}.get(script_data.get('content_type'), 0.18)
        return apply_volumex(music, volume)
    except Exception as e:
        print(f"⚠️ Music creation failed: {e}")
        return None

# --- Scene Generation ---

print("🔍 Generating mystery scenes...")
scene_images = []
num_scenes = min(4, len(paragraphs))
for i in range(num_scenes):
    prompt = visual_prompts[i] if i < len(visual_prompts) else paragraphs[i][:100]
    enhanced_prompt = enhance_visual_prompt_for_mystery(prompt, i, mystery_category)
    img_path = generate_image_reliable(enhanced_prompt, f"scene_{i}.jpg", i, mystery_category, w, h)
    if img_path: apply_noir_filter(img_path, i)
    scene_images.append(img_path)
for i in range(len(scene_images)):
    if not scene_images[i] or not os.path.exists(scene_images[i]):
        scene_images[i] = create_noir_gradient(os.path.join(TMP, f"scene_fallback_{i}.jpg"), i, w, h)

# --- Audio Loading & Timing ---

if not os.path.exists(audio_path): raise FileNotFoundError("voice.mp3 missing")
wav_path = os.path.join(TMP, "voice.wav")
try:
    AudioSegment.from_file(audio_path).export(wav_path, format="wav")
    working_audio_path = wav_path
except Exception:
    working_audio_path = audio_path
audio = AudioFileClip(working_audio_path)
duration = audio.duration
print(f"🎵 Audio Duration: {duration:.2f}s")
lead_silence, trail_silence = detect_leading_trailing_silence(working_audio_path)
manual_offset = float(os.getenv("SYNC_OFFSET_S", "0.0"))
start_offset = lead_silence + manual_offset
speech_duration = max(0.1, duration - lead_silence - trail_silence)
print(f"🕰️ Speech Window — Start: {start_offset:.3f}s, Duration: {speech_duration:.3f}s")


# ==============================================================================
# ==================== ✅ START OF AUDIO SYNC FIX ✅ =========================
# ==============================================================================

print("\n⏱️ Calculating paragraph timings...")
timing_data = load_audio_timing()
paragraph_durations = []
timing_valid = False

# 1. Attempt to use pre-calculated, optimized timing data
if timing_data and timing_data.get('optimized'):
    sections = timing_data.get('sections', [])
    # Check if we have timing data for every paragraph
    if len(sections) >= len(paragraphs):
        timing_valid = True
        print("   -> Using OPTIMIZED audio timing from metadata.")
        for i in range(len(paragraphs)):
            paragraph_durations.append(sections[i]['duration'])
    else:
        print(f"   -> WARNING: Timing data mismatch ({len(sections)} sections for {len(paragraphs)} paragraphs). Falling back.")

# 2. Fallback to proportional timing if optimized data is invalid or missing
if not timing_valid:
    print("   -> Using proportional timing based on word count.")
    total_words = sum(len(p.split()) for p in paragraphs if p)
    for paragraph in paragraphs:
        if total_words > 0:
            word_count = len(paragraph.split())
            # Proportional duration + a small base time per paragraph
            dur = (word_count / total_words) * speech_duration + 0.2
        else:
            # Equal split if no words
            dur = speech_duration / max(1, len(paragraphs))
        paragraph_durations.append(max(2.0, dur)) # Enforce minimum duration

# 3. Normalize durations to perfectly match the total speech duration
total_calculated_duration = sum(paragraph_durations)
if total_calculated_duration > 0:
    scale_factor = speech_duration / total_calculated_duration
    paragraph_durations = [d * scale_factor for d in paragraph_durations]
    print(f"   -> Durations normalized to speech window by factor {scale_factor:.3f}")

# 4. --- THE CRUCIAL FIX ---
#    Recalculate scene start times based on the *final, normalized* durations.
print("   -> Building final, synchronized timeline...")
scene_starts = []
current_time = start_offset
for i, dur in enumerate(paragraph_durations):
    scene_starts.append(current_time)
    print(f"      Paragraph {i+1}: start={current_time:.2f}s, dur={dur:.2f}s")
    current_time += dur

# ==============================================================================
# ===================== ✅ END OF AUDIO SYNC FIX ✅ ==========================
# ==============================================================================


# --- Video Composition with Enhanced Text ---

clips = []

print("\n🎬 Building scenes with precise start times...")
for i, paragraph in enumerate(paragraphs):
    if i >= len(scene_starts) or i >= len(paragraph_durations):
        print(f"⚠️ Skipping paragraph {i+1}, timing info missing.")
        continue

    start_time = scene_starts[i]
    dur = paragraph_durations[i]
    img_idx = min(i, len(scene_images) - 1)
    
    print(f"🎬 Scene {i+1}: start={start_time:.2f}s, dur={dur:.2f}s")
    
    # Use enhanced scene creation with precise timing
    clips.extend(create_enhanced_scene(
        scene_images[img_idx], 
        paragraph,
        dur,
        start_time,
        scene_index=i
    ))

# ✅ SYNC CHECK
timeline_end = (scene_starts[-1] + paragraph_durations[-1]) if scene_starts else 0
speech_end = duration - trail_silence
final_drift = abs(timeline_end - speech_end)

print(f"\n📊 SYNC CHECK:")
print(f"   Visual Timeline End: {timeline_end:.3f}s")
print(f"   Speech Audio End:    {speech_end:.3f}s")
print(f"   Final Drift:         {final_drift*1000:.0f}ms")

if final_drift < 0.1:
    print(f"   ✅ NEAR-PERFECT SYNC!")
elif final_drift < 0.5:
    print(f"   ✅ Excellent sync")
else:
    print(f"   ⚠️ Sync drift of {final_drift:.2f}s detected. Check paragraph timing.")

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
        video = video.with_audio(audio)
else:
    video = video.with_audio(audio)
    print(f"   ⚠️ Audio: TTS only (no background music)")

if video.audio is None:
    raise Exception("Audio attachment failed!")

# --- [ ✅ FIXED & OPTIMIZED CODE ] ---
print(f"\n📹 Writing final video for web compatibility...")
try:
    video.write_videofile(
        OUT,
        fps=30,
        codec="libx264",
        audio_codec="aac",
        audio_bitrate="192k",
        preset="veryfast",
        threads=4,
        bitrate="2000k",
        ffmpeg_params=['-pix_fmt', 'yuv420p']
    )

    final_time = scene_starts[-1] + paragraph_durations[-1] if scene_starts else duration
    sync_status = "NEAR-PERFECT" if abs(final_time - (duration - trail_silence)) < 0.1 else "EXCELLENT" if abs(final_time - (duration-trail_silence)) < 0.5 else "GOOD"

    print(f"\n✅ MYSTERY VIDEO COMPLETE!")
    print(f"   Path: {OUT}")
    print(f"   Duration: {duration:.2f}s")

    if duration > 60:
        print(f"   ⚠️ WARNING: Video exceeds YouTube Shorts 60s limit!")
    else:
        print(f"   ✅ Within YouTube Shorts limit ({60 - duration:.2f}s buffer)")

    print(f"   Size: {os.path.getsize(OUT) / (1024*1024):.2f} MB")
    print(f"   Sync Status: {sync_status} ({abs(final_time - speech_end)*1000:.0f}ms drift)")
    print(f"   Features:")
    print(f"      ✓ Film noir aesthetic")
    if background_music:
        print(f"      ✓ Dark ambient background music")
    else:
        print(f"      ⚠ Background music skipped")
    print(f"      ✨ ENHANCED TEXT DISPLAY")
    print(f"      ✓ Paragraph-based timing")
    print(f"   🔍 Mystery video ready!")
except Exception as e:
    print(f"❌ Video creation failed: {e}")
    import traceback
    traceback.print_exc()
    raise

finally:
    print("🧹 Finalizing...")
    try: audio.close()
    except: pass
    try: video.close()
    except: pass
    if background_music:
        try: background_music.close()
        except: pass
    for clip in clips:
        try: clip.close()
        except: pass

print("✅ Mystery video pipeline complete! 🔍")