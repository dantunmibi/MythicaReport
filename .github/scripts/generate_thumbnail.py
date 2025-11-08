#!/usr/bin/env python3
"""
🔍 Generate Mystery Thumbnail - ROBUST VERSION
Features:
- Dark noir aesthetic
- Typewriter/serif fonts
- Evidence tag overlays (UNSOLVED, CLASSIFIED)
- Film grain and vignette
- Minimal text (mystery name only)
- Documentary-style composition
- Vintage photograph aesthetic
"""

import os
import json
import requests
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
from io import BytesIO
import platform
from tenacity import retry, stop_after_attempt, wait_exponential
from time import sleep
import time
import textwrap
import random
import re
import numpy as np

TMP = os.getenv("GITHUB_WORKSPACE", ".") + "/tmp"

# 🔍 NOIR COLOR PALETTE (Dark & Mysterious)
NOIR_COLORS = {
    'deep_black': (15, 15, 20),
    'dark_slate': (30, 35, 45),
    'crimson_evidence': (185, 28, 28),
    'aged_gold': (212, 175, 55),
    'fog_gray': (80, 85, 90),
    'shadow_blue': (25, 35, 50),
    'vintage_tan': (200, 180, 150),
}


def get_font_path(size=60, serif=True):
    """Get serif/typewriter font for mystery aesthetic"""
    system = platform.system()
    font_paths = []
    
    if system == "Windows":
        font_paths = [
            "C:/Windows/Fonts/cour.ttf",      # Courier New (typewriter)
            "C:/Windows/Fonts/georgia.ttf",   # Georgia (serif)
            "C:/Windows/Fonts/times.ttf",     # Times New Roman
            "C:/Windows/Fonts/arial.ttf",     # Arial fallback
        ]
    elif system == "Darwin":
        font_paths = [
            "/System/Library/Fonts/Supplemental/Courier New.ttf",
            "/System/Library/Fonts/Supplemental/Georgia.ttf",
            "/Library/Fonts/Arial.ttf",
        ]
    else:
        font_paths = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ]
    
    for font_path in font_paths:
        if os.path.exists(font_path):
            try:
                return ImageFont.truetype(font_path, size)
            except Exception as e:
                print(f"⚠️ Could not load {font_path}: {e}")
    
    print("⚠️ Using default font")
    return ImageFont.load_default()


# Load script
with open(os.path.join(TMP, "script.json"), "r", encoding="utf-8") as f:
    data = json.load(f)

title = data.get("title", "MYSTERY")
topic = data.get("topic", "mystery")
hook = data.get("hook", "")
key_phrase = data.get("key_phrase", "")
mystery_category = data.get("mystery_category", "disappearance")
content_type = data.get("content_type", "evening_prime")

# 🔍 SMART TEXT SELECTION FOR MYSTERY (Extract mystery name)
display_text = None

# Priority 1: Key phrase (usually mystery name like "FLIGHT 19")
if key_phrase and len(key_phrase) > 3:
    display_text = key_phrase.upper()
    print(f"🔍 Using KEY PHRASE: {display_text}")

# Priority 2: Extract mystery name from title (before colon)
elif ':' in title:
    mystery_name = title.split(':')[0].strip()
    display_text = mystery_name.upper()
    print(f"🎯 Extracted from title: {display_text}")

# Priority 3: First 3-5 words of title
else:
    words = title.split()[:4]
    display_text = ' '.join(words).upper()
    print(f"📝 Using first words: {display_text}")

# Ensure it's not too long (mystery uses minimal text)
if len(display_text) > 25:
    # Try to shorten to just the main mystery name
    words = display_text.split()[:3]
    display_text = ' '.join(words)

print(f"📊 Final thumbnail text: {display_text}")
print(f"   Length: {len(display_text)} chars")
print(f"   Category: {mystery_category}")

# Canvas dimensions
w = 720
h = 1280

# Safe zones
SAFE_ZONE_MARGIN = 60
TEXT_MAX_WIDTH = w - (2 * SAFE_ZONE_MARGIN)


def generate_thumbnail_huggingface(prompt):
    """🔍 Generate noir mystery thumbnail using HuggingFace"""
    try:
        hf_token = os.getenv('HUGGINGFACE_API_KEY')
        if not hf_token:
            raise Exception("Missing HUGGINGFACE_API_KEY")

        headers = {"Authorization": f"Bearer {hf_token}"}
        
        # 🔍 Mystery noir negative prompt
        negative_mystery = (
            "blurry, low quality, watermark, text overlay, logo, ui, interface, "
            "happy, cheerful, smiling, bright, colorful, pastel, soft, modern, "
            "cartoon, anime, illustration, painting, 3d render, "
            "compression, pixelated, amateur, contemporary"
        )
        
        payload = {
            "inputs": f"{prompt}, film noir photography, black and white, high contrast, dramatic shadows, vintage 1940s aesthetic, mysterious, documentary style, no text",
            "parameters": {
                "negative_prompt": negative_mystery,
                "num_inference_steps": 4,
                "guidance_scale": 0.0,
                "width": 720,
                "height": 1280,
            }
        }

        models = [
            "black-forest-labs/FLUX.1-schnell",
            "stabilityai/stable-diffusion-xl-base-1.0",
            "black-forest-labs/FLUX.1-dev"
        ]

        for model in models:
            url = f"https://router.huggingface.co/hf-inference/models/{model}"
            print(f"🤗 Trying {model}")

            response = requests.post(url, headers=headers, json=payload, timeout=120)

            if response.status_code == 200 and len(response.content) > 1000:
                print(f"✅ {model} succeeded")
                return response.content
            elif response.status_code == 402:
                print(f"💰 {model} requires payment")
                continue
            elif response.status_code in [503, 429]:
                print(f"⌛ {model} loading/rate-limited")
                time.sleep(2)
                continue

        raise Exception("All HuggingFace models failed")

    except Exception as e:
        print(f"⚠️ HuggingFace failed: {e}")
        raise


def generate_thumbnail_pollinations(prompt):
    """🔍 Pollinations backup with noir aesthetic"""
    try:
        negative = (
            "text, logo, watermark, ui, interface, happy, cheerful, "
            "bright, colorful, cartoon, anime, modern, soft lighting"
        )

        enhanced = (
            f"{prompt}, film noir photography, black and white, "
            "high contrast, dramatic shadows, vintage aesthetic, "
            "mysterious, documentary style, no text"
        )

        seed = random.randint(1, 999999)
        url = (
            f"https://image.pollinations.ai/prompt/{requests.utils.quote(enhanced)}"
            f"?width=720&height=1280&negative={requests.utils.quote(negative)}"
            f"&nologo=true&notext=true&enhance=true&model=flux&seed={seed}"
        )

        print(f"🌐 Pollinations (seed={seed})")
        response = requests.get(url, timeout=120)

        if response.status_code == 200:
            print(f"✅ Pollinations succeeded")
            return response.content

        raise Exception(f"Pollinations failed: {response.status_code}")

    except Exception as e:
        print(f"⚠️ Pollinations failed: {e}")
        raise


def generate_mystery_fallback(bg_path, mystery_category):
    """🔍 Mystery-specific fallback with curated noir photos"""
    
    # Mystery category-specific keywords
    keywords_map = {
        'disappearance': ['abandoned', 'empty', 'fog', 'mystery', 'dark-sky'],
        'crime': ['noir', 'detective', 'shadow', 'investigation', 'urban-night'],
        'historical': ['vintage', 'old-photo', 'ancient', 'archive', 'historical'],
        'conspiracy': ['documents', 'classified', 'secret', 'mystery', 'evidence']
    }
    
    keywords = keywords_map.get(mystery_category, keywords_map['disappearance'])
    keyword = random.choice(keywords)
    
    print(f"🔎 Searching mystery image for '{mystery_category}' (keyword: '{keyword}')...")

    # Try Unsplash
    try:
        seed = random.randint(1, 9999)
        url = f"https://source.unsplash.com/720x1280/?{requests.utils.quote(keyword)}&sig={seed}"
        print(f"🖼️ Unsplash: '{keyword}' (seed={seed})")
        response = requests.get(url, timeout=30, allow_redirects=True)
        
        if response.status_code == 200:
            with open(bg_path, "wb") as f:
                f.write(response.content)
            print(f"✅ Unsplash succeeded")
            return bg_path
            
    except Exception as e:
        print(f"⚠️ Unsplash error: {e}")

    # 🔍 Curated Pexels mystery/noir photos
    try:
        print("📸 Trying curated Pexels noir photos...")
        
        mystery_pexels = {
            'disappearance': [
                3783471, 3617500, 3861431,  # Abandoned/empty
                2310713, 1363876, 1363875   # Fog/mystery
            ],
            'crime': [
                2582937, 2559941, 2566581,  # Noir/detective
                3617500, 1363876, 2566590   # Dark urban
            ],
            'historical': [
                1418595, 1141853, 1303081,  # Vintage
                2325729, 1907785, 2832382   # Historical
            ],
            'conspiracy': [
                7319070, 7319311, 5668826,  # Documents
                3617500, 5668838, 7319316   # Classified aesthetic
            ]
        }
        
        photos = mystery_pexels.get(mystery_category, mystery_pexels['disappearance'])
        photo_id = random.choice(photos)
        
        url = f"https://images.pexels.com/photos/{photo_id}/pexels-photo-{photo_id}.jpeg?auto=compress&cs=tinysrgb&w=720&h=1280&fit=crop"
        print(f"📸 Pexels photo {photo_id}")
        
        response = requests.get(url, timeout=30)
        if response.status_code == 200:
            with open(bg_path, "wb") as f:
                f.write(response.content)
            
            img = Image.open(bg_path).convert("RGB")
            img = img.resize((720, 1280), Image.LANCZOS)
            img.save(bg_path, quality=95)
            print(f"✅ Pexels succeeded")
            return bg_path
            
    except Exception as e:
        print(f"⚠️ Pexels error: {e}")

    # Picsum fallback (grayscale for noir)
    try:
        seed = random.randint(1, 1000)
        url = f"https://picsum.photos/720/1280?random={seed}&grayscale"
        print(f"🎲 Picsum (seed={seed})")
        response = requests.get(url, timeout=30, allow_redirects=True)
        
        if response.status_code == 200:
            with open(bg_path, "wb") as f:
                f.write(response.content)
            return bg_path
            
    except Exception as e:
        print(f"⚠️ Picsum failed: {e}")

    return None


@retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=2, min=4, max=20))
def generate_thumbnail_bg(mystery_category, content_type):
    """🔍 Generate noir mystery thumbnail background"""
    bg_path = os.path.join(TMP, "thumb_bg.png")
    
    # Mystery-specific prompts
    category_prompts = {
        'disappearance': "abandoned location, empty space, fog, eerie atmosphere, noir photography",
        'crime': "crime scene, investigation, detective noir, shadowy urban, dark alley",
        'historical': "vintage photograph, ancient artifact, aged document, historical",
        'conspiracy': "classified documents, secret files, evidence, mysterious papers"
    }
    
    base_prompt = category_prompts.get(mystery_category, category_prompts['disappearance'])
    prompt = f"{base_prompt}, film noir, black and white photography, high contrast, dramatic shadows, vintage 1940s aesthetic, mysterious, documentary style, no text, seed={random.randint(1000,9999)}"
    
    # Try AI providers
    providers = [
        ("Pollinations", generate_thumbnail_pollinations),
        ("HuggingFace", generate_thumbnail_huggingface)
    ]
    
    for name, func in providers:
        try:
            print(f"🎨 Trying {name}")
            content = func(prompt)
            with open(bg_path, "wb") as f:
                f.write(content)
            
            if os.path.getsize(bg_path) > 1000:
                print(f"✅ {name} succeeded")
                return bg_path
                
        except:
            continue

    print("🖼️ AI failed, trying curated noir photos")
    result = generate_mystery_fallback(bg_path, mystery_category)
    
    if result and os.path.exists(bg_path):
        return bg_path
    
    # 🔍 Dark noir gradient fallback
    print("⚠️ All failed, creating noir gradient")
    return create_noir_gradient(bg_path, mystery_category)


def create_noir_gradient(bg_path, mystery_category):
    """Create dark noir gradient background"""
    img = Image.new("RGB", (720, 1280), (0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    # Mystery category-specific gradients
    color_schemes = {
        'disappearance': [NOIR_COLORS['deep_black'], NOIR_COLORS['fog_gray']],
        'crime': [NOIR_COLORS['deep_black'], NOIR_COLORS['crimson_evidence']],
        'historical': [NOIR_COLORS['shadow_blue'], NOIR_COLORS['vintage_tan']],
        'conspiracy': [NOIR_COLORS['dark_slate'], NOIR_COLORS['aged_gold']]
    }
    
    colors = color_schemes.get(mystery_category, color_schemes['disappearance'])
    
    for y in range(1280):
        ratio = y / 1280
        r = int(colors[0][0] + (colors[1][0] - colors[0][0]) * ratio)
        g = int(colors[0][1] + (colors[1][1] - colors[0][1]) * ratio)
        b = int(colors[0][2] + (colors[1][2] - colors[0][2]) * ratio)
        draw.line([(0, y), (720, y)], fill=(r, g, b))
    
    # Add film grain
    try:
        arr = np.array(img)
        noise = np.random.normal(0, 20, arr.shape).astype(np.int16)
        arr = np.clip(arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)
    except:
        pass
    
    img.save(bg_path, quality=95)
    return bg_path

def compress_thumbnail_under_limit(image, output_path, max_kb=2000):
    """Compress thumbnail to under 2MB limit"""
    # Apply softening filter
    image = image.filter(ImageFilter.SMOOTH)
    
    # Try quality levels from 90 down to 80
    for quality in [90, 85, 80]:
        image.save(output_path, "PNG", quality=quality, optimize=True)
        size_kb = os.path.getsize(output_path) / 1024
        
        if size_kb <= max_kb:
            print(f"   ✅ Compressed to {size_kb:.1f} KB (quality={quality})")
            return True
    
    # If still too large, convert to JPEG
    print(f"   ⚠️ PNG still too large, converting to JPEG...")
    rgb_image = image.convert("RGB")
    jpeg_path = output_path.replace('.png', '.jpg')
    rgb_image.save(jpeg_path, "JPEG", quality=85, optimize=True)
    
    # Rename back to .png (YouTube accepts either)
    os.remove(output_path)
    os.rename(jpeg_path, output_path)
    
    size_kb = os.path.getsize(output_path) / 1024
    print(f"   ✅ JPEG: {size_kb:.1f} KB")
    return True


# Generate background
print("🔍 Generating mystery thumbnail background...")
bg_path = generate_thumbnail_bg(mystery_category, content_type)
img = Image.open(bg_path).convert("RGB")

# Ensure dimensions
if img.size != (720, 1280):
    img = img.resize((720, 1280), Image.LANCZOS)

# 🔍 NOIR ENHANCEMENT (desaturate, high contrast, dark)
# Desaturate heavily (30% color remaining)
enhancer = ImageEnhance.Color(img)
img = enhancer.enhance(0.30)

# High contrast (noir look)
enhancer = ImageEnhance.Contrast(img)
img = enhancer.enhance(1.70)

# Darker overall
enhancer = ImageEnhance.Brightness(img)
img = enhancer.enhance(0.65)

img = img.convert("RGBA")

# 🔍 HEAVY NOIR VIGNETTE (darker edges)
vignette = Image.new("RGBA", img.size, (0, 0, 0, 0))
vd = ImageDraw.Draw(vignette)

center_x, center_y = w // 2, h // 2
max_radius = int((w**2 + h**2)**0.5) // 2

for radius in range(0, max_radius, 15):
    alpha = int(160 * (radius / max_radius))  # Heavy vignette
    vd.ellipse(
        [center_x - radius, center_y - radius, center_x + radius, center_y + radius],
        outline=(0, 0, 0, alpha),
        width=40
    )

img = Image.alpha_composite(img, vignette)

# 🔍 ADD FILM GRAIN (vintage noir feel)
try:
    img_rgb = img.convert("RGB")
    arr = np.array(img_rgb)
    noise = np.random.normal(0, 15, arr.shape).astype(np.int16)
    arr = np.clip(arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr).convert("RGBA")
except:
    pass

draw = ImageDraw.Draw(img)

# 🔍 ADD "UNSOLVED" EVIDENCE STAMP (top-right)
stamp_font = get_font_path(size=45, serif=True)
stamp_text = "UNSOLVED"

# Get stamp text size
dummy_img = Image.new('RGB', (1, 1))
dummy_draw = ImageDraw.Draw(dummy_img)
stamp_bbox = dummy_draw.textbbox((0, 0), stamp_text, font=stamp_font)
stamp_width = stamp_bbox[2] - stamp_bbox[0]
stamp_height = stamp_bbox[3] - stamp_bbox[1]

# Position stamp
stamp_x = w - stamp_width - 80
stamp_y = 60

# Draw red stamp background
stamp_bg = Image.new("RGBA", img.size, (0, 0, 0, 0))
stamp_bg_draw = ImageDraw.Draw(stamp_bg)

# Red rectangle with border
stamp_padding = 15
stamp_bg_draw.rectangle(
    [
        stamp_x - stamp_padding, 
        stamp_y - stamp_padding,
        stamp_x + stamp_width + stamp_padding,
        stamp_y + stamp_height + stamp_padding
    ],
    fill=(185, 28, 28, 220),  # Crimson with transparency
    outline=(255, 255, 255, 255),
    width=3
)

img = Image.alpha_composite(img, stamp_bg)

# Draw stamp text
stamp_overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
stamp_draw = ImageDraw.Draw(stamp_overlay)
stamp_draw.text((stamp_x, stamp_y), stamp_text, font=stamp_font, fill=(255, 255, 255, 255))

img = Image.alpha_composite(img, stamp_overlay)

draw = ImageDraw.Draw(img)

# Smart text wrapping (minimal use for mystery)
def smart_text_wrap(text, font_obj, max_width, draw_obj):
    """Wrap text without splitting words"""
    words = text.split()
    lines = []
    current_line = []
    
    for word in words:
        test_line = ' '.join(current_line + [word])
        bbox = draw_obj.textbbox((0, 0), test_line, font=font_obj)
        text_width = bbox[2] - bbox[0]
        
        if text_width <= max_width:
            current_line.append(word)
        else:
            if current_line:
                lines.append(' '.join(current_line))
            current_line = [word]
    
    if current_line:
        lines.append(' '.join(current_line))
    
    return lines


# Find optimal font size (smaller for mystery, minimal text)
font_size = 70  # Start smaller than motivation
min_font_size = 40
max_height = h * 0.25  # Less vertical space (minimal text)
text_lines = []

print(f"🎯 Finding optimal font size for mystery text...")

while font_size >= min_font_size:
    test_font = get_font_path(font_size, serif=True)
    wrapped_lines = smart_text_wrap(display_text, test_font, TEXT_MAX_WIDTH, dummy_draw)
    
    total_height = 0
    max_line_width = 0
    
    for line in wrapped_lines:
        bbox = dummy_draw.textbbox((0, 0), line, font=test_font)
        line_width = bbox[2] - bbox[0]
        line_height = bbox[3] - bbox[1]
        total_height += line_height
        max_line_width = max(max_line_width, line_width)
    
    if len(wrapped_lines) > 1:
        total_height += (len(wrapped_lines) - 1) * 20
    
    if total_height <= max_height and max_line_width <= TEXT_MAX_WIDTH:
        text_lines = wrapped_lines
        print(f"✅ Font {font_size}px: {len(wrapped_lines)} lines")
        break
    
    font_size -= 5

if not text_lines:
    font_size = min_font_size
    test_font = get_font_path(font_size, serif=True)
    text_lines = smart_text_wrap(display_text, test_font, TEXT_MAX_WIDTH, dummy_draw)

main_font = get_font_path(font_size, serif=True)
print(f"📝 Final font: {font_size}px (serif) for {len(text_lines)} lines")

# Position text (bottom for mystery - documentary subtitle style)
line_spacing = 20
total_text_height = sum([
    dummy_draw.textbbox((0, 0), line, font=main_font)[3] - 
    dummy_draw.textbbox((0, 0), line, font=main_font)[1] 
    for line in text_lines
]) + (len(text_lines) - 1) * line_spacing

# Bottom positioning (documentary subtitle style)
start_y = h - total_text_height - SAFE_ZONE_MARGIN - 120
start_y = max(SAFE_ZONE_MARGIN + 150, start_y)  # Don't overlap stamp

current_y = start_y

# 🔍 SUBTLE TEXT RENDERING (documentary style, not aggressive)
for i, line in enumerate(text_lines):
    bbox = draw.textbbox((0, 0), line, font=main_font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    
    x = (w - text_w) // 2
    y = current_y
    
    # Clamp to safe zones
    x = max(SAFE_ZONE_MARGIN, min(x, w - SAFE_ZONE_MARGIN - text_w))
    
    # 🔍 MODERATE SHADOW (subtle, not overpowering)
    shadow_overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    sd = ImageDraw.Draw(shadow_overlay)
    
    for offset in [5, 3, 1]:
        shadow_alpha = int(180 * (offset / 5))
        sd.text((x + offset, y + offset), line, font=main_font, fill=(0, 0, 0, shadow_alpha))
    
    img = Image.alpha_composite(img, shadow_overlay)
    
    # 🔍 STROKE (moderate thickness - documentary style)
    stroke_overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    so = ImageDraw.Draw(stroke_overlay)
    
    stroke_width = 4  # Moderate (vs 6 for motivation)
    for sx in range(-stroke_width, stroke_width + 1):
        for sy in range(-stroke_width, stroke_width + 1):
            if sx != 0 or sy != 0:
                so.text((x + sx, y + sy), line, font=main_font, fill=(0, 0, 0, 255))
    
    img = Image.alpha_composite(img, stroke_overlay)
    
    # 🔍 OFF-WHITE TEXT (not pure white - more vintage)
    text_overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    to = ImageDraw.Draw(text_overlay)
    to.text((x, y), line, font=main_font, fill=(232, 232, 232, 255))  # Off-white
    img = Image.alpha_composite(img, text_overlay)
    
    current_y += text_h + line_spacing

# Save thumbnail
thumb_path = os.path.join(TMP, "thumbnail.png")
final_img = img.convert("RGB")

if final_img.size != (720, 1280):
    final_img = final_img.resize((720, 1280), Image.LANCZOS)

# 🔍 SUBTLE SHARPEN (not overly sharp - vintage feel)
final_img = final_img.filter(ImageFilter.SHARPEN)

# Compress to under 2MB
compress_thumbnail_under_limit(final_img, thumb_path, max_kb=2000)

saved_img = Image.open(thumb_path)

print(f"\n✅ MYSTERY THUMBNAIL COMPLETE")
print(f"   Path: {thumb_path}")
print(f"   Size: {os.path.getsize(thumb_path) / 1024:.1f} KB")
print(f"   Dimensions: {saved_img.size}")
print(f"   Text: {text_lines}")
print(f"   Font: {font_size}px (serif/typewriter)")
print(f"   Stamp: {stamp_text}")
print(f"   Category: {mystery_category}")
print(f"   Style: Film noir, documentary")
print(f"   🔍 MYSTERY THUMBNAIL READY!")