#!/usr/bin/env python3
"""
🔍 Generate Mystery TTS - PRODUCTION VERSION
Creates deep, mysterious voiceover optimized for documentary-style mystery narration

Features:
- Documentary narrator voices (p326, p287, p226)
- Slower, deliberate pacing (0.85x for suspense)
- Natural pauses for dramatic reveals
- Clear articulation for complex narratives
- Optimized for 60-90 second mystery stories
- FIXED: Proper paragraph timing synchronization
"""

import os
import json
from pathlib import Path
import subprocess   
import re

TMP = os.getenv("GITHUB_WORKSPACE", ".") + "/tmp"
os.makedirs(TMP, exist_ok=True)

# 🔍 MYSTERY TTS CONFIGURATION
PRIMARY_MODEL = "tts_models/en/vctk/vits"

# ✅ DOCUMENTARY NARRATOR SPEAKERS (Researched for mystery storytelling)
NARRATOR_SPEAKERS = {
    'p326': 'Deep authoritative male - Best for serious mysteries',
    'p287': 'Rich cinematic male - Best for dramatic reveals',
    'p226': 'Clear documentary male - Best for complex stories',
    'p259': 'Mature storyteller male - Backup option'
}

FALLBACK_MODELS = [
    "tts_models/en/ljspeech/tacotron2-DDC",
    "tts_models/en/ljspeech/glow-tts",
    "tts_models/en/ljspeech/speedy-speech"
]

# Speed settings for mystery content (PRD: 0.85 = 15% slower for suspense)
SPEED_SETTINGS = {
    'evening_prime': 0.85,     # Standard mystery pace (famous cases)
    'late_night': 0.82,        # Slightly slower (darker mysteries)
    'weekend_binge': 0.87,     # Slightly faster (complex narratives)
    'general': 0.85            # Default: documentary pace
}

# 🔍 CONTENT TYPE BASED SPEAKER SELECTION
SPEAKER_BY_CONTENT = {
    'evening_prime': 'p326',      # Deep authoritative (accessible mysteries)
    'late_night': 'p287',         # Rich cinematic (darker tone)
    'weekend_binge': 'p226',      # Clear documentary (complex stories)
    'general': 'p326'             # Default: authoritative
}


def clean_text_for_tts(text):
    """
    Clean text for TTS - remove special characters and formatting
    while preserving natural speech flow
    """
    # Store original for comparison
    original = text
    
    # Remove asterisks used for emphasis
    text = text.replace('*', '')
    
    # Remove other formatting characters
    text = text.replace('_', '')  # Remove underscores
    text = text.replace('~', '')  # Remove tildes
    text = text.replace('`', '')  # Remove backticks
    text = text.replace('^', '')  # Remove carets
    text = text.replace('#', '')  # Remove hashtags (unless you want "number")
    text = text.replace('|', '')  # Remove pipes
    text = text.replace('\\', '')  # Remove backslashes
    text = text.replace('PM', 'P M')  # Separate PM for clarity 
    text = text.replace('AM', 'A M')  # Separate AM for clarity
    
    # Clean up brackets that might contain meta information
    text = re.sub(r'\[.*?\]', '', text)  # Remove content in square brackets
    text = re.sub(r'\{.*?\}', '', text)  # Remove content in curly brackets
    
    # Clean up markdown-style links if any
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)  # Convert [text](url) to just text
    
    # Remove any HTML tags if present
    text = re.sub(r'<[^>]+>', '', text)
    
    # Clean up multiple spaces that result from removals
    text = re.sub(r' +', ' ', text)
    
    # Clean up multiple line breaks but preserve paragraph structure
    text = re.sub(r'\n\n+', '\n\n', text)
    
    # Remove leading/trailing whitespace from each line
    lines = text.split('\n')
    lines = [line.strip() for line in lines]
    text = '\n'.join(lines)
    
    # Remove empty lines that aren't paragraph breaks
    text = re.sub(r'\n(?!\n)', ' ', text)  # Single newlines to spaces
    text = re.sub(r' +', ' ', text)  # Clean up multiple spaces again
    
    return text.strip()


def load_script():
    """Load the generated mystery script"""
    script_path = os.path.join(TMP, "script.json")
    
    if not os.path.exists(script_path):
        print("❌ Script file not found!")
        exit(1)
    
    with open(script_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def build_tts_text_for_mystery(script_data):
    """
    Build TTS text for mystery narrative
    
    ✅ FIXED: Ensure paragraph splitting matches video creation script exactly
    """
    
    content_type = script_data.get('content_type', 'general')
    mystery_category = script_data.get('mystery_category', 'disappearance')
    
    print(f"🔍 Building TTS for {content_type} mystery ({mystery_category})")
    
    # Get the script text
    full_script = script_data.get('script', '')
    
    if not full_script:
        print("⚠️ No script field found, trying fallback...")
        hook = script_data.get('hook', '')
        bullets = script_data.get('bullets', [])
        cta = script_data.get('cta', '')
        full_script = f"{hook}\n\n{' '.join(bullets)}\n\n{cta}"
    
    # Clean text for TTS - Remove special characters
    full_script_clean = clean_text_for_tts(full_script)
    
    # Show what was cleaned
    if full_script != full_script_clean:
        print("🧹 Text cleaned for TTS (removed special characters)")
        removed_chars = set(re.findall(r'[*_~`^#\[\]\{\}|\\]', full_script))
        if removed_chars:
            print(f"   Removed characters: {', '.join(removed_chars)}")
    
    # ✅ CRITICAL FIX: Match the exact paragraph splitting logic from create_video.py
    # Split by double newline first
    paragraphs = [p.strip() for p in full_script_clean.split('\n\n') if p.strip()]
    
    # If we get too few paragraphs, try more aggressive splitting
    if len(paragraphs) <= 2:
        print(f"⚠️ Only {len(paragraphs)} paragraphs detected, attempting better split...")
        
        # Try splitting by single newline if they're substantial
        alt_paragraphs = [p.strip() for p in full_script_clean.split('\n') if p.strip() and len(p) > 50]
        
        if len(alt_paragraphs) > len(paragraphs):
            paragraphs = alt_paragraphs
            print(f"   ✅ Found {len(paragraphs)} paragraphs using single newline split")
        else:
            # Try splitting by sentences into logical chunks
            sentences = re.split(r'(?<=[.!?])\s+', full_script_clean)
            
            # Group sentences into paragraphs of roughly equal size
            if len(sentences) >= 4:
                target_para_count = min(6, max(4, len(sentences) // 3))
                sentences_per_para = len(sentences) // target_para_count
                
                paragraphs = []
                for i in range(0, len(sentences), sentences_per_para):
                    para = ' '.join(sentences[i:i+sentences_per_para])
                    if para.strip():
                        paragraphs.append(para.strip())
                
                print(f"   ✅ Created {len(paragraphs)} paragraphs from {len(sentences)} sentences")
    
    # Ensure we have at least 3 paragraphs for proper video structure
    if len(paragraphs) < 3 and len(full_script_clean) > 200:
        print("⚠️ Still too few paragraphs, forcing split...")
        words = full_script_clean.split()
        words_per_para = len(words) // 4  # Target 4 paragraphs
        
        paragraphs = []
        for i in range(0, len(words), words_per_para):
            para = ' '.join(words[i:i+words_per_para])
            if para.strip():
                paragraphs.append(para.strip())
        
        print(f"   ✅ Force-created {len(paragraphs)} paragraphs")
    
    print(f"📝 Mystery Narrative Structure:")
    print(f"   Total paragraphs: {len(paragraphs)}")
    print(f"   Total words: {len(full_script_clean.split())}")
    
    # Preview each section
    for i, para in enumerate(paragraphs[:8], 1):
        preview = para[:60] + "..." if len(para) > 60 else para
        word_count = len(para.split())
        print(f"   Section {i}: {word_count} words - {preview}")
    
    # Calculate expected duration
    word_count = len(full_script_clean.split())
    
    # Mystery pacing: slower for suspense
    base_wpm = {
        'evening_prime': 140,
        'late_night': 130,
        'weekend_binge': 145,
        'general': 140
    }
    
    wpm = base_wpm.get(content_type, 140)
    
    # Natural pauses from paragraph breaks
    pause_time = (len(paragraphs) - 1) * 0.8
    word_time = (word_count / wpm) * 60
    estimated_duration = word_time + pause_time
    
    print(f"\n⏱️ Duration Estimate:")
    print(f"   Words: {word_count}")
    print(f"   Target WPM: {wpm}")
    print(f"   Paragraph pauses: {pause_time:.1f}s")
    print(f"   Estimated total: {estimated_duration:.1f}s")
    
    if estimated_duration < 55:
        print(f"   ⚠️ Script may be too short (under 60s)")
    elif estimated_duration > 95:
        print(f"   ⚠️ Script may be too long (over 90s)")
    else:
        print(f"   ✅ Duration within target range (60-90s)")
    
    return full_script_clean, paragraphs, estimated_duration


def generate_audio_coqui(text, output_path, speaker_id, speed=0.85):
    """
    Generate audio using Coqui TTS with mystery narrator voice
    Optimized for documentary-style storytelling
    """
    try:
        from TTS.api import TTS
        
        print(f"\n🔊 Loading Coqui TTS model: {PRIMARY_MODEL}")
        print(f"   🎤 Target speaker: {speaker_id} ({NARRATOR_SPEAKERS.get(speaker_id, 'Unknown')})")
        print(f"   ⚡ Speed: {speed}x (slower for dramatic suspense)")
        
        # ✅ Clean text before sending to TTS
        text_clean = clean_text_for_tts(text)
        
        tts = TTS(model_name=PRIMARY_MODEL, progress_bar=False)
        
        # ✅ Check if model supports multiple speakers
        has_speakers = hasattr(tts, 'speakers') and tts.speakers is not None
        
        if has_speakers:
            print(f"   📢 Multi-speaker model detected")
            print(f"   🎭 Available speakers: {len(tts.speakers)}")
            
            # Verify speaker exists (case-insensitive check)
            available_speakers = [str(s).lower() for s in tts.speakers]
            speaker_lower = speaker_id.lower()
            
            if speaker_lower not in available_speakers:
                print(f"   ⚠️ Speaker '{speaker_id}' not in model")
                print(f"   Available speakers: {list(tts.speakers)[:10]}")
                
                # Try to find best narrator alternative
                for alt_speaker in NARRATOR_SPEAKERS.keys():
                    if alt_speaker.lower() in available_speakers:
                        speaker_id = alt_speaker
                        print(f"   ✅ Using alternative narrator: {speaker_id}")
                        break
                else:
                    # Use first available speaker as last resort
                    speaker_id = str(tts.speakers[0])
                    print(f"   🔄 Using first available: {speaker_id}")
            else:
                print(f"   ✅ Speaker '{speaker_id}' confirmed available")
            
            # Generate with speaker parameter
            print(f"   🎙️ Generating with speaker={speaker_id}, speed={speed}")
            tts.tts_to_file(
                text=text_clean,
                speaker=speaker_id,
                file_path=output_path,
                speed=speed
            )
        else:
            print(f"   📢 Single-speaker model (no speaker selection)")
            # Generate without speaker parameter
            tts.tts_to_file(
                text=text_clean,
                file_path=output_path,
                speed=speed
            )
        
        # Verify output
        if os.path.exists(output_path) and os.path.getsize(output_path) > 1000:
            file_size = os.path.getsize(output_path) / 1024
            print(f"   ✅ Audio generated: {file_size:.1f} KB")
            return True
        else:
            print(f"   ⚠️ Output file invalid or too small")
            return False
        
    except Exception as e:
        print(f"⚠️ Coqui TTS failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def generate_audio_espeak(text, output_path, speed_factor=0.85):
    """
    Fallback: Generate using espeak with mystery narrator settings
    Documentary-style voice optimized for storytelling
    """
    
    print(f"\n🔊 Using espeak fallback...")
    
    content_type = os.getenv('CONTENT_TYPE', 'general')
    
    # espeak speed (words per minute for documentary narration)
    base_speed = 140
    speed = int(base_speed * speed_factor)
    
    # Deeper pitch for authority but not too deep (documentary narrator)
    pitch = 30  # Moderate depth (range 0-99, default 50)
    
    # Pauses between words for clear articulation
    gap = 15  # milliseconds between words (less than motivation)
    
    print(f"   Speed: {speed} WPM (documentary pace)")
    print(f"   Pitch: {pitch} (narrator depth)")
    print(f"   Gap: {gap}ms (clear articulation)")
    
    # ✅ CLEAN TEXT to be safe
    text_cleaned = clean_text_for_tts(text)
    
    # Mystery-specific pause handling
    # Respect paragraph breaks and natural punctuation
    text_with_pauses = text_cleaned.replace('\n\n', ' [[800]] ')  # 800ms for paragraph breaks
    text_with_pauses = text_with_pauses.replace('.', '. [[400]] ')  # 400ms after sentences
    text_with_pauses = text_with_pauses.replace('?', '? [[500]] ')  # 500ms after questions
    text_with_pauses = text_with_pauses.replace("But here's where it gets strange", 
                                                 " [[600]] But here's where it gets strange [[600]] ")
    
    # Generate WAV first
    wav_path = output_path.replace('.mp3', '_temp.wav')
    
    try:
        subprocess.run([
            'espeak-ng',
            '-v', 'en-us',
            '-s', str(speed),
            '-p', str(pitch),
            '-g', str(gap),
            '-a', '180',  # Amplitude
            '-w', wav_path,
            text_with_pauses
        ], check=True, capture_output=True)
        
        # Convert to MP3 with mystery-optimized audio processing
        # Less bass boost than motivation, more clarity for storytelling
        subprocess.run([
            'ffmpeg',
            '-i', wav_path,
            '-af', 'equalizer=f=100:t=q:w=1:g=2,dynaudnorm,acompressor=threshold=-20dB:ratio=3',  # Subtle bass, normalize
            '-codec:a', 'libmp3lame',
            '-b:a', '192k',
            '-y',
            output_path
        ], check=True, capture_output=True, stderr=subprocess.DEVNULL)
        
        # Clean up temp
        if os.path.exists(wav_path):
            os.remove(wav_path)
        
        print(f"   ✅ espeak generated with mystery optimizations")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"   ❌ espeak failed: {e}")
        return False


def generate_audio_with_fallback(full_text, output_path):
    """
    Try Coqui TTS first, then espeak fallback
    Optimized for mystery documentary narration
    """
    
    content_type = os.getenv('CONTENT_TYPE', 'general')
    mystery_type = os.getenv('MYSTERY_TYPE', 'auto')
    
    # ✅ Select narrator based on content type
    speaker_id = SPEAKER_BY_CONTENT.get(content_type, 'p326')
    
    # Get speed setting (0.85 for mystery)
    speed = SPEED_SETTINGS.get(content_type, 0.85)
    
    print(f"\n🎙️ Generating mystery voiceover...")
    print(f"   Content: {content_type}")
    print(f"   Mystery Type: {mystery_type}")
    print(f"   Narrator: {speaker_id} ({NARRATOR_SPEAKERS[speaker_id]})")
    print(f"   Speed: {speed}x (documentary pace)")
    print(f"   Style: Serious, mysterious, documentary narrator")
    
    # Try Coqui TTS (primary)
    success = generate_audio_coqui(full_text, output_path, speaker_id, speed)
    
    if success:
        print("   ✅ Primary TTS successful")
        return True
    
    # Try fallback models
    print(f"\n🔄 Trying fallback models...")
    
    try:
        from TTS.api import TTS
        
        # ✅ Clean text once for all fallback attempts
        text_clean = clean_text_for_tts(full_text)
        
        for fallback_model in FALLBACK_MODELS:
            try:
                print(f"   Trying: {fallback_model}")
                tts = TTS(model_name=fallback_model, progress_bar=False)
                
                tts.tts_to_file(
                    text=text_clean,
                    file_path=output_path
                )
                
                if os.path.exists(output_path) and os.path.getsize(output_path) > 1000:
                    print(f"   ✅ Fallback success: {fallback_model}")
                    return True
                    
            except Exception as e:
                print(f"   ⚠️ Failed: {str(e)[:50]}...")
                continue
        
    except ImportError:
        print("   ⚠️ Coqui TTS not available")
    
    # Final fallback: espeak
    print(f"\n🔄 Using espeak as final fallback...")
    return generate_audio_espeak(full_text, output_path, speed)


def optimize_audio_timing(audio_path, expected_duration, paragraphs):
    """
    ✅ ENHANCED: Optimize audio timing with silence detection and word-proportional distribution
    """
    
    try:
        # Get actual audio duration
        result = subprocess.run([
            'ffprobe',
            '-v', 'error',
            '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            audio_path
        ], capture_output=True, text=True, check=True)
        
        actual_duration = float(result.stdout.strip())
        
        # ✅ NEW: Detect silence at start/end using ffmpeg
        lead_silence = 0.0
        trail_silence = 0.0
        
        try:
            silence_cmd = [
                'ffmpeg', '-i', audio_path,
                '-af', 'silencedetect=noise=-40dB:d=0.1',
                '-f', 'null', '-'
            ]
            silence_result = subprocess.run(silence_cmd, capture_output=True, text=True, stderr=subprocess.STDOUT)
            
            silence_starts = []
            silence_ends = []
            
            for line in silence_result.stdout.split('\n'):
                if 'silence_start' in line:
                    match = re.search(r'silence_start: ([\d.]+)', line)
                    if match:
                        silence_starts.append(float(match.group(1)))
                elif 'silence_end' in line:
                    match = re.search(r'silence_end: ([\d.]+)', line)
                    if match:
                        silence_ends.append(float(match.group(1)))
            
            # Leading silence: if starts near 0
            if silence_ends and silence_starts and silence_starts[0] < 0.1:
                lead_silence = silence_ends[0]
            
            # Trailing silence: if ends near duration
            if silence_starts:
                last_silence_start = silence_starts[-1]
                if actual_duration - last_silence_start < 2.0:
                    trail_silence = actual_duration - last_silence_start
            
            print(f"🔧 Silence Detection: Lead={lead_silence:.3f}s, Trail={trail_silence:.3f}s")
            
        except Exception as e:
            print(f"⚠️ Silence detection failed: {e}, continuing without it")
        
        # Calculate speech window
        speech_start = lead_silence
        speech_duration = actual_duration - lead_silence - trail_silence
        
        print(f"\n⏱️ Enhanced Audio Timing Analysis:")
        print(f"   Total duration: {actual_duration:.2f}s")
        print(f"   Expected: {expected_duration:.2f}s")
        print(f"   Speech window: {speech_start:.2f}s to {speech_start + speech_duration:.2f}s")
        print(f"   Speech duration: {speech_duration:.2f}s")
        print(f"   Paragraphs: {len(paragraphs)}")
        
        # Create section timings
        if not paragraphs or len(paragraphs) == 0:
            print("❌ No paragraphs provided!")
            section_timings = [{
                'name': 'full_script',
                'text_preview': 'Full script',
                'start': 0.0,
                'duration': actual_duration,
                'end': actual_duration,
                'words': 0
            }]
        else:
            # ✅ ENHANCED: Word-proportional timing
            total_words = sum(len(p.split()) for p in paragraphs)
            
            if total_words == 0:
                total_words = 1
            
            current_time = speech_start
            section_timings = []
            
            for i, paragraph in enumerate(paragraphs):
                para_words = len(paragraph.split())
                
                # Calculate proportional duration based on word count
                if total_words > 0:
                    word_duration = (para_words / total_words) * speech_duration
                else:
                    word_duration = speech_duration / len(paragraphs)
                
                # Ensure minimum duration for readability (1.5s base + 0.15s per word)
                min_duration = 1.5 + (para_words * 0.15)
                
                # Use the larger of proportional or minimum
                section_duration = max(word_duration, min_duration)
                
                section_timings.append({
                    'name': f'paragraph_{i+1}',
                    'text_preview': paragraph[:60] + "..." if len(paragraph) > 60 else paragraph,
                    'start': current_time,
                    'duration': section_duration,
                    'end': current_time + section_duration,
                    'words': para_words
                })
                
                current_time += section_duration
            
            # ✅ Normalize to fit speech window EXACTLY
            if section_timings:
                total_timed = sum(s['duration'] for s in section_timings)
                if total_timed > 0 and abs(total_timed - speech_duration) > 0.1:
                    scale = speech_duration / total_timed
                    current_time = speech_start
                    for timing in section_timings:
                        timing['duration'] *= scale
                        timing['start'] = current_time
                        timing['end'] = current_time + timing['duration']
                        current_time = timing['end']
                    print(f"   ✅ Normalized by scale factor: {scale:.3f}")
        
        print(f"\n📊 Enhanced Section Timings ({len(section_timings)} sections):")
        for i, timing in enumerate(section_timings):
            if i < 5 or i >= len(section_timings) - 1:
                print(f"   {timing['name']}: {timing['start']:.2f}s - {timing['end']:.2f}s "
                      f"({timing['words']} words, {timing['duration']:.2f}s)")
        
        if len(section_timings) > 6:
            print(f"   ... ({len(section_timings) - 6} more sections)")
        
        # Validate
        if len(section_timings) == 0:
            print("❌ No sections created! Using fallback...")
            section_timings = [{
                'name': 'full_script',
                'text_preview': 'Full script',
                'start': 0.0,
                'duration': actual_duration,
                'end': actual_duration,
                'words': total_words if 'total_words' in locals() else 0
            }]
        
        # Save timing metadata
        timing_path = os.path.join(TMP, "audio_timing.json")
        with open(timing_path, 'w') as f:
            json.dump({
                'total_duration': actual_duration,
                'speech_start': speech_start,
                'speech_duration': speech_duration,
                'lead_silence': lead_silence,
                'trail_silence': trail_silence,
                'sections': section_timings,
                'paragraph_count': len(paragraphs),
                'section_count': len(section_timings),
                'optimized': True,
                'timing_method': 'enhanced_proportional',
                'niche': 'mystery'
            }, f, indent=2)
        
        print(f"\n✅ Enhanced timing optimization complete")
        print(f"   Method: Word-proportional with silence detection")
        print(f"   Sections: {len(section_timings)}")
        print(f"   Saved to: {timing_path}")
        
        return section_timings
        
    except Exception as e:
        print(f"⚠️ Timing optimization failed: {e}")
        import traceback
        traceback.print_exc()
        
        # Return basic fallback
        if 'actual_duration' in locals():
            duration = actual_duration
        else:
            duration = expected_duration
            
        return [{
            'name': 'full_script',
            'text_preview': 'Full script',
            'start': 0.0,
            'duration': duration,
            'end': duration,
            'words': len(' '.join(paragraphs).split()) if paragraphs else 0
        }]

def save_metadata(audio_path, script_data, full_text, estimated_duration):
    """Save audio metadata for video creation"""
    
    try:
        # Get actual duration
        result = subprocess.run([
            'ffprobe',
            '-v', 'error',
            '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            audio_path
        ], capture_output=True, text=True, check=True)
        
        duration = float(result.stdout.strip())
    except:
        duration = estimated_duration
    
    word_count = len(full_text.split())
    wpm = (word_count / duration) * 60 if duration > 0 else 0
    
    content_type = script_data.get('content_type', 'general')
    
    metadata = {
        'audio_duration': duration,
        'estimated_duration': estimated_duration,
        'word_count': word_count,
        'character_count': len(full_text),
        'wpm': round(wpm, 1),
        'paragraph_count': full_text.count('\n\n') + 1,
        'content_type': content_type,
        'mystery_category': script_data.get('mystery_category', 'unknown'),
        'narrator': SPEAKER_BY_CONTENT.get(content_type, 'p326'),
        'speed': SPEED_SETTINGS.get(content_type, 0.85),
        'model': PRIMARY_MODEL,
        'niche': 'mystery',
        'style': 'documentary_narrator',
        'optimized': True
    }
    
    metadata_path = os.path.join(TMP, "audio_metadata.json")
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n📊 Audio Metadata:")
    print(f"   Duration: {duration:.2f}s (target: 60-90s)")
    print(f"   Words: {word_count}")
    print(f"   WPM: {wpm:.1f} (documentary pace)")
    print(f"   Narrator: {metadata['narrator']}")
    print(f"   File size: {os.path.getsize(audio_path) / 1024:.1f} KB")
    print(f"   ✅ Metadata: {metadata_path}")
    
    # Validate duration
    if duration < 55:
        print(f"   ⚠️ Audio shorter than expected (under 60s)")
    elif duration > 95:
        print(f"   ⚠️ Audio longer than expected (over 90s)")
    else:
        print(f"   ✅ Duration within target range")


def main():
    """Main TTS generation function for mystery narration"""
    
    print("\n" + "="*70)
    print("🔍 GENERATING MYSTERY VOICEOVER")
    print("="*70)
    
    # Load mystery script
    script_data = load_script()
    
    print(f"📝 Mystery: {script_data['title']}")
    print(f"🎯 Content Type: {script_data.get('content_type', 'general')}")
    print(f"🎭 Category: {script_data.get('mystery_category', 'unknown')}")
    
    # Build TTS text with proper paragraph splitting
    full_text, paragraphs, estimated_duration = build_tts_text_for_mystery(script_data)
    
    print(f"\n🎙️ Preparing narrator for {len(full_text)} character script")
    print(f"   First 100 chars: {full_text[:100]}...")
    
    # Output path
    output_path = os.path.join(TMP, "voice.mp3")
    
    # Generate audio with documentary narrator voice
    success = generate_audio_with_fallback(full_text, output_path)
    
    if not success or not os.path.exists(output_path):
        print("\n❌ All TTS methods failed!")
        exit(1)
    
    # ✅ CRITICAL: Pass paragraphs for proper timing
    section_timings = optimize_audio_timing(output_path, estimated_duration, paragraphs)
    
    # Save metadata
    save_metadata(output_path, script_data, full_text, estimated_duration)
    
    print("\n" + "="*70)
    print("✅ MYSTERY VOICEOVER GENERATION COMPLETE!")
    print("="*70)
    print(f"Output: {output_path}")
    print(f"Sections: {len(section_timings)}")
    print(f"Style: Documentary narrator, mysterious, deliberate")
    print(f"Ready for film noir video creation 🔍")


if __name__ == '__main__':
    main()