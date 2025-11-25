#!/usr/bin/env python3
"""
🎯 Optimal Scheduler v6.0 - Mythica Report
Validates posting windows with schedule awareness and mystery type detection.

v6.0 FEATURES:
- Reads posting_schedule.json v6.0 (7-day predictable schedule)
- Extracts mystery_type from schedule slot
- Passes category to script generation for end hook alignment
- Ensures end hook promises match actual posting day
"""

import os
import json
from datetime import datetime, timedelta
import pytz

def check_schedule():
    """
    🆕 v6.0: Enhanced schedule check with category awareness
    
    Checks if current time falls within valid posting window AND
    extracts the mystery_type for this slot to ensure end hooks match.
    
    This ensures end hook promises are TRUE:
    - "Dark history every Monday" only appears on Monday posts
    - "Medical mysteries every Wednesday" only on Wednesday posts
    """
    schedule_file = 'config/posting_schedule.json'
    TOLERANCE_MINUTES = int(os.getenv("SCHEDULE_TOLERANCE_MINUTES", "120"))

    # Handle manual override
    if os.getenv('IGNORE_SCHEDULE') == 'true':
        print("✅ Schedule check BYPASSED by user input (ignore_schedule: true).")
        print("⚠️ WARNING: Manual runs should specify MYSTERY_TYPE env var to ensure correct end hooks!")
        set_github_output(
            should_post='true',
            priority='manual',
            content_type='manual_dispatch',
            mystery_type='auto',  # Will be determined by script generator
            current_time=datetime.now(pytz.UTC).strftime("%Y-%m-%d %H:%M UTC")
        )
        return

    # Load schedule
    try:
        with open(schedule_file, 'r') as f:
            schedule_data = json.load(f)['schedule']
        
        target_tz_name = schedule_data.get('timezone', 'UTC')
        target_tz = pytz.timezone(target_tz_name)
        weekly_schedule = schedule_data['weekly_schedule']
        
        print(f"📅 Loaded schedule v{schedule_data.get('optimization_notes', {}).get('version', 'unknown')}")
        
    except (FileNotFoundError, KeyError) as e:
        print(f"❌ Error: Could not load or parse schedule file '{schedule_file}'. Details: {e}")
        set_github_output('false')
        return

    # Get current time in target timezone
    now = datetime.now(target_tz)
    current_day_name = now.strftime('%A')
    
    print(f"ℹ️ Current time: {now.strftime('%Y-%m-%d %H:%M:%S %Z')} (Day: {current_day_name})")
    print(f"ℹ️ Checking with tolerance window of ±{TOLERANCE_MINUTES} minutes")

    # Check today's schedule
    if current_day_name in weekly_schedule:
        match = check_day_schedule(now, weekly_schedule[current_day_name], now, TOLERANCE_MINUTES)
        if match:
            print(f"✅ Match found for today ({current_day_name})!")
            print(f"🎯 Mystery Type: {match.get('mystery_type', 'auto')}")
            print(f"🎬 End Hook Will Promise: '{get_end_hook_preview(match.get('mystery_type'), current_day_name)}'")
            
            set_github_output(
                should_post='true',
                priority=match['priority'],
                content_type=match['content_type'],
                mystery_type=match.get('mystery_type', 'auto'),  # 🆕 Pass category
                current_time=now.strftime("%Y-%m-%d %H:%M %Z")
            )
            return
    
    # Check yesterday's schedule for late-night slots
    yesterday = now - timedelta(days=1)
    yesterday_name = yesterday.strftime('%A')
    
    if yesterday_name in weekly_schedule:
        for slot in weekly_schedule[yesterday_name]:
            slot_hour, slot_minute = map(int, slot['time'].split(':'))
            if slot_hour >= 22:  # Only check late-night slots
                slot_datetime = yesterday.replace(hour=slot_hour, minute=slot_minute, second=0, microsecond=0)
                time_diff_minutes = abs((now - slot_datetime).total_seconds() / 60)
                
                if time_diff_minutes <= TOLERANCE_MINUTES:
                    print(f"✅ Match found for late-night slot from {yesterday_name}!")
                    print(f"   -> Scheduled time: {slot['time']} UTC ({yesterday_name})")
                    print(f"   -> Current time: {now.strftime('%H:%M')} UTC ({current_day_name})")
                    print(f"   -> Time difference: {time_diff_minutes:.1f} minutes")
                    print(f"   -> Mystery Type: {slot.get('mystery_type', 'auto')}")
                    print(f"🎬 End Hook Will Promise: '{get_end_hook_preview(slot.get('mystery_type'), yesterday_name)}'")
                    
                    set_github_output(
                        should_post='true',
                        priority=slot['priority'],
                        content_type=slot['content_type'],
                        mystery_type=slot.get('mystery_type', 'auto'),  # 🆕 Pass category
                        current_time=now.strftime("%Y-%m-%d %H:%M %Z")
                    )
                    return

    print(f"ℹ️ No active posting window found at the current time.")
    print(f"\n📅 NEXT SCHEDULED POSTS:")
    show_next_scheduled_slots(weekly_schedule, now, target_tz)
    
    set_github_output('false', current_time=now.strftime("%Y-%m-%d %H:%M %Z"))


def check_day_schedule(now, day_slots, reference_date, tolerance_minutes):
    """
    🆕 v6.0: Check if current time matches any slot AND extract mystery_type
    """
    for slot in day_slots:
        slot_time_str = slot['time']
        slot_hour, slot_minute = map(int, slot_time_str.split(':'))
        
        # Create datetime for slot
        slot_datetime = reference_date.replace(hour=slot_hour, minute=slot_minute, second=0, microsecond=0)
        
        # Calculate time difference
        time_diff_minutes = abs((now - slot_datetime).total_seconds() / 60)
        
        if time_diff_minutes <= tolerance_minutes:
            print(f"   -> Matched slot: {slot_time_str} UTC")
            print(f"   -> Actual time: {now.strftime('%H:%M')} UTC")
            print(f"   -> Time difference: {time_diff_minutes:.1f} minutes")
            print(f"   -> Content Type: {slot.get('content_type', slot.get('type', 'general'))}")
            print(f"   -> Priority: {slot['priority']}")
            
            # 🆕 v6.0: Ensure we return the mystery_type and content_type
            return {
                'priority': slot['priority'],
                'content_type': slot.get('content_type', slot.get('type', 'evening_prime')),
                'mystery_type': slot.get('mystery_type', slot.get('type', 'auto')),  # Fallback for old format
                'time': slot_time_str
            }
    
    return None


def get_end_hook_preview(mystery_type, day_name):
    """
    🆕 v6.0: Preview what end hook will be generated
    Helps verify schedule alignment
    """
    end_hook_previews = {
        'dark_history': f"The truth was buried. History forgot. Dark history every Monday.",
        'disturbing_medical': f"Doctors still can't explain it today. Medical mysteries every Wednesday.",
        'dark_experiments': f"The files remain classified. Secrets buried. Secret research every Thursday.",
        'disappearance': f"They were never found. No trace. Unsolved cases every {day_name}.",
        'crime': f"The case remains unsolved to today. True crime every Saturday.",
        'phenomena': f"No explanation exists. None makes sense. Strange phenomena every Sunday.",
    }
    
    return end_hook_previews.get(mystery_type, f"More mysteries every {day_name}.")


def show_next_scheduled_slots(weekly_schedule, current_time, target_tz):
    """
    🆕 v6.0: Show next 3 scheduled posts
    Helps debug scheduling issues
    """
    upcoming = []
    
    # Check next 7 days
    for day_offset in range(7):
        check_date = current_time + timedelta(days=day_offset)
        day_name = check_date.strftime('%A')
        
        if day_name in weekly_schedule:
            for slot in weekly_schedule[day_name]:
                slot_hour, slot_minute = map(int, slot['time'].split(':'))
                slot_datetime = check_date.replace(hour=slot_hour, minute=slot_minute, second=0, microsecond=0)
                
                if slot_datetime > current_time:
                    upcoming.append({
                        'datetime': slot_datetime,
                        'day': day_name,
                        'time': slot['time'],
                        'type': slot.get('mystery_type', slot.get('type', 'unknown'))
                    })
    
    # Sort by datetime and show next 3
    upcoming.sort(key=lambda x: x['datetime'])
    
    for i, slot in enumerate(upcoming[:3], 1):
        time_until = slot['datetime'] - current_time
        hours_until = int(time_until.total_seconds() / 3600)
        
        print(f"   {i}. {slot['day']} {slot['time']} UTC - {slot['type'].replace('_', ' ').title()} (in {hours_until}h)")


def set_github_output(should_post='false', priority='low', content_type='off_schedule', mystery_type='auto', current_time='N/A'):
    """
    🆕 v6.0: Enhanced output with mystery_type for end hook alignment
    """
    if "GITHUB_OUTPUT" in os.environ:
        with open(os.environ['GITHUB_OUTPUT'], 'a') as f:
            f.write(f'should_post={should_post}\n')
            f.write(f'priority={priority}\n')
            f.write(f'content_type={content_type}\n')
            f.write(f'mystery_type={mystery_type}\n')  # 🆕 NEW OUTPUT
            f.write(f'current_time={current_time}\n')
    else:
        print("--- GITHUB_OUTPUT (local run) ---")
        print(f"should_post={should_post}")
        print(f"priority={priority}")
        print(f"content_type={content_type}")
        print(f"mystery_type={mystery_type}")  # 🆕 NEW OUTPUT
        print(f"current_time={current_time}")


if __name__ == "__main__":
    print("🎯 Mythica Report v6.0 - Optimal Scheduler")
    print("   Features: Schedule-aware, category detection, end hook alignment")
    print("="*70)
    check_schedule()