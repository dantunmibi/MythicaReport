import os
import json
from datetime import datetime, timedelta
import pytz

def check_schedule():
    """
    Checks if the current time falls within a valid posting window based on a JSON schedule.
    This script is robust against minor GitHub Actions scheduler delays.
    """
    # Configuration
    schedule_file = 'config/posting_schedule.json'
    # Define a tolerance window in minutes. The job must have started within this window.
    # e.g., for a 19:00 job, this will match if the script runs between 19:00 and 19:14.
    TOLERANCE_MINUTES = 15

    # Handle manual override
    if os.getenv('IGNORE_SCHEDULE') == 'true':
        print("✅ Schedule check BYPASSED by user input (ignore_schedule: true).")
        set_github_output('true', 'manual', 'manual_dispatch')
        return

    # Load schedule and determine timezone
    try:
        with open(schedule_file, 'r') as f:
            schedule_data = json.load(f)['schedule']
        
        target_tz_name = schedule_data.get('timezone', 'UTC')
        target_tz = pytz.timezone(target_tz_name)
        weekly_schedule = schedule_data['weekly_schedule']
    except (FileNotFoundError, KeyError) as e:
        print(f"❌ Error: Could not load or parse schedule file '{schedule_file}'. Details: {e}")
        set_github_output('false')
        return

    # Get current time in the target timezone
    now = datetime.now(target_tz)
    current_day_name = now.strftime('%A')
    print(f"ℹ️ Checking schedule for: {now.strftime('%Y-%m-%d %H:%M:%S %Z')} (Day: {current_day_name})")

    if current_day_name not in weekly_schedule:
        print(f"ℹ️ No schedule defined for {current_day_name}.")
        set_github_output('false')
        return

    # --- The Robust Time Check ---
    for slot in weekly_schedule[current_day_name]:
        slot_time_str = slot['time']
        slot_hour, slot_minute = map(int, slot_time_str.split(':'))
        
        # Create a datetime object for the slot time on the current day
        slot_datetime = now.replace(hour=slot_hour, minute=slot_minute, second=0, microsecond=0)
        
        # Define the valid window: from the slot time up to the tolerance limit
        window_start = slot_datetime
        window_end = slot_datetime + timedelta(minutes=TOLERANCE_MINUTES)
        
        # Check if the current time falls within this window
        if window_start <= now < window_end:
            print(f"✅ Match found! Current time {now.strftime('%H:%M')} is within the '{slot_time_str}' window.")
            print(f"   -> Content Type: {slot['type']}, Priority: {slot['priority']}")
            set_github_output(
                should_post='true',
                priority=slot['priority'],
                content_type=slot['type'],
                current_time=now.strftime("%Y-%m-%d %H:%M %Z")
            )
            return

    print(f"ℹ️ No active posting window found at the current time.")
    set_github_output('false')

def set_github_output(should_post, priority='low', content_type='off_schedule', current_time='N/A'):
    """Writes outputs for subsequent GitHub Actions steps."""
    with open(os.environ['GITHUB_OUTPUT'], 'a') as f:
        f.write(f'should_post={should_post}\n')
        f.write(f'priority={priority}\n')
        f.write(f'content_type={content_type}\n')
        f.write(f'current_time={current_time}\n')

if __name__ == "__main__":
    check_schedule()