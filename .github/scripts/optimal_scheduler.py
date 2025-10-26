#!/usr/bin/env python3
"""
🔮 Mythica Report - Optimal Posting Scheduler
Ensures a consistent and varied feed of mystery stories by rotating through content pillars at peak engagement times.
"""

import os
import json
from datetime import datetime, timedelta
import pytz

# Mystery content performs best with consistency and variety.
# Schedule focuses on primetime viewing hours (EST).
OPTIMAL_SCHEDULE = {
    # Monday: Start the week with a classic
    0: {
        "times": [20],  # 8 PM EST
        "content_types": ["unsolved_true_crime"],
        "priority": ["high"]
    },
    # Tuesday: Spooky Tuesday
    1: {
        "times": [21],  # 9 PM EST
        "content_types": ["paranormal_hauntings"],
        "priority": ["high"]
    },
    # Wednesday: Mid-week historical deep dive
    2: {
        "times": [20],  # 8 PM EST
        "content_types": ["historical_mysteries"],
        "priority": ["high"]
    },
    # Thursday: Look to the stars
    3: {
        "times": [21],  # 9 PM EST
        "content_types": ["cosmic_and_sci_fi"],
        "priority": ["highest"] # High engagement topic
    },
    # Friday: Double feature for the weekend kickoff
    4: {
        "times": [15, 21],  # 3 PM (after school/work), 9 PM (primetime)
        "content_types": ["internet_and_modern", "unsolved_true_crime"],
        "priority": ["medium", "highest"]
    },
    # Saturday: Binge day - triple feature
    5: {
        "times": [12, 16, 21],  # Noon, 4 PM, 9 PM
        "content_types": ["high_strangeness", "historical_mysteries", "paranormal_hauntings"],
        "priority": ["medium", "high", "highest"]
    },
    # Sunday: Wind-down with a compelling story
    6: {
        "times": [19],  # 7 PM EST
        "content_types": ["unsolved_true_crime"], # High engagement to start the week's retention
        "priority": ["highest"]
    }
}

# Content pillars for Mythica Report, ensuring a diverse feed.
CONTENT_PILLARS = {
    "unsolved_true_crime": {
        "percentage": 25,
        "description": "Classic unsolved cases, strange disappearances, and modern true crime with a mysterious twist.",
        "narrative_style": "investigative_and_tense",
        "visual_style": "archival_photos, case_files, maps, dark_tones"
    },
    "paranormal_hauntings": {
        "percentage": 20,
        "description": "Ghost stories, hauntings, poltergeists, and encounters with the supernatural.",
        "narrative_style": "chilling_and_atmospheric",
        "visual_style": "creepy_locations, pov_footage, old_photos, ghostly_effects"
    },
    "historical_mysteries": {
        "percentage": 15,
        "description": "Mysteries from the ancient world to the 20th century, like lost colonies, strange artifacts, and historical figures.",
        "narrative_style": "documentary_and_intriguing",
        "visual_style": "old_paintings, historical_documents, ruins, period_reenactments"
    },
    "cosmic_and_sci_fi": {
        "percentage": 15,
        "description": "UFO encounters, strange signals from space (Wow! Signal), alien abduction cases, and other sci-fi-like mysteries.",
        "narrative_style": "awe_and_suspense",
        "visual_style": "space, stars, radar_screens, cgi_spacecraft, witness_sketches"
    },
    "internet_and_modern": {
        "percentage": 15,
        "description": "Mysteries born on the internet, like Cicada 3301, ARGs, strange YouTube channels, and digital ghosts.",
        "narrative_style": "fast_paced_and_tech_focused",
        "visual_style": "screen_recordings, code, forums, digital_glitches"
    },
    "high_strangeness": {
        "percentage": 10,
        "description": "The truly weird: Glitches in the matrix, Fortean phenomena, cryptids, and stories that defy categorization.",
        "narrative_style": "uncanny_and_mind_bending",
        "visual_style": "surreal_visuals, abstract_animation, unsettling_footage"
    }
}

def get_current_time():
    """Get current time in EST"""
    est = pytz.timezone('US/Eastern')
    return datetime.now(est)

def should_post_now():
    """Determine if the current hour is a scheduled posting time."""
    current = get_current_time()
    weekday = current.weekday()
    hour = current.hour
    
    if weekday not in OPTIMAL_SCHEDULE:
        return False, "low", "off_schedule", current
    
    schedule = OPTIMAL_SCHEDULE[weekday]
    
    # Check if the current hour matches any of the scheduled hours
    try:
        idx = schedule["times"].index(hour)
        content_type = schedule["content_types"][idx]
        priority = schedule["priority"][idx]
        return True, priority, content_type, current
    except ValueError:
        # Current hour is not in the list of scheduled times for today
        return False, "low", "off_schedule", current

def get_next_optimal_slot():
    """Get the next scheduled posting time."""
    current = get_current_time()
    
    # Check all upcoming slots in the next 7 days
    for day_offset in range(8):
        check_date = current + timedelta(days=day_offset)
        weekday = check_date.weekday()
        
        if weekday not in OPTIMAL_SCHEDULE:
            continue
        
        schedule = OPTIMAL_SCHEDULE[weekday]
        for idx, hour in enumerate(schedule["times"]):
            slot_time = check_date.replace(hour=hour, minute=0, second=0, microsecond=0)
            
            if slot_time > current:
                return {
                    "time": slot_time.strftime("%A %I:%M %p EST"),
                    "datetime": slot_time.isoformat(),
                    "content_type": schedule["content_types"][idx],
                    "priority": schedule["priority"][idx],
                    "day_name": slot_time.strftime("%A"),
                    "time_only": slot_time.strftime("%I:%M %p")
                }
    
    return None

def generate_weekly_schedule():
    """Generate the full week's posting schedule for display."""
    schedule = {}
    
    for weekday, config in OPTIMAL_SCHEDULE.items():
        day_name = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"][weekday]
        slots = []
        
        for idx, hour in enumerate(config["times"]):
            slots.append({
                "time": f"{hour:02d}:00",
                "content_type": config["content_types"][idx],
                "priority": config["priority"][idx]
            })
        
        schedule[day_name] = sorted(slots, key=lambda x: x['time'])
    
    return schedule

def main():
    """Main scheduler logic for Mythica Report."""
    should_post, priority, content_type, current_time = should_post_now()
    next_slot = get_next_optimal_slot()
    weekly = generate_weekly_schedule()
    
    # Create the output data structure
    output = {
        "should_post_now": should_post,
        "current_time_est": current_time.strftime("%Y-%m-%d %H:%M EST"),
        "current_priority": priority,
        "current_content_type": content_type,
        "next_optimal_slot": next_slot,
        "weekly_schedule": weekly,
        "content_pillars": CONTENT_PILLARS,
        "niche": "mystery_stories",
        "channel": "Mythica Report",
        "posting_philosophy": "Maintain a consistent schedule with a diverse rotation of mystery sub-genres to maximize engagement and build an audience habit."
    }
    
    # Save the schedule data to a JSON file
    os.makedirs("tmp", exist_ok=True)
    with open("tmp/posting_schedule.json", "w") as f:
        json.dump(output, f, indent=2)
    
    # Print a summary to the console
    print("=" * 60)
    print("🔮 MYTHICA REPORT - POSTING SCHEDULER")
    print("=" * 60)
    print(f"Current Time: {current_time.strftime('%A, %B %d, %Y at %I:%M %p EST')}")
    print(f"Should Post: {'✅ YES' if should_post else '❌ NO'}")
    print(f"Priority: {priority.upper()}")
    print(f"Content Type: {content_type.replace('_', ' ').title()}")
    print()
    
    if next_slot:
        print(f"📅 Next Optimal Slot:")
        print(f"   {next_slot['time']}")
        print(f"   Content: {next_slot['content_type'].replace('_', ' ').title()}")
        print(f"   Priority: {next_slot['priority'].upper()}")
    
    print("\n" + "-" * 25)
    print("📊 This Week's Schedule:")
    print("-" * 25)
    for day, slots in weekly.items():
        print(f"\n{day}:")
        for slot in slots:
            emoji_map = {"highest": "🔥", "high": "⭐", "medium": "•"}
            emoji = emoji_map.get(slot['priority'], '•')
            print(f"  {emoji} {slot['time']} - {slot['content_type'].replace('_', ' ').title()} ({slot['priority']})")
    
    print("\n" + "=" * 60)
    
    # Set GitHub Action outputs if running in a GH environment
    if "GITHUB_OUTPUT" in os.environ:
        with open(os.environ["GITHUB_OUTPUT"], "a") as f:
            f.write(f"should_post={'true' if should_post else 'false'}\n")
            f.write(f"priority={priority}\n")
            f.write(f"content_type={content_type}\n")
            f.write(f"current_time_est={output['current_time_est']}\n")

if __name__ == "__main__":
    main()