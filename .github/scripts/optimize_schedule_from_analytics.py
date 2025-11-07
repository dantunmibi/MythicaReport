#!/usr/bin/env python3
"""
📊 YouTube Analytics Optimizer for Mythica Report
ENHANCED: Uses REAL retention data from YouTube Analytics API

Run manually every Sunday:
python .github/scripts/optimize_schedule_from_analytics.py
"""

import os
import json
from datetime import datetime, timedelta
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

TMP = os.getenv("GITHUB_WORKSPACE", ".") + "/tmp"
if not os.path.exists(TMP):
    os.makedirs(TMP)

SCHEDULE_FILE = "config/posting_schedule.json"

def get_youtube_credentials():
    """Get YouTube API credentials with Analytics scope"""
    creds = Credentials(
        token=None,
        refresh_token=os.getenv('GOOGLE_REFRESH_TOKEN'),
        token_uri='https://oauth2.googleapis.com/token',
        client_id=os.getenv('GOOGLE_CLIENT_ID'),
        client_secret=os.getenv('GOOGLE_CLIENT_SECRET'),
        scopes=[
            'https://www.googleapis.com/auth/youtube.readonly',
            'https://www.googleapis.com/auth/yt-analytics.readonly'
        ]
    )
    return creds

def get_channel_id(youtube):
    """Get the channel ID for the authenticated user"""
    try:
        request = youtube.channels().list(
            part='id',
            mine=True
        )
        response = request.execute()
        
        if response['items']:
            channel_id = response['items'][0]['id']
            print(f"✅ Channel ID: {channel_id}")
            return channel_id
        else:
            print("❌ No channel found for this account")
            return None
    except HttpError as e:
        print(f"❌ Error getting channel ID: {e}")
        return None

def fetch_recent_videos(youtube, days=30):
    """Fetch recent video IDs and metadata"""
    
    print(f"\n📊 Fetching videos from last {days} days...")
    
    try:
        # Get channel's uploads playlist
        channels_response = youtube.channels().list(
            part='contentDetails',
            mine=True
        ).execute()
        
        uploads_playlist_id = channels_response['items'][0]['contentDetails']['relatedPlaylists']['uploads']
        
        # Get recent videos
        videos = []
        next_page_token = None
        
        while len(videos) < 100:  # Limit to last 100 videos
            request = youtube.playlistItems().list(
                part='snippet,contentDetails',
                playlistId=uploads_playlist_id,
                maxResults=50,
                pageToken=next_page_token
            )
            
            response = request.execute()
            videos.extend(response['items'])
            
            next_page_token = response.get('nextPageToken')
            if not next_page_token:
                break
        
        # Filter to last N days
        cutoff_date = datetime.now() - timedelta(days=days)
        recent_videos = []
        
        for v in videos:
            pub_date_str = v['snippet']['publishedAt']
            pub_date = datetime.fromisoformat(pub_date_str.replace('Z', '+00:00'))
            
            if pub_date > cutoff_date:
                recent_videos.append({
                    'video_id': v['contentDetails']['videoId'],
                    'title': v['snippet']['title'],
                    'published_at': pub_date_str,
                    'published_datetime': pub_date
                })
        
        print(f"✅ Found {len(recent_videos)} videos in last {days} days")
        return recent_videos
        
    except HttpError as e:
        print(f"❌ Error fetching videos: {e}")
        return []

def fetch_analytics_data(youtube_analytics, channel_id, video_ids, start_date, end_date):
    """Fetch REAL retention data from YouTube Analytics API"""
    
    print(f"\n📊 Fetching analytics data for {len(video_ids)} videos...")
    
    video_analytics = []
    
    # YouTube Analytics API has a limit on video IDs per request
    # Process in batches of 200
    batch_size = 200
    
    for i in range(0, len(video_ids), batch_size):
        batch = video_ids[i:i+batch_size]
        video_filter = ','.join(batch)
        
        try:
            # Request analytics data
            request = youtube_analytics.reports().query(
                ids=f'channel=={channel_id}',
                startDate=start_date.strftime('%Y-%m-%d'),
                endDate=end_date.strftime('%Y-%m-%d'),
                metrics='views,estimatedMinutesWatched,averageViewDuration,averageViewPercentage,subscribersGained,subscribersLost',
                dimensions='video',
                filters=f'video=={video_filter}',
                maxResults=200
            )
            
            response = request.execute()
            
            if 'rows' in response:
                for row in response['rows']:
                    video_analytics.append({
                        'video_id': row[0],
                        'views': int(row[1]) if len(row) > 1 else 0,
                        'estimated_minutes_watched': float(row[2]) if len(row) > 2 else 0,
                        'average_view_duration': int(row[3]) if len(row) > 3 else 0,  # seconds
                        'average_view_percentage': float(row[4]) if len(row) > 4 else 0,  # REAL RETENTION %
                        'subscribers_gained': int(row[5]) if len(row) > 5 else 0,
                        'subscribers_lost': int(row[6]) if len(row) > 6 else 0
                    })
            
            print(f"   ✅ Batch {i//batch_size + 1}: {len(response.get('rows', []))} videos")
            
        except HttpError as e:
            print(f"   ⚠️ Analytics API error for batch {i//batch_size + 1}: {e}")
            continue
    
    print(f"✅ Retrieved analytics for {len(video_analytics)} videos")
    return video_analytics

def merge_video_data(videos, analytics):
    """Merge video metadata with analytics data"""
    
    analytics_by_id = {a['video_id']: a for a in analytics}
    
    merged = []
    for video in videos:
        video_id = video['video_id']
        
        if video_id in analytics_by_id:
            merged.append({
                **video,
                **analytics_by_id[video_id],
                'retention_percentage': analytics_by_id[video_id]['average_view_percentage']
            })
    
    print(f"✅ Merged data for {len(merged)} videos")
    return merged

def analyze_performance_by_slot(video_data):
    """Group videos by posting time slot and analyze retention"""
    
    print("\n📊 Analyzing performance by time slot...")
    
    slot_performance = {}
    
    for video in video_data:
        # Parse publication time
        pub_time = video['published_datetime']
        hour = pub_time.hour
        day = pub_time.strftime('%A')
        
        # Map to slot
        slot_key = f"{day}_{hour:02d}:00"
        
        if slot_key not in slot_performance:
            slot_performance[slot_key] = {
                'videos': [],
                'avg_retention': 0,
                'avg_views': 0,
                'avg_watch_time': 0,
                'total_videos': 0,
                'subscribers_net': 0
            }
        
        slot_performance[slot_key]['videos'].append(video)
        slot_performance[slot_key]['total_videos'] += 1
    
    # Calculate averages
    for slot in slot_performance:
        videos = slot_performance[slot]['videos']
        
        if videos:
            slot_performance[slot]['avg_retention'] = sum(v['retention_percentage'] for v in videos) / len(videos)
            slot_performance[slot]['avg_views'] = sum(v['views'] for v in videos) / len(videos)
            slot_performance[slot]['avg_watch_time'] = sum(v['average_view_duration'] for v in videos) / len(videos)
            slot_performance[slot]['subscribers_net'] = sum(v['subscribers_gained'] - v['subscribers_lost'] for v in videos)
    
    # Rank slots by retention
    ranked_slots = sorted(
        slot_performance.items(),
        key=lambda x: x[1]['avg_retention'],
        reverse=True
    )
    
    print("\n🏆 TOP PERFORMING SLOTS (by retention):")
    for i, (slot, data) in enumerate(ranked_slots[:10], 1):
        print(f"{i}. {slot}:")
        print(f"   Retention: {data['avg_retention']:.1f}%")
        print(f"   Views: {data['avg_views']:.0f}")
        print(f"   Watch time: {data['avg_watch_time']:.0f}s")
        print(f"   Subscribers: {data['subscribers_net']:+d}")
        print(f"   Videos: {data['total_videos']}")
    
    print("\n⚠️ WORST PERFORMING SLOTS:")
    for i, (slot, data) in enumerate(reversed(ranked_slots[-5:]), 1):
        print(f"{i}. {slot}:")
        print(f"   Retention: {data['avg_retention']:.1f}%")
        print(f"   Views: {data['avg_views']:.0f}")
        print(f"   Videos: {data['total_videos']}")
    
    return slot_performance

def suggest_schedule_optimization(slot_performance, min_videos=3):
    """
    Suggest schedule changes based on performance
    
    Criteria:
    - INCREASE: avg_retention > 70% with 3+ videos
    - DECREASE: avg_retention < 50% with 3+ videos
    - REMOVE: avg_retention < 40% with 5+ videos
    """
    
    print("\n💡 SCHEDULE OPTIMIZATION RECOMMENDATIONS:")
    
    recommendations = []
    
    for slot, data in slot_performance.items():
        if data['total_videos'] < min_videos:
            continue  # Not enough data
        
        if data['avg_retention'] > 70:
            recommendations.append({
                'action': 'INCREASE',
                'slot': slot,
                'current_retention': data['avg_retention'],
                'reason': f"High retention ({data['avg_retention']:.1f}%) - consider adding more posts",
                'priority': 'high',
                'data': data
            })
        
        elif data['avg_retention'] < 40 and data['total_videos'] >= 5:
            recommendations.append({
                'action': 'REMOVE',
                'slot': slot,
                'current_retention': data['avg_retention'],
                'reason': f"Low retention ({data['avg_retention']:.1f}%) - consider removing this slot",
                'priority': 'high',
                'data': data
            })
        
        elif data['avg_retention'] < 50:
            recommendations.append({
                'action': 'DECREASE',
                'slot': slot,
                'current_retention': data['avg_retention'],
                'reason': f"Below-average retention ({data['avg_retention']:.1f}%) - reduce frequency",
                'priority': 'medium',
                'data': data
            })
    
    # Sort by priority and retention
    recommendations.sort(key=lambda x: (x['priority'] == 'high', x['current_retention']), reverse=True)
    
    for rec in recommendations:
        print(f"\n{rec['action']}: {rec['slot']}")
        print(f"   {rec['reason']}")
        print(f"   Current: {rec['current_retention']:.1f}% retention, {rec['data']['avg_views']:.0f} views")
    
    return recommendations

def save_optimization_report(slot_performance, recommendations, video_data):
    """Save detailed optimization report"""
    
    output = {
        'generated_at': datetime.now().isoformat(),
        'analysis_period': {
            'start': (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d'),
            'end': datetime.now().strftime('%Y-%m-%d'),
            'total_videos': len(video_data)
        },
        'slot_performance': {
            slot: {
                'avg_retention': data['avg_retention'],
                'avg_views': data['avg_views'],
                'avg_watch_time': data['avg_watch_time'],
                'subscribers_net': data['subscribers_net'],
                'total_videos': data['total_videos']
            }
            for slot, data in slot_performance.items()
        },
        'recommendations': [
            {
                'action': rec['action'],
                'slot': rec['slot'],
                'retention': rec['current_retention'],
                'reason': rec['reason'],
                'priority': rec['priority']
            }
            for rec in recommendations
        ],
        'summary': {
            'best_slot': max(slot_performance.items(), key=lambda x: x[1]['avg_retention'])[0] if slot_performance else None,
            'worst_slot': min(slot_performance.items(), key=lambda x: x[1]['avg_retention'])[0] if slot_performance else None,
            'avg_retention_overall': sum(d['avg_retention'] for d in slot_performance.values()) / len(slot_performance) if slot_performance else 0,
            'total_subscribers_gained': sum(d['subscribers_net'] for d in slot_performance.values())
        }
    }
    
    output_file = os.path.join(TMP, "analytics_optimization.json")
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n✅ Optimization report saved to: {output_file}")
    
    # Also save a human-readable summary
    summary_file = os.path.join(TMP, "analytics_summary.txt")
    with open(summary_file, 'w') as f:
        f.write("="*70 + "\n")
        f.write("MYTHICA REPORT - ANALYTICS OPTIMIZATION SUMMARY\n")
        f.write("="*70 + "\n\n")
        f.write(f"Analysis Period: {output['analysis_period']['start']} to {output['analysis_period']['end']}\n")
        f.write(f"Videos Analyzed: {output['analysis_period']['total_videos']}\n\n")
        
        f.write("OVERALL PERFORMANCE:\n")
        f.write(f"  Average Retention: {output['summary']['avg_retention_overall']:.1f}%\n")
        f.write(f"  Total Subscribers Gained: {output['summary']['total_subscribers_gained']:+d}\n")
        f.write(f"  Best Slot: {output['summary']['best_slot']}\n")
        f.write(f"  Worst Slot: {output['summary']['worst_slot']}\n\n")
        
        f.write("RECOMMENDATIONS:\n")
        for rec in recommendations:
            f.write(f"\n{rec['action']}: {rec['slot']}\n")
            f.write(f"  Retention: {rec['current_retention']:.1f}%\n")
            f.write(f"  {rec['reason']}\n")
    
    print(f"✅ Summary saved to: {summary_file}")
    
    return output

def main():
    """Main execution function"""
    
    print("="*70)
    print("📊 MYTHICA REPORT - ANALYTICS OPTIMIZER (ENHANCED)")
    print("="*70)
    print("Using REAL retention data from YouTube Analytics API\n")
    
    try:
        # Get credentials
        print("🔐 Authenticating with YouTube...")
        creds = get_youtube_credentials()
        
        # Build API clients
        youtube = build('youtube', 'v3', credentials=creds)
        youtube_analytics = build('youtubeAnalytics', 'v2', credentials=creds)
        
        # Get channel ID
        channel_id = get_channel_id(youtube)
        if not channel_id:
            print("❌ Could not get channel ID. Exiting.")
            return
        
        # Fetch recent videos
        days = 30
        videos = fetch_recent_videos(youtube, days=days)
        
        if not videos:
            print("⚠️ No videos found in the last 30 days")
            return
        
        # Fetch analytics data
        video_ids = [v['video_id'] for v in videos]
        start_date = datetime.now() - timedelta(days=days)
        end_date = datetime.now()
        
        analytics = fetch_analytics_data(youtube_analytics, channel_id, video_ids, start_date, end_date)
        
        if not analytics:
            print("⚠️ No analytics data available")
            return
        
        # Merge data
        video_data = merge_video_data(videos, analytics)
        
        # Analyze performance by slot
        slot_performance = analyze_performance_by_slot(video_data)
        
        # Generate recommendations
        recommendations = suggest_schedule_optimization(slot_performance)
        
        # Save report
        report = save_optimization_report(slot_performance, recommendations, video_data)
        
        print("\n" + "="*70)
        print("✅ ANALYTICS OPTIMIZATION COMPLETE")
        print("="*70)
        print("\n📋 NEXT STEPS:")
        print("1. Review tmp/analytics_summary.txt for recommendations")
        print("2. Manually update config/posting_schedule.json")
        print("3. Commit and push changes")
        print("4. Run this script again next Sunday")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()