"""
MLB Film Room Clip Downloader
Downloads video clips from MLB Film Room using the MLB Stats API
and optionally runs YOLO inference on them.
"""

import argparse
import requests
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Optional

# Import predict function from main.py
from main import predict, BASE_DIR

# Configuration
MLB_STATS_API_BASE = "https://statsapi.mlb.com/api/v1"
VIDEOS_DIR = BASE_DIR / "videos"
VIDEOS_DIR.mkdir(exist_ok=True)


def search_mlb_clips(
    date: Optional[str] = None,
    date_range: Optional[str] = None,
    team: Optional[str] = None,
    player: Optional[str] = None,
    game_id: Optional[str] = None,
    play_type: Optional[str] = None,
    limit: int = 10
) -> List[Dict]:
    """
    Search MLB Film Room for video clips based on criteria.
    
    Args:
        date: Single date in YYYY-MM-DD format
        date_range: Date range in YYYY-MM-DD:YYYY-MM-DD format
        team: Team abbreviation (e.g., 'NYY', 'BOS')
        player: Player name or ID
        game_id: Specific game ID
        play_type: Type of play (e.g., 'home_run', 'strikeout')
        limit: Maximum number of clips to return
    
    Returns:
        List of dictionaries containing clip metadata
    """
    clips = []
    
    try:
        # If game_id is provided, get highlights for that specific game
        if game_id:
            url = f"{MLB_STATS_API_BASE}/game/{game_id}/content"
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                highlights = data.get('highlights', {}).get('highlights', {}).get('items', [])
                for highlight in highlights[:limit]:
                    clip_info = extract_clip_info(highlight, game_id)
                    if clip_info:
                        clips.append(clip_info)
        else:
            # Search by date range
            dates = []
            if date:
                dates = [date]
            elif date_range:
                start_date, end_date = date_range.split(':')
                start = datetime.strptime(start_date, '%Y-%m-%d')
                end = datetime.strptime(end_date, '%Y-%m-%d')
                current = start
                while current <= end:
                    dates.append(current.strftime('%Y-%m-%d'))
                    current += timedelta(days=1)
            else:
                # Default to last 7 days
                today = datetime.now()
                for i in range(7):
                    dates.append((today - timedelta(days=i)).strftime('%Y-%m-%d'))
            
            # Get games for each date
            for date_str in dates[:30]:  # Limit to 30 days max
                games = get_games_for_date(date_str, team)
                for game in games[:5]:  # Limit games per day
                    game_pk = game.get('gamePk')
                    if game_pk:
                        url = f"{MLB_STATS_API_BASE}/game/{game_pk}/content"
                        response = requests.get(url, timeout=10)
                        if response.status_code == 200:
                            data = response.json()
                            highlights = data.get('highlights', {}).get('highlights', {}).get('items', [])
                            for highlight in highlights:
                                # Filter by play type if specified
                                if play_type and not matches_play_type(highlight, play_type):
                                    continue
                                clip_info = extract_clip_info(highlight, game_pk)
                                if clip_info:
                                    clips.append(clip_info)
                                    if len(clips) >= limit:
                                        return clips
    
    except Exception as e:
        print(f"⚠️  Error searching for clips: {e}")
    
    return clips[:limit]


def get_games_for_date(date: str, team: Optional[str] = None) -> List[Dict]:
    """Get games for a specific date."""
    try:
        url = f"{MLB_STATS_API_BASE}/schedule"
        params = {
            'date': date,
            'sportId': 1,  # MLB
            'hydrate': 'linescore'
        }
        if team:
            # Try to find team ID
            team_id = get_team_id(team)
            if team_id:
                params['teamId'] = team_id
        
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            data = response.json()
            dates = data.get('dates', [])
            if dates:
                return dates[0].get('games', [])
    except Exception as e:
        print(f"⚠️  Error getting games for {date}: {e}")
    return []


def get_team_id(team_abbr: str) -> Optional[int]:
    """Convert team abbreviation to team ID."""
    team_map = {
        'NYY': 147, 'BOS': 111, 'TB': 139, 'TOR': 141, 'BAL': 110,
        'CLE': 114, 'MIN': 142, 'CWS': 145, 'DET': 116, 'KC': 118,
        'HOU': 117, 'TEX': 140, 'SEA': 136, 'LAA': 108, 'OAK': 133,
        'ATL': 144, 'PHI': 143, 'NYM': 121, 'MIA': 146, 'WSH': 120,
        'MIL': 158, 'CHC': 112, 'STL': 138, 'CIN': 113, 'PIT': 134,
        'LAD': 119, 'SD': 135, 'SF': 137, 'ARI': 109, 'COL': 115
    }
    return team_map.get(team_abbr.upper())


def matches_play_type(highlight: Dict, play_type: str) -> bool:
    """Check if highlight matches the specified play type."""
    title = highlight.get('title', '').lower()
    description = highlight.get('description', '').lower()
    keywords = highlight.get('keywords', [])
    
    play_type_lower = play_type.lower()
    play_keywords = {
        'home_run': ['home run', 'homer', 'hr'],
        'strikeout': ['strikeout', 'strikes out', 'k'],
        'double': ['double', '2b'],
        'triple': ['triple', '3b'],
        'single': ['single', '1b'],
        'walk': ['walk', 'bb', 'base on balls'],
        'hit': ['hit', 'single', 'double', 'triple', 'home run'],
        'out': ['out', 'caught', 'fielded']
    }
    
    if play_type_lower in play_keywords:
        keywords_to_match = play_keywords[play_type_lower]
        text = f"{title} {description} {' '.join(keywords)}".lower()
        return any(kw in text for kw in keywords_to_match)
    
    return play_type_lower in title or play_type_lower in description


def extract_clip_info(highlight: Dict, game_id: str) -> Optional[Dict]:
    """Extract relevant information from highlight data."""
    try:
        playbacks = highlight.get('playbacks', [])
        if not playbacks:
            return None
        
        # Get the best quality video URL
        video_url = None
        for playback in playbacks:
            if playback.get('name') == 'mp4Avc':
                video_url = playback.get('url')
                break
        if not video_url:
            # Fallback to first available
            video_url = playbacks[0].get('url')
        
        if not video_url:
            return None
        
        return {
            'title': highlight.get('title', 'Untitled'),
            'description': highlight.get('description', ''),
            'url': video_url,
            'game_id': game_id,
            'date': highlight.get('date', ''),
            'duration': highlight.get('duration', ''),
            'keywords': highlight.get('keywords', [])
        }
    except Exception as e:
        print(f"⚠️  Error extracting clip info: {e}")
        return None


def download_clip(video_url: str, output_path: Path, title: str = "") -> bool:
    """
    Download a video clip from URL.
    
    Args:
        video_url: URL of the video to download
        output_path: Path where video should be saved
        title: Title for filename (sanitized)
    
    Returns:
        True if successful, False otherwise
    """
    try:
        # Sanitize filename
        safe_title = "".join(c for c in title if c.isalnum() or c in (' ', '-', '_')).rstrip()
        safe_title = safe_title.replace(' ', '_')[:50]  # Limit length
        
        if not output_path.suffix:
            output_path = output_path.parent / f"{safe_title}.mp4"
        
        # Check if file already exists
        if output_path.exists():
            print(f"⏭️  Skipping {output_path.name} (already exists)")
            return True
        
        print(f"📥 Downloading: {title[:60]}...")
        
        # Check if URL is a streaming format (M3U8, etc.)
        is_streaming = video_url.endswith('.m3u8') or 'm3u8' in video_url.lower()
        
        if is_streaming:
            # Try yt-dlp for streaming URLs
            try:
                import yt_dlp
                ydl_opts = {
                    'outtmpl': str(output_path.with_suffix('')),
                    'format': 'best',
                    'quiet': False,
                    'no_warnings': False,
                }
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    ydl.download([video_url])
                if output_path.exists():
                    print(f"✅ Downloaded: {output_path.name}")
                    return True
            except ImportError:
                print("⚠️  yt-dlp not installed. Install with: pip install yt-dlp")
                print("   Falling back to direct download...")
            except Exception as e:
                print(f"⚠️  yt-dlp failed: {e}")
                print("   Falling back to direct download...")
        
        # Try direct download
        response = requests.get(video_url, stream=True, timeout=30, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        if response.status_code == 200:
            total_size = int(response.headers.get('content-length', 0))
            with open(output_path, 'wb') as f:
                downloaded = 0
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            percent = (downloaded / total_size) * 100
                            print(f"\r   Progress: {percent:.1f}%", end='', flush=True)
            print(f"\n✅ Downloaded: {output_path.name}")
            return True
        else:
            print(f"⚠️  Failed to download: HTTP {response.status_code}")
            if is_streaming:
                print("   Try installing yt-dlp: pip install yt-dlp")
            return False
            
    except Exception as e:
        print(f"❌ Error downloading clip: {e}")
        return False


def download_clips_batch(clips: List[Dict], output_dir: Path, inference: bool = False) -> List[Path]:
    """
    Download multiple clips in batch.
    
    Args:
        clips: List of clip dictionaries from search_mlb_clips
        output_dir: Directory to save clips
        inference: Whether to run inference on downloaded clips
    
    Returns:
        List of paths to downloaded clips
    """
    downloaded_paths = []
    
    print(f"\n📦 Downloading {len(clips)} clip(s)...")
    print("=" * 60)
    
    for i, clip in enumerate(clips, 1):
        print(f"\n[{i}/{len(clips)}] {clip['title']}")
        
        # Create filename from title and game ID
        safe_title = "".join(c for c in clip['title'] if c.isalnum() or c in (' ', '-', '_')).rstrip()
        safe_title = safe_title.replace(' ', '_')[:50]
        filename = f"{safe_title}_{clip['game_id']}.mp4"
        output_path = output_dir / filename
        
        if download_clip(clip['url'], output_path, clip['title']):
            downloaded_paths.append(output_path)
            
            # Run inference if requested
            if inference:
                print(f"🔍 Running inference on {output_path.name}...")
                try:
                    predict(str(output_path))
                    print(f"✅ Inference complete for {output_path.name}")
                except Exception as e:
                    print(f"⚠️  Inference error: {e}")
    
    print("\n" + "=" * 60)
    print(f"✅ Downloaded {len(downloaded_paths)}/{len(clips)} clips")
    
    return downloaded_paths


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Download MLB Film Room clips and optionally run YOLO inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download home runs from last week
  python mlb_clip_downloader.py --play-type home_run --date-range 2024-01-01:2024-01-07 --limit 10

  # Download and run inference on strikeouts
  python mlb_clip_downloader.py --team NYY --play-type strikeout --inference

  # Download specific game highlights
  python mlb_clip_downloader.py --game-id 747175 --inference
        """
    )
    
    # Date options
    date_group = parser.add_mutually_exclusive_group()
    date_group.add_argument('--date', type=str, help='Single date (YYYY-MM-DD)')
    date_group.add_argument('--date-range', type=str, help='Date range (YYYY-MM-DD:YYYY-MM-DD)')
    
    # Search filters
    parser.add_argument('--team', type=str, help='Team abbreviation (e.g., NYY, BOS)')
    parser.add_argument('--player', type=str, help='Player name or ID')
    parser.add_argument('--game-id', type=str, help='Specific game ID')
    parser.add_argument('--play-type', type=str, 
                       help='Type of play (home_run, strikeout, double, triple, single, walk, hit, out)')
    
    # Options
    parser.add_argument('--limit', type=int, default=10, help='Maximum number of clips (default: 10)')
    parser.add_argument('--inference', action='store_true', 
                       help='Run YOLO inference on downloaded clips')
    parser.add_argument('--output-dir', type=str, default=None,
                       help=f'Output directory (default: {VIDEOS_DIR})')
    
    args = parser.parse_args()
    
    # Set output directory
    output_dir = Path(args.output_dir) if args.output_dir else VIDEOS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Search for clips
    print("🔍 Searching MLB Film Room...")
    clips = search_mlb_clips(
        date=args.date,
        date_range=args.date_range,
        team=args.team,
        player=args.player,
        game_id=args.game_id,
        play_type=args.play_type,
        limit=args.limit
    )
    
    if not clips:
        print("❌ No clips found matching your criteria")
        return
    
    print(f"✅ Found {len(clips)} clip(s)")
    
    # Download clips
    downloaded = download_clips_batch(clips, output_dir, inference=args.inference)
    
    if downloaded:
        print(f"\n📁 Clips saved to: {output_dir}")
        if args.inference:
            print(f"🎯 Inference results saved to: {BASE_DIR / 'runs' / 'predictions'}")


if __name__ == "__main__":
    main()

