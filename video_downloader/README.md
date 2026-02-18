# Video Downloader

A Python tool for downloading MLB video clips from the MLB Film Room using the MLB Stats API. The tool can search for clips by various criteria (date, team, player, game ID, play type) and optionally run YOLO object detection inference on downloaded videos.

## Features

- 🔍 **Search MLB Film Room** - Search for video clips using multiple criteria
- 📥 **Download Videos** - Download clips directly or via streaming URLs
- 🎯 **Optional Inference** - Automatically run YOLO inference on downloaded clips
- 🏷️ **Smart Filtering** - Filter by team, player, game ID, play type, or date range
- 📁 **Organized Storage** - Videos saved to organized directory structure

## Requirements

Install the required dependencies:

```bash
pip install requests yt-dlp
```

**Note:** `yt-dlp` is optional but recommended for downloading streaming video formats (M3U8). If not installed, the tool will attempt direct downloads.

## Usage

### Basic Usage

```bash
# Download clips from the last 7 days (default)
python mlb_clip_downloader.py

# Download clips from a specific date
python mlb_clip_downloader.py --date 2024-01-15

# Download clips from a date range
python mlb_clip_downloader.py --date-range 2024-01-01:2024-01-07
```

### Search Filters

```bash
# Search by team
python mlb_clip_downloader.py --team NYY

# Search by play type
python mlb_clip_downloader.py --play-type home_run

# Search by specific game
python mlb_clip_downloader.py --game-id 747175

# Combine filters
python mlb_clip_downloader.py --team BOS --play-type strikeout --date 2024-01-15
```

### With Inference

```bash
# Download and run YOLO inference on clips
python mlb_clip_downloader.py --play-type home_run --inference

# Download specific game highlights with inference
python mlb_clip_downloader.py --game-id 747175 --inference
```

### Command Line Options

| Option | Description | Example |
|--------|-------------|---------|
| `--date` | Single date (YYYY-MM-DD) | `--date 2024-01-15` |
| `--date-range` | Date range (YYYY-MM-DD:YYYY-MM-DD) | `--date-range 2024-01-01:2024-01-07` |
| `--team` | Team abbreviation | `--team NYY`, `--team BOS` |
| `--player` | Player name or ID | `--player "Mike Trout"` |
| `--game-id` | Specific game ID | `--game-id 747175` |
| `--play-type` | Type of play | `home_run`, `strikeout`, `double`, `triple`, `single`, `walk`, `hit`, `out` |
| `--limit` | Maximum number of clips (default: 10) | `--limit 20` |
| `--inference` | Run YOLO inference on downloaded clips | `--inference` |
| `--output-dir` | Custom output directory | `--output-dir ./my_videos` |

## Supported Play Types

- `home_run` - Home runs
- `strikeout` - Strikeouts
- `double` - Doubles
- `triple` - Triples
- `single` - Singles
- `walk` - Walks
- `hit` - Any hit
- `out` - Outs

## Supported Teams

All 30 MLB teams are supported. Use the team abbreviation:

- **AL East**: NYY, BOS, TB, TOR, BAL
- **AL Central**: CLE, MIN, CWS, DET, KC
- **AL West**: HOU, TEX, SEA, LAA, OAK
- **NL East**: ATL, PHI, NYM, MIA, WSH
- **NL Central**: MIL, CHC, STL, CIN, PIT
- **NL West**: LAD, SD, SF, ARI, COL

## Examples

### Download Home Runs from Last Week

```bash
python mlb_clip_downloader.py --play-type home_run --date-range 2024-01-01:2024-01-07 --limit 10
```

### Download and Analyze Strikeouts

```bash
python mlb_clip_downloader.py --team NYY --play-type strikeout --inference
```

### Download Specific Game Highlights

```bash
python mlb_clip_downloader.py --game-id 747175 --inference
```

## Output

Downloaded videos are saved to the `videos/` directory by default. Filenames are automatically generated from the clip title and game ID:

```
videos/
  ├── Alek_Thomas_game-tying_two-run_single_747175.mp4
  ├── Christian_Koss_two-run_homer_3_776866.mp4
  └── ...
```

If inference is enabled, results are saved to:
```
object recognition/runs/predictions/
```

## Integration with Object Recognition

This tool integrates with the `object recognition` module. When `--inference` is used, it automatically calls the YOLO prediction function to detect baseball objects (Ball, Catcher's glove, Batter's bat, Homeplate) in the downloaded videos.

## Notes

- The tool uses the MLB Stats API which is publicly available and doesn't require authentication
- Some video URLs may be streaming formats (M3U8) which require `yt-dlp` for reliable downloading
- Downloaded files are skipped if they already exist (prevents re-downloading)
- The tool respects rate limits and includes appropriate timeouts

## Troubleshooting

### "yt-dlp not installed" Warning

If you see this warning, install yt-dlp for better streaming video support:

```bash
pip install yt-dlp
```

### No Clips Found

- Verify the date range is valid (clips may not be available for very old dates)
- Check that the team abbreviation is correct
- Try a broader search (remove filters or increase date range)

### Download Failures

- Some video URLs may be temporary or require authentication
- Try installing `yt-dlp` for better compatibility with streaming formats
- Check your internet connection

## API Reference

The tool uses the MLB Stats API endpoints:
- `https://statsapi.mlb.com/api/v1/schedule` - Get games for a date
- `https://statsapi.mlb.com/api/v1/game/{game_id}/content` - Get highlights for a game
