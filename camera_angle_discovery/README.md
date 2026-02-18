# Camera Angle Discovery System

Automatically discovers and categorizes camera angles in baseball game videos using unsupervised machine learning. The system extracts frames, analyzes them with pre-trained vision models, clusters similar camera angles, and organizes video segments into subfolders by angle.

## Features

- **AI-Powered Angle Classification**: Uses CLIP's text-image matching to intelligently classify camera angles based on descriptions (e.g., "center field view", "home plate closeup")
- **Global Angle Discovery**: Trains on ALL videos to discover common camera angles across the entire dataset
- **Automatic Angle Discovery**: Uses unsupervised clustering (DBSCAN, K-means, or hierarchical) OR AI classification to discover camera angles
- **Pre-trained Models**: Supports CLIP and ResNet50 for robust feature extraction
- **Video Segmentation**: Automatically segments videos by discovered angles
- **Organized Output**: Creates organized folder structure with metadata and visualizations
- **Global Angle Collection**: Collects all segments by angle across all videos for easy comparison
- **Shared Video Directory**: Integrates with shared `video_downloader/videos/` directory

## Installation

### 1. Install Python Dependencies

```bash
cd camera_angle_discovery
pip install -r requirements.txt
```

### 2. Install FFmpeg

FFmpeg is required for video segmentation. Install based on your OS:

**macOS:**
```bash
brew install ffmpeg
```

**Ubuntu/Debian:**
```bash
sudo apt-get install ffmpeg
```

**Windows:**
Download from [ffmpeg.org](https://ffmpeg.org/download.html)

### 3. Verify Installation

```bash
ffmpeg -version
```

## Usage

### Basic Usage

Process a specific video from the shared directory:
```bash
python main.py --video "Alek_Thomas_game-tying_two-run_single_747175.mp4"
```

Process all videos in the shared directory:
```bash
python main.py --batch
```

Process a custom video path:
```bash
python main.py --video-path "/path/to/custom/video.mp4"
```

### Configuration

Edit `config.yaml` to customize settings:

```yaml
video:
  frame_extraction_rate: 5.0  # Extract 5 frames per second (better temporal resolution)
  min_segment_duration: 0.5   # Minimum 0.5 seconds per segment (captures quick cuts)

features:
  model: "clip"  # Use CLIP model (or "resnet50")
  batch_size: 32

clustering:
  method: "ai_classifier"  # "ai_classifier", "dbscan", "kmeans", "hierarchical"
  use_ai_classifier: true  # Use AI to classify angles based on descriptions
  # angle_descriptions: null  # Custom descriptions (null = use defaults)
```

### Command Line Options

```
--video          Video filename from shared directory
--video-path     Full path to video file
--batch          Process all videos in shared directory
--config         Path to custom config file
```

## How It Works

The system uses a **two-phase approach** to discover camera angles:

### Phase 1: Model Setup
1. **Frame Extraction**: Extracts frames from ALL videos at specified intervals (default: 5 fps for better temporal resolution)
2. **Feature Extraction**: Uses pre-trained vision models (CLIP or ResNet50) to extract feature vectors from all frames
3. **Angle Model Setup**:
   - **AI Classifier Mode**: Uses CLIP's text-image matching to classify angles based on descriptions (e.g., "center field view", "home plate closeup")
   - **Clustering Mode**: Applies unsupervised clustering to ALL features from ALL videos to discover common camera angles
4. **Model Training**: Creates a global angle model (clustering) or loads AI classifier (classification)

### Phase 2: Video Processing
5. **Angle Assignment**: For each video:
   - **AI Classifier**: Matches frames to angle descriptions using CLIP's semantic understanding
   - **Clustering**: Assigns frames to discovered global angles using nearest-neighbor matching
6. **Temporal Smoothing**: Smooths angle transitions to reduce noise within each video
7. **Video Segmentation**: Extracts video segments for each discovered angle
8. **Per-Video Organization**: Organizes segments into subfolders per video with metadata and visualizations

### Phase 3: Global Angle Collection
9. **Global Organization**: Collects all segments from all videos and organizes them by angle into global folders
10. **Cross-Video Analysis**: Creates folders where you can see all segments of the same angle across all videos

This approach ensures that:
- Common camera angles (e.g., center field, first base, third base) are discovered across all videos
- Each video is labeled consistently using the same set of global angles
- Multiple angles per video are properly identified and categorized

## Output Structure

The system creates two types of organization:

### Per-Video Organization
```
output/
└── video_name/
    ├── angle_0/
    │   ├── segment_001.mp4
    │   ├── segment_002.mp4
    │   └── ...
    ├── angle_1/
    │   └── ...
    ├── metadata.json
    ├── summary.txt
    └── angle_timeline.png
```

### Global Angle Collection (NEW)
All segments from all videos are also organized by angle:
```
output/
└── angles_collected/
    ├── angle_0/
    │   ├── video1_seg_001.mp4
    │   ├── video2_seg_001.mp4
    │   ├── video3_seg_001.mp4
    │   └── ... (all segments from all videos with this angle)
    ├── angle_1/
    │   └── ... (all segments from all videos with this angle)
    ├── angle_2/
    │   └── ...
    ├── global_metadata.json
    └── global_summary.txt
```

This allows you to:
- View all segments of the same camera angle across all videos
- Compare how the same angle appears in different videos
- Easily access all examples of a specific camera angle

### Metadata

`metadata.json` contains:
- Number of angles discovered
- Total segments
- Duration and frame counts for each angle
- Segment file paths and timestamps

### Visualization

`angle_timeline.png` shows:
- Timeline of angle changes throughout the video
- Distribution of frames across angles

## Configuration Options

### Video Settings

- `input_path`: Direct path to video (null to use shared directory)
- `frame_extraction_rate`: Frames per second to extract (default: 5.0) - higher rate captures quick cuts better
- `min_segment_duration`: Minimum duration in seconds for segments (default: 0.5) - reduced to capture quick camera cuts

### Feature Extraction

- `model`: "clip" or "resnet50" (default: "clip")
- `batch_size`: Batch size for processing (default: 32)
- `image_size`: Input image size (default: 224)

### Clustering/Classification

- `method`: "ai_classifier", "dbscan", "kmeans", "hierarchical", or "hybrid" (default: "ai_classifier")
- `use_ai_classifier`: Use AI classification instead of clustering (default: true)
- `angle_descriptions`: Custom angle descriptions (null = use improved defaults with 20 specific baseball angles)
- `confidence_threshold`: Minimum confidence for AI classification (default: 0.2, range: 0.0-1.0)
- `clip_model`: CLIP model to use (default: "openai/clip-vit-base-patch32", try "openai/clip-vit-large-patch14" for better accuracy)
- `use_hybrid`: Use clustering to discover angles, then AI to name them (default: false)
- `eps`: DBSCAN epsilon parameter (default: 0.2) - only used if not using AI classifier
- `min_samples`: DBSCAN minimum samples (default: 5)
- `n_clusters`: Number of clusters for K-means/hierarchical (default: 8, null = auto-detect)
- `temporal_window`: Frames for temporal smoothing (default: 3)

### Output

- `base_dir`: Base output directory (default: "output")
- `create_visualization`: Create timeline visualization (default: true)

## Shared Video Directory

The system integrates with the shared `video_downloader/videos/` directory. Videos downloaded using `mlb_clip_downloader.py` are automatically available for processing.

## Examples

### Example 1: Process Single Video

When processing a single video, the system will still train on that video to discover angles:
```bash
python main.py --video "Corbin_Carrolls_RBI_single_747175.mp4"
```

### Example 2: Batch Process All Videos (Recommended)

**This is the recommended approach** - it trains on all videos to discover global camera angles, then applies them to each video:
```bash
python main.py --batch
```

This ensures consistent angle labeling across all videos and better discovery of common camera angles.

### Example 3: Use Custom Config

```bash
python main.py --video "video.mp4" --config custom_config.yaml
```

## Troubleshooting

### FFmpeg Not Found

If you see "FFmpeg not found" errors:
1. Install FFmpeg (see Installation section)
2. Verify installation: `ffmpeg -version`
3. Ensure FFmpeg is in your PATH

### CUDA/GPU Issues

The system automatically uses available hardware:
- CUDA (NVIDIA GPU)
- MPS (Apple Silicon)
- CPU (fallback)

To force CPU usage, modify `feature_extractor.py`:
```python
self.device = torch.device('cpu')
```

### Memory Issues

If running out of memory:
1. Reduce `batch_size` in config
2. Reduce `frame_extraction_rate` to extract fewer frames
3. Use ResNet50 instead of CLIP (smaller model)

### No Angles Discovered

If clustering finds no angles:
1. **Try AI Classifier**: Set `use_ai_classifier: true` and `method: "ai_classifier"` - this uses semantic understanding
2. Adjust `eps` parameter (try 0.2-0.5) for DBSCAN
3. Reduce `min_samples` for DBSCAN
4. Try different clustering method (kmeans with fixed n_clusters)

### Improving Classification Accuracy

1. **Use Better CLIP Model**: For improved accuracy, use a larger CLIP model:
   ```yaml
   clustering:
     clip_model: "openai/clip-vit-large-patch14"  # Larger, more accurate model
   ```
   Note: This uses more memory but provides better classification.

2. **Adjust Confidence Threshold**: Filter out uncertain classifications:
   ```yaml
   clustering:
     confidence_threshold: 0.3  # Higher = more strict (0.0-1.0)
   ```

3. **Custom Angle Descriptions**: Provide specific descriptions for your use case:
   ```yaml
   clustering:
     angle_descriptions:
       - "center field camera high angle view from behind home plate showing the entire baseball diamond"
       - "first base camera angle from the first base side showing the first baseman and batter"
       - "your specific camera angle description here"
   ```

4. **Use Hybrid Approach**: Combine clustering discovery with AI naming:
   ```yaml
   clustering:
     method: "hybrid"
     use_hybrid: true
   ```

### Default Angle Descriptions

The system now includes 20 improved angle descriptions covering:
- Main broadcast angles (center field, first/third base, home plate, pitcher)
- Infield angles (base lines, second base, shortstop)
- Outfield angles (left/right/center field)
- Special angles (dugout, aerial, replay, fan view)

These descriptions use more specific baseball terminology for better accuracy.

## Advanced Features

### Custom Angle Naming

The system can be extended to use CLIP for descriptive angle names (e.g., "center_field_view", "home_plate_closeup"). See `feature_extractor.py` for CLIP text encoding capabilities.

### Quality Filtering

Add blur detection or quality metrics before clustering to filter low-quality frames.

### Transition Detection

Enhance `video_segmenter.py` to detect smooth camera movements vs. hard cuts for better segmentation.

## License

Part of the BaseballVision project.

## Contributing

This system is designed to work with the shared `video_downloader` folder and integrates with other BaseballVision projects.
