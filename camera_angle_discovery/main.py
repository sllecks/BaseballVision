"""
Camera Angle Discovery System - Main Entry Point
Automatically discovers and categorizes camera angles in baseball game videos
"""

import os
# Suppress Hugging Face tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import yaml
from pathlib import Path
import sys
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from src.video_processor import VideoProcessor
from src.feature_extractor import FeatureExtractor
from src.angle_clusterer import AngleClusterer
from src.angle_classifier import AngleClassifier
from src.video_segmenter import VideoSegmenter
from src.organizer import Organizer


def load_config(config_path: Path = None) -> dict:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config file (default: config.yaml in project root)
        
    Returns:
        Configuration dictionary
    """
    if config_path is None:
        config_path = PROJECT_ROOT / "config.yaml"
    
    if not config_path.exists():
        print(f"⚠️  Config file not found: {config_path}")
        print("   Using default configuration...")
        return get_default_config()
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def get_default_config() -> dict:
    """Get default configuration."""
    return {
        'video': {
            'input_path': None,
            'videos_dir': '../video_downloader/videos',
            'frame_extraction_rate': 5.0,  # Increased for better temporal resolution
            'min_segment_duration': 0.5  # Reduced to capture quick cuts
        },
        'features': {
            'model': 'clip',
            'batch_size': 32,
            'image_size': 224
        },
        'clustering': {
            'method': 'kmeans',  # kmeans works better for multiple angles
            'min_samples': 5,
            'eps': 0.2,
            'n_clusters': 8,  # Default to 8 angles
            'temporal_window': 3  # Reduced to preserve quick cuts
        },
        'output': {
            'base_dir': 'output',
            'save_frames': False,
            'create_visualization': True
        }
    }


def process_video(
    video_input: str = None,
    config: dict = None,
    batch_mode: bool = False
):
    """
    Process a video or batch of videos.
    Uses global clustering across all videos to discover common camera angles.
    
    Args:
        video_input: Video filename or path (optional)
        config: Configuration dictionary (optional)
        batch_mode: If True, process all videos in shared directory
    """
    if config is None:
        config = load_config()
    
    # Initialize components
    processor = VideoProcessor(config)
    extractor = FeatureExtractor(config)
    clusterer = AngleClusterer(config)
    
    # Get video(s) to process
    if batch_mode:
        videos = processor.list_available_videos()
        if not videos:
            print("❌ No videos found in shared directory")
            return
        print(f"📹 Found {len(videos)} video(s) to process\n")
    else:
        if video_input:
            video_path = processor.get_video_path(video_input)
        else:
            video_path = processor.get_video_path()
        videos = [video_path]
    
    # Determine which method to use
    clustering_config = config.get('clustering', {})
    use_ai_classifier = clustering_config.get('use_ai_classifier', False)
    method = clustering_config.get('method', 'kmeans').lower()
    
    # Initialize classifier/clusterer
    if use_ai_classifier or method == 'ai_classifier':
        print("🤖 Using AI classifier mode")
        classifier = AngleClassifier(config)
        clusterer = None
        print(f"   AI classifier will identify {len(classifier.get_angle_names())} angle types")
    else:
        print("🔍 Using clustering mode - will train on all videos")
        classifier = None
        # For clustering, we need to collect features first
        all_features_list = []
        
        # PHASE 1: Extract features from ALL videos for clustering
        print(f"\n{'=' * 60}")
        print("PHASE 1: Extracting Features for Global Clustering")
        print(f"{'=' * 60}\n")
        
        for video_path in videos:
            print(f"📹 Extracting features from: {video_path.name}")
            try:
                # Extract frames
                frames, timestamps = processor.extract_frames(video_path)
                
                if len(frames) == 0:
                    print(f"   ⚠️  No frames extracted. Skipping...")
                    continue
                
                # Extract features
                features = extractor.extract(frames)
                
                all_features_list.append(features)
                print(f"   ✅ Extracted {len(features)} feature vectors")
                
                # Free frames from memory
                del frames
                
            except Exception as e:
                print(f"   ❌ Error processing {video_path.name}: {e}")
                continue
        
        if len(all_features_list) == 0:
            print("❌ No features extracted from any video")
            return
        
        # Combine all features
        all_features = np.vstack(all_features_list)
        print(f"\n📊 Total features collected: {all_features.shape[0]} from {len(videos)} video(s)")
        
        # Train global clustering model
        clusterer.fit_global_angles(all_features)
        
        # Store clusterer for later use (it has the trained model)
        trained_clusterer = clusterer
        
        # Free combined features from memory (we'll re-extract per video)
        del all_features, all_features_list
    
    # PHASE 2: Process each video independently (memory efficient)
    print(f"\n{'=' * 60}")
    print("PHASE 2: Processing Videos and Creating Segments")
    print(f"{'=' * 60}\n")
    
    # Collect all segments for global organization
    all_segments_global = []
    
    # Process each video independently to save memory
    for video_path in videos:
        print(f"\n{'=' * 60}")
        print(f"Processing: {video_path.name}")
        print(f"{'=' * 60}\n")
        
        try:
            # Extract frames (process immediately, don't store)
            frames, timestamps = processor.extract_frames(video_path)
            
            if len(frames) == 0:
                print("⚠️  No frames extracted. Skipping...")
                continue
            
            # Extract features
            video_features = extractor.extract(frames)
            
            # Free frames from memory immediately
            del frames
            
            # Classify/assign angles
            if use_ai_classifier or method == 'ai_classifier' or method == 'hybrid':
                # Use AI classifier
                if method == 'hybrid':
                    print("🔄 Using hybrid approach: clustering + AI naming...")
                    # First cluster to discover angles
                    temp_clusterer = AngleClusterer(config)
                    temp_clusterer.fit_global_angles(video_features)
                    angle_labels = temp_clusterer.predict_angles(video_features, timestamps)
                    # Then use AI to get better names (optional - for now just use clustering)
                else:
                    print("🤖 Using AI angle classifier...")
                    angle_labels = classifier.classify_features(video_features)
                
                # Apply temporal smoothing
                temp_clusterer = AngleClusterer(config)
                angle_labels = temp_clusterer.apply_temporal_smoothing(angle_labels, timestamps)
            else:
                # Use clustering approach
                angle_labels = trained_clusterer.predict_angles(video_features, timestamps)
            
            # Free features from memory
            del video_features
            
            # Get statistics
            temp_clusterer = AngleClusterer(config)
            stats = temp_clusterer.get_angle_statistics(angle_labels, timestamps)
            print(f"\n📊 Angle Statistics:")
            for angle_id, angle_info in stats['angles'].items():
                print(f"   Angle {angle_id}: {angle_info['n_frames']} frames "
                      f"({angle_info['percentage']:.1f}%), "
                      f"{angle_info['duration']:.2f}s")
            
            # Create segments
            segmenter = VideoSegmenter(config)
            segments = segmenter.create_segments(
                angle_labels,
                timestamps,
                video_path
            )
            
            if not segments:
                print("⚠️  No segments created. Skipping organization...")
                continue
            
            # Organize segments per video
            organizer = Organizer(config)
            output_dir = organizer.organize(
                segments,
                video_path.stem,
                angle_labels,
                timestamps
            )
            
            # Add segments to global collection
            for segment in segments:
                segment['video_name'] = video_path.stem
            all_segments_global.extend(segments)
            
            # Free labels from memory
            del angle_labels, timestamps
            
            print(f"\n✅ Processing complete!")
            print(f"   Output directory: {output_dir}")
            
        except Exception as e:
            print(f"\n❌ Error processing {video_path.name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # PHASE 3: Organize all segments by angle globally
    if all_segments_global:
        print(f"\n{'=' * 60}")
        print("PHASE 3: Global Angle Organization")
        print(f"{'=' * 60}\n")
        
        organizer = Organizer(config)
        global_output_dir = organizer.organize_by_angle_global(all_segments_global)
        
        print(f"\n✅ Global organization complete!")
        print(f"   All segments organized by angle in: {global_output_dir}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Camera Angle Discovery System - Automatically discover and categorize camera angles in baseball videos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process a specific video from shared directory
  python main.py --video "Alek_Thomas_game-tying_two-run_single_747175.mp4"
  
  # Process all videos in shared directory
  python main.py --batch
  
  # Process a custom video path
  python main.py --video-path "/path/to/custom/video.mp4"
  
  # Use custom config file
  python main.py --video "video.mp4" --config custom_config.yaml
        """
    )
    
    parser.add_argument(
        '--video',
        type=str,
        default=None,
        help='Video filename from shared directory or relative path'
    )
    
    parser.add_argument(
        '--video-path',
        type=str,
        default=None,
        help='Full path to video file'
    )
    
    parser.add_argument(
        '--batch',
        action='store_true',
        help='Process all videos in shared directory'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to config YAML file (default: config.yaml)'
    )
    
    args = parser.parse_args()
    
    # Load config
    config_path = Path(args.config) if args.config else None
    config = load_config(config_path)
    
    # Determine video input
    video_input = args.video_path or args.video
    
    # Process video(s)
    process_video(
        video_input=video_input,
        config=config,
        batch_mode=args.batch
    )


if __name__ == "__main__":
    main()
