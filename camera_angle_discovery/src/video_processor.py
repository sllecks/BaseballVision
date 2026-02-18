"""
Video Processor - Extract frames from video files
Integrates with shared video_downloader/videos directory
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
import sys


class VideoProcessor:
    """Process video files and extract frames for analysis."""
    
    def __init__(self, config: dict, videos_dir: Optional[Path] = None):
        """
        Initialize video processor.
        
        Args:
            config: Configuration dictionary with video settings
            videos_dir: Optional path to shared videos directory
        """
        self.config = config
        self.video_config = config.get('video', {})
        self.frame_rate = self.video_config.get('frame_extraction_rate', 1.0)
        
        # Set up videos directory
        if videos_dir is None:
            # Try to import from shared video_downloader
            try:
                VIDEO_DOWNLOADER_DIR = Path(__file__).parent.parent.parent / "video_downloader"
                sys.path.insert(0, str(VIDEO_DOWNLOADER_DIR))
                from mlb_clip_downloader import VIDEOS_DIR
                self.videos_dir = VIDEOS_DIR
            except ImportError:
                # Fallback to config path
                videos_dir_path = self.video_config.get('videos_dir', '../video_downloader/videos')
                self.videos_dir = Path(__file__).parent.parent.parent / videos_dir_path
        else:
            self.videos_dir = Path(videos_dir)
    
    def get_video_path(self, video_input: Optional[str] = None) -> Path:
        """
        Get video path from input or shared directory.
        
        Args:
            video_input: Video filename or full path
            
        Returns:
            Path to video file
        """
        input_path = self.video_config.get('input_path')
        
        # If video_input is provided, use it
        if video_input:
            video_path = Path(video_input)
            if video_path.is_absolute() or video_path.exists():
                return video_path
            # Try relative to videos_dir
            video_path = self.videos_dir / video_input
            if video_path.exists():
                return video_path
        
        # If input_path is set in config
        if input_path:
            video_path = Path(input_path)
            if video_path.is_absolute() or video_path.exists():
                return video_path
            # Try relative to videos_dir
            video_path = self.videos_dir / input_path
            if video_path.exists():
                return video_path
        
        # Default: list videos from shared directory
        raise ValueError(
            f"No video specified. Available videos in {self.videos_dir}:\n" +
            "\n".join([f"  - {v.name}" for v in self.videos_dir.glob("*.mp4")[:10]])
        )
    
    def list_available_videos(self) -> List[Path]:
        """List all available videos in the shared videos directory."""
        if not self.videos_dir.exists():
            return []
        
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
        videos = []
        for ext in video_extensions:
            videos.extend(self.videos_dir.glob(f"*{ext}"))
        
        return sorted(videos)
    
    def extract_frames(self, video_path: Optional[Path] = None) -> Tuple[List[np.ndarray], List[float]]:
        """
        Extract frames from video at specified intervals.
        
        Args:
            video_path: Path to video file (optional, uses config if not provided)
            
        Returns:
            Tuple of (frames list, timestamps list)
        """
        if video_path is None:
            video_path = self.get_video_path()
        
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        print(f"📹 Processing video: {video_path.name}")
        
        # Open video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        
        print(f"   FPS: {fps:.2f}, Duration: {duration:.2f}s, Total frames: {total_frames}")
        
        # Calculate frame interval
        frame_interval = int(fps / self.frame_rate) if fps > 0 else 1
        if frame_interval < 1:
            frame_interval = 1
        
        frames = []
        timestamps = []
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Extract frame at specified intervals
            if frame_count % frame_interval == 0:
                timestamp = frame_count / fps if fps > 0 else frame_count
                frames.append(frame.copy())
                timestamps.append(timestamp)
            
            frame_count += 1
        
        cap.release()
        
        print(f"✅ Extracted {len(frames)} frames at {self.frame_rate} fps")
        
        return frames, timestamps
    
    def get_video_info(self, video_path: Optional[Path] = None) -> dict:
        """
        Get video metadata.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Dictionary with video information
        """
        if video_path is None:
            video_path = self.get_video_path()
        
        video_path = Path(video_path)
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        info = {
            'path': str(video_path),
            'name': video_path.name,
            'fps': cap.get(cv2.CAP_PROP_FPS),
            'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        }
        
        if info['fps'] > 0:
            info['duration'] = info['frame_count'] / info['fps']
        else:
            info['duration'] = 0
        
        cap.release()
        
        return info
