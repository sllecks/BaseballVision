"""
Video Segmenter - Segment video based on discovered camera angles
Uses ffmpeg or moviepy for video cutting
"""

import subprocess
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np


class VideoSegmenter:
    """Segment video into clips based on discovered camera angles."""
    
    def __init__(self, config: dict):
        """
        Initialize video segmenter.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.video_config = config.get('video', {})
        self.min_segment_duration = self.video_config.get('min_segment_duration', 2.0)
        self.output_config = config.get('output', {})
        self.base_dir = Path(self.output_config.get('base_dir', 'output'))
    
    def create_segments(
        self,
        angle_labels: np.ndarray,
        timestamps: List[float],
        video_path: Path,
        output_dir: Optional[Path] = None
    ) -> List[Dict]:
        """
        Create video segments for each discovered angle.
        
        Args:
            angle_labels: Angle label for each frame
            timestamps: Timestamp for each frame
            video_path: Path to source video
            output_dir: Output directory for segments (optional)
            
        Returns:
            List of segment dictionaries with metadata
        """
        if output_dir is None:
            output_dir = self.base_dir / "segments"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"✂️  Creating video segments...")
        
        # Group consecutive frames with same angle
        segments = self._group_consecutive_segments(angle_labels, timestamps)
        
        # Filter segments by minimum duration
        segments = [s for s in segments if s['duration'] >= self.min_segment_duration]
        
        print(f"   Found {len(segments)} segments (min duration: {self.min_segment_duration}s)")
        
        # Extract video segments
        segment_files = []
        for i, segment in enumerate(segments):
            segment_file = self._extract_segment(
                video_path,
                segment,
                output_dir,
                segment_index=i
            )
            if segment_file:
                segment['file_path'] = segment_file
                segment_files.append(segment)
        
        print(f"✅ Created {len(segment_files)} video segments")
        
        return segment_files
    
    def _group_consecutive_segments(
        self,
        angle_labels: np.ndarray,
        timestamps: List[float]
    ) -> List[Dict]:
        """
        Group consecutive frames with the same angle into segments.
        
        Args:
            angle_labels: Angle labels for each frame
            timestamps: Timestamps for each frame
            
        Returns:
            List of segment dictionaries
        """
        segments = []
        current_angle = angle_labels[0]
        current_start = 0
        
        for i in range(1, len(angle_labels)):
            if angle_labels[i] != current_angle:
                # End of current segment
                segment = {
                    'angle_id': int(current_angle),
                    'start_frame': current_start,
                    'end_frame': i - 1,
                    'start_time': timestamps[current_start],
                    'end_time': timestamps[i - 1],
                    'duration': timestamps[i - 1] - timestamps[current_start],
                    'n_frames': i - current_start
                }
                segments.append(segment)
                
                # Start new segment
                current_angle = angle_labels[i]
                current_start = i
        
        # Add final segment
        segment = {
            'angle_id': int(current_angle),
            'start_frame': current_start,
            'end_frame': len(angle_labels) - 1,
            'start_time': timestamps[current_start],
            'end_time': timestamps[-1],
            'duration': timestamps[-1] - timestamps[current_start],
            'n_frames': len(angle_labels) - current_start
        }
        segments.append(segment)
        
        return segments
    
    def _extract_segment(
        self,
        video_path: Path,
        segment: Dict,
        output_dir: Path,
        segment_index: int
    ) -> Optional[Path]:
        """
        Extract a single video segment using ffmpeg.
        
        Args:
            video_path: Path to source video
            segment: Segment metadata dictionary
            output_dir: Output directory
            segment_index: Index of segment
            
        Returns:
            Path to extracted segment file, or None if failed
        """
        angle_id = segment['angle_id']
        start_time = segment['start_time']
        duration = segment['duration']
        
        # Create filename
        video_name = video_path.stem
        segment_filename = f"{video_name}_angle_{angle_id}_seg_{segment_index:03d}.mp4"
        output_path = output_dir / segment_filename
        
        # Use ffmpeg to extract segment
        try:
            cmd = [
                'ffmpeg',
                '-i', str(video_path),
                '-ss', str(start_time),
                '-t', str(duration),
                '-c', 'copy',  # Copy codec (fast, no re-encoding)
                '-avoid_negative_ts', 'make_zero',
                '-y',  # Overwrite output file
                str(output_path)
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            
            if output_path.exists():
                return output_path
            else:
                print(f"   ⚠️  Segment extraction failed: {segment_filename}")
                return None
        
        except subprocess.CalledProcessError as e:
            print(f"   ⚠️  FFmpeg error for segment {segment_index}: {e.stderr}")
            return None
        except FileNotFoundError:
            print("   ⚠️  FFmpeg not found. Please install ffmpeg.")
            print("   On macOS: brew install ffmpeg")
            print("   On Ubuntu: sudo apt-get install ffmpeg")
            return None
    
    def check_ffmpeg_available(self) -> bool:
        """Check if ffmpeg is available."""
        try:
            subprocess.run(
                ['ffmpeg', '-version'],
                capture_output=True,
                check=True
            )
            return True
        except (FileNotFoundError, subprocess.CalledProcessError):
            return False
