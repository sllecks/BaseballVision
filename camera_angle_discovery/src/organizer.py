"""
Organizer - Organize video segments into subfolders by angle
Creates metadata and summary visualizations
"""

import json
import shutil
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict


class Organizer:
    """Organize video segments into subfolders by camera angle."""
    
    def __init__(self, config: dict):
        """
        Initialize organizer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.output_config = config.get('output', {})
        self.base_dir = Path(self.output_config.get('base_dir', 'output'))
        self.create_visualization = self.output_config.get('create_visualization', True)
    
    def organize(
        self,
        segments: List[Dict],
        video_name: str,
        angle_labels: Optional[np.ndarray] = None,
        timestamps: Optional[List[float]] = None
    ) -> Path:
        """
        Organize segments into subfolders by angle (per-video organization).
        
        Args:
            segments: List of segment dictionaries
            video_name: Name of source video
            angle_labels: Optional angle labels for visualization
            timestamps: Optional timestamps for visualization
            
        Returns:
            Path to output directory
        """
        print(f"📁 Organizing segments into folders...")
        
        # Create output structure
        output_dir = self.base_dir / video_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Group segments by angle
        segments_by_angle = defaultdict(list)
        for segment in segments:
            angle_id = segment['angle_id']
            segments_by_angle[angle_id].append(segment)
        
        # Create subfolders and move segments
        angle_metadata = {}
        for angle_id, angle_segments in segments_by_angle.items():
            angle_dir = output_dir / f"angle_{angle_id}"
            angle_dir.mkdir(exist_ok=True)
            
            # Move segments to angle folder
            for i, segment in enumerate(angle_segments):
                if 'file_path' in segment and segment['file_path'].exists():
                    # Create new filename
                    new_filename = f"segment_{i+1:03d}.mp4"
                    new_path = angle_dir / new_filename
                    
                    # Copy segment file
                    shutil.copy2(segment['file_path'], new_path)
                    segment['file_path'] = new_path
                    segment['relative_path'] = f"angle_{angle_id}/{new_filename}"
                    # Store video name for global organization
                    segment['video_name'] = video_name
            
            # Create metadata for this angle
            total_duration = sum(s['duration'] for s in angle_segments)
            angle_metadata[angle_id] = {
                'angle_id': angle_id,
                'n_segments': len(angle_segments),
                'total_duration': total_duration,
                'segments': [
                    {
                        'filename': s.get('relative_path', ''),
                        'start_time': s['start_time'],
                        'end_time': s['end_time'],
                        'duration': s['duration'],
                        'n_frames': s['n_frames']
                    }
                    for s in angle_segments
                ]
            }
        
        # Save metadata
        metadata_path = output_dir / "metadata.json"
        metadata = {
            'video_name': video_name,
            'n_angles': len(segments_by_angle),
            'total_segments': len(segments),
            'angles': angle_metadata
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✅ Organized {len(segments)} segments into {len(segments_by_angle)} angle folders")
        print(f"   Output directory: {output_dir}")
        
        # Create visualization if requested
        if self.create_visualization and angle_labels is not None and timestamps is not None:
            self._create_visualization(angle_labels, timestamps, output_dir, video_name)
        
        # Create summary
        self._create_summary(metadata, output_dir)
        
        return output_dir
    
    def _create_visualization(
        self,
        angle_labels: np.ndarray,
        timestamps: List[float],
        output_dir: Path,
        video_name: str
    ):
        """Create timeline visualization showing angle changes."""
        print(f"📊 Creating visualization...")
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))
        
        # Timeline plot
        unique_angles = np.unique(angle_labels)
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_angles)))
        angle_color_map = {angle: colors[i] for i, angle in enumerate(unique_angles)}
        
        # Plot angle timeline
        for i, (angle, timestamp) in enumerate(zip(angle_labels, timestamps)):
            ax1.barh(0, 0.1, left=timestamp, height=0.5, color=angle_color_map[angle], alpha=0.7)
        
        ax1.set_xlabel('Time (seconds)', fontsize=12)
        ax1.set_ylabel('Camera Angle', fontsize=12)
        ax1.set_title(f'Camera Angle Timeline - {video_name}', fontsize=14, fontweight='bold')
        ax1.set_yticks([])
        ax1.set_xlim(0, max(timestamps))
        ax1.grid(axis='x', alpha=0.3)
        
        # Create legend
        legend_elements = [
            mpatches.Patch(facecolor=angle_color_map[angle], label=f'Angle {angle}')
            for angle in sorted(unique_angles)
        ]
        ax1.legend(handles=legend_elements, loc='upper right')
        
        # Distribution plot
        angle_counts = [np.sum(angle_labels == angle) for angle in unique_angles]
        ax2.bar(range(len(unique_angles)), angle_counts, color=[angle_color_map[a] for a in unique_angles])
        ax2.set_xlabel('Camera Angle', fontsize=12)
        ax2.set_ylabel('Number of Frames', fontsize=12)
        ax2.set_title('Angle Distribution', fontsize=14, fontweight='bold')
        ax2.set_xticks(range(len(unique_angles)))
        ax2.set_xticklabels([f'Angle {angle}' for angle in unique_angles])
        ax2.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        # Save visualization
        viz_path = output_dir / "angle_timeline.png"
        plt.savefig(viz_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"   Saved visualization: {viz_path}")
    
    def _create_summary(self, metadata: Dict, output_dir: Path):
        """Create text summary of results."""
        summary_path = output_dir / "summary.txt"
        
        with open(summary_path, 'w') as f:
            f.write(f"Camera Angle Discovery Summary\n")
            f.write(f"{'=' * 50}\n\n")
            f.write(f"Video: {metadata['video_name']}\n")
            f.write(f"Total Angles Discovered: {metadata['n_angles']}\n")
            f.write(f"Total Segments: {metadata['total_segments']}\n\n")
            
            f.write("Angle Details:\n")
            f.write("-" * 50 + "\n")
            
            for angle_id in sorted(metadata['angles'].keys()):
                angle_info = metadata['angles'][angle_id]
                f.write(f"\nAngle {angle_id}:\n")
                f.write(f"  Segments: {angle_info['n_segments']}\n")
                f.write(f"  Total Duration: {angle_info['total_duration']:.2f}s\n")
                f.write(f"  Segment Files:\n")
                for seg in angle_info['segments']:
                    f.write(f"    - {seg['filename']} ({seg['duration']:.2f}s)\n")
        
        print(f"   Saved summary: {summary_path}")
    
    def organize_by_angle_global(
        self,
        all_segments: List[Dict],
        output_dir: Optional[Path] = None
    ) -> Path:
        """
        Organize all segments from all videos into global angle folders.
        Each angle gets its own folder with segments from all videos.
        
        Args:
            all_segments: List of all segment dictionaries from all videos
            output_dir: Optional output directory (default: base_dir/angles_collected)
            
        Returns:
            Path to output directory
        """
        if output_dir is None:
            output_dir = self.base_dir / "angles_collected"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"📁 Organizing all segments by angle across all videos...")
        
        # Group segments by angle across all videos
        segments_by_angle = defaultdict(list)
        for segment in all_segments:
            angle_id = segment['angle_id']
            segments_by_angle[angle_id].append(segment)
        
        # Create angle folders and copy segments
        global_metadata = {
            'n_angles': len(segments_by_angle),
            'total_segments': len(all_segments),
            'angles': {}
        }
        
        for angle_id, angle_segments in segments_by_angle.items():
            angle_dir = output_dir / f"angle_{angle_id}"
            angle_dir.mkdir(exist_ok=True)
            
            # Copy segments to angle folder with video name prefix
            angle_segment_info = []
            for segment in angle_segments:
                if 'file_path' in segment and segment['file_path'].exists():
                    # Get video name from segment metadata or file path
                    video_name = segment.get('video_name', 'unknown')
                    segment_num = len(angle_segment_info) + 1
                    
                    # Create filename with video name and segment info
                    safe_video_name = "".join(c for c in video_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
                    safe_video_name = safe_video_name.replace(' ', '_')[:50]
                    new_filename = f"{safe_video_name}_seg_{segment_num:03d}.mp4"
                    new_path = angle_dir / new_filename
                    
                    # Copy segment file
                    shutil.copy2(segment['file_path'], new_path)
                    
                    angle_segment_info.append({
                        'filename': new_filename,
                        'video_name': video_name,
                        'start_time': segment['start_time'],
                        'end_time': segment['end_time'],
                        'duration': segment['duration'],
                        'n_frames': segment.get('n_frames', 0)
                    })
            
            # Create metadata for this angle
            total_duration = sum(s['duration'] for s in angle_segments)
            unique_videos = set(s.get('video_name', 'unknown') for s in angle_segments)
            global_metadata['angles'][angle_id] = {
                'angle_id': angle_id,
                'n_segments': len(angle_segments),
                'total_duration': total_duration,
                'n_videos': len(unique_videos),
                'videos': sorted(list(unique_videos)),
                'segments': angle_segment_info
            }
        
        # Save global metadata
        metadata_path = output_dir / "global_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(global_metadata, f, indent=2)
        
        # Create global summary
        self._create_global_summary(global_metadata, output_dir)
        
        print(f"✅ Organized {len(all_segments)} segments from all videos into {len(segments_by_angle)} angle folders")
        print(f"   Output directory: {output_dir}")
        
        return output_dir
    
    def _create_global_summary(self, metadata: Dict, output_dir: Path):
        """Create text summary of global angle organization."""
        summary_path = output_dir / "global_summary.txt"
        
        with open(summary_path, 'w') as f:
            f.write(f"Global Camera Angle Collection Summary\n")
            f.write(f"{'=' * 60}\n\n")
            f.write(f"Total Angles: {metadata['n_angles']}\n")
            f.write(f"Total Segments: {metadata['total_segments']}\n\n")
            
            f.write("Angle Details:\n")
            f.write("-" * 60 + "\n")
            
            for angle_id in sorted(metadata['angles'].keys()):
                angle_info = metadata['angles'][angle_id]
                f.write(f"\nAngle {angle_id}:\n")
                f.write(f"  Total Segments: {angle_info['n_segments']}\n")
                f.write(f"  Total Duration: {angle_info['total_duration']:.2f}s\n")
                f.write(f"  Number of Videos: {angle_info['n_videos']}\n")
                f.write(f"  Segment Files:\n")
                for seg in angle_info['segments']:
                    f.write(f"    - {seg['filename']} (from {seg['video_name']}, {seg['duration']:.2f}s)\n")
        
        print(f"   Saved global summary: {summary_path}")
