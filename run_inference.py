"""
Simple script to run inference on baseball videos.
Usage: python run_inference.py <video_path> [--conf CONFIDENCE] [--weights WEIGHTS_PATH]
"""

import argparse
import sys
from pathlib import Path
from main import predict, BASE_DIR

def main():
    parser = argparse.ArgumentParser(
        description="Run YOLO inference on baseball videos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run inference on a video with default settings
  python run_inference.py videos/my_video.mp4
  
  # Run with custom confidence threshold
  python run_inference.py videos/my_video.mp4 --conf 0.3
  
  # Run with custom model weights
  python run_inference.py videos/my_video.mp4 --weights runs/baseball_detect/weights/best.pt
        """
    )
    
    parser.add_argument(
        "video_path",
        type=str,
        help="Path to the video file to run inference on"
    )
    
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold (0.0-1.0). Default: 0.25"
    )
    
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Path to model weights file. Default: runs/baseball_detect/weights/best.pt"
    )
    
    args = parser.parse_args()
    
    # Validate video path
    video_path = Path(args.video_path)
    if not video_path.is_absolute():
        video_path = BASE_DIR / video_path
    
    if not video_path.exists():
        print(f"❌ Error: Video file not found: {video_path}")
        sys.exit(1)
    
    if not video_path.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv', '.webm']:
        print(f"⚠️  Warning: File extension '{video_path.suffix}' may not be a video file")
    
    # Validate confidence
    if not 0.0 <= args.conf <= 1.0:
        print(f"❌ Error: Confidence must be between 0.0 and 1.0")
        sys.exit(1)
    
    print("🎯 Baseball Vision - Video Inference")
    print("=" * 50)
    print(f"📹 Video: {video_path.name}")
    print(f"🎚️  Confidence threshold: {args.conf}")
    if args.weights:
        print(f"⚖️  Weights: {args.weights}")
    else:
        print(f"⚖️  Weights: Using default (runs/baseball_detect/weights/best.pt)")
    print("=" * 50)
    print("\n🔍 Running inference...\n")
    
    try:
        results = predict(
            source=str(video_path),
            weights_path=args.weights,
            conf=args.conf
        )
        
        if results:
            output_path = BASE_DIR / "runs" / "predictions" / video_path.name
            print("\n" + "=" * 50)
            print("✅ Inference complete!")
            print(f"📁 Output saved to: {output_path}")
            print("=" * 50)
        else:
            print("\n❌ Inference failed. Check error messages above.")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n\n⚠️  Inference interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error during inference: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()

