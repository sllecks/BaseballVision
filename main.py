"""
Baseball Vision - YOLO Training Script
Trains a YOLOv8 model to detect: Ball, Catcher's glove, Batter's bat, Homeplate
"""

import os
import shutil
import random
from pathlib import Path
from ultralytics import YOLO

# Configuration
BASE_DIR = Path(__file__).parent
FRAMES_DIR = BASE_DIR / "Frames"
ANNOTATION_DIRS = [BASE_DIR / "annotation 1", BASE_DIR / "annotation 2"]
DATASET_DIR = BASE_DIR / "dataset"
VAL_SPLIT = 0.2  # 20% for validation

# Class names (from obj.names)
CLASS_NAMES = ["Ball", "Catcher's glove", "Batter's bat", "Homeplate"]


def validate_annotation(label_file, max_objects=100):
    """Validate annotation file and count objects."""
    try:
        with open(label_file, 'r') as f:
            lines = f.readlines()
        
        valid_lines = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 5:
                return False, 0  # Invalid format
            try:
                class_id = int(parts[0])
                if class_id < 0 or class_id >= len(CLASS_NAMES):
                    return False, 0  # Invalid class ID
                # Validate coordinates are in [0, 1] range
                coords = [float(x) for x in parts[1:]]
                if any(x < 0 or x > 1 for x in coords):
                    return False, 0  # Coordinates out of range
                valid_lines.append(line)
            except ValueError:
                return False, 0  # Invalid number format
        
        num_objects = len(valid_lines)
        if num_objects > max_objects:
            return False, num_objects  # Too many objects
        
        return True, num_objects
    except Exception as e:
        return False, 0


def prepare_dataset(max_objects_per_image=100):
    """Organize images and labels into YOLO dataset structure."""
    print("🔧 Preparing dataset...")
    
    # Create dataset directories
    train_images = DATASET_DIR / "train" / "images"
    train_labels = DATASET_DIR / "train" / "labels"
    val_images = DATASET_DIR / "val" / "images"
    val_labels = DATASET_DIR / "val" / "labels"
    
    for d in [train_images, train_labels, val_images, val_labels]:
        d.mkdir(parents=True, exist_ok=True)
    
    # Collect all annotation files from both annotation directories
    label_files = []
    for ann_dir in ANNOTATION_DIRS:
        obj_train_data = ann_dir / "obj_train_data"
        if obj_train_data.exists():
            label_files.extend(list(obj_train_data.glob("*.txt")))
    
    print(f"📁 Found {len(label_files)} annotation files")
    
    # Find matching images
    all_images = list(FRAMES_DIR.glob("*.jpg")) + list(FRAMES_DIR.glob("*.png"))
    # Also check _More subdirectory if it exists
    more_dir = FRAMES_DIR / "_More"
    if more_dir.exists():
        all_images.extend(list(more_dir.glob("*.jpg")) + list(more_dir.glob("*.png")))
    
    image_dict = {img.stem: img for img in all_images}
    print(f"🖼️  Found {len(image_dict)} images")
    
    # Match labels to images and validate annotations
    valid_pairs = []
    invalid_count = 0
    too_many_objects = 0
    for label_file in label_files:
        stem = label_file.stem
        if stem in image_dict:
            is_valid, num_objects = validate_annotation(label_file, max_objects_per_image)
            if is_valid:
                valid_pairs.append((image_dict[stem], label_file))
            elif num_objects > max_objects_per_image:
                too_many_objects += 1
            else:
                invalid_count += 1
    
    print(f"✅ Matched {len(valid_pairs)} valid image-label pairs")
    if invalid_count > 0:
        print(f"⚠️  Skipped {invalid_count} invalid annotation files")
    if too_many_objects > 0:
        print(f"⚠️  Skipped {too_many_objects} images with >{max_objects_per_image} objects")
    
    if len(valid_pairs) == 0:
        print("❌ No matching image-label pairs found!")
        print("   Sample label stems:", [f.stem for f in label_files[:5]])
        print("   Sample image stems:", list(image_dict.keys())[:5])
        return False
    
    # Shuffle and split into train/val
    random.seed(42)
    random.shuffle(valid_pairs)
    
    split_idx = int(len(valid_pairs) * (1 - VAL_SPLIT))
    train_pairs = valid_pairs[:split_idx]
    val_pairs = valid_pairs[split_idx:]
    
    print(f"📊 Split: {len(train_pairs)} train, {len(val_pairs)} val")
    
    # Copy files to dataset directories
    def copy_pair(img_path, label_path, img_dest, label_dest):
        shutil.copy2(img_path, img_dest / img_path.name)
        shutil.copy2(label_path, label_dest / (img_path.stem + ".txt"))
    
    for img, label in train_pairs:
        copy_pair(img, label, train_images, train_labels)
    
    for img, label in val_pairs:
        copy_pair(img, label, val_images, val_labels)
    
    # Create dataset.yaml
    # Use relative path from project root (where script is located)
    dataset_relative_path = str(DATASET_DIR.relative_to(BASE_DIR))
    yaml_content = f"""# Baseball Vision Dataset
path: {dataset_relative_path}
train: train/images
val: val/images

# Classes
names:
  0: Ball
  1: "Catcher's glove"
  2: "Batter's bat"
  3: Homeplate

nc: 4
"""
    
    yaml_path = DATASET_DIR / "dataset.yaml"
    yaml_path.write_text(yaml_content)
    yaml_relative = str(yaml_path.relative_to(BASE_DIR))
    print(f"📝 Created {yaml_relative}")
    
    return True


def train_model(
    model_size: str = "n",  # n, s, m, l, x (nano to extra-large)
    epochs: int = 100,
    imgsz: int = 640,
    batch: int = 8,  # Reduced from 16 to avoid shape mismatch errors
    resume: bool = False,
    device: str = "mps"  # Use Apple Metal GPU by default
):
    """Train YOLOv8 model on the baseball dataset."""
    
    # Change to project root directory so relative paths in YAML work correctly
    original_cwd = os.getcwd()
    os.chdir(BASE_DIR)
    
    try:
        yaml_path = DATASET_DIR / "dataset.yaml"
        
        if not yaml_path.exists():
            print("❌ Dataset not prepared. Run prepare_dataset() first.")
            return None
        
        # Load pretrained YOLOv8 model
        model_name = f"yolov8{model_size}.pt"
        print(f"🚀 Loading YOLOv8{model_size.upper()} model...")
        model = YOLO(model_name)
        
        # Train the model
        print(f"🏋️ Training for {epochs} epochs on {device.upper()}...")
        # Use relative paths
        yaml_relative = str(yaml_path.relative_to(BASE_DIR))
        project_relative = "runs"
        results = model.train(
            data=yaml_relative,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            device=device,  # Use MPS (Apple Metal) for faster training
            project=project_relative,
            name="baseball_detect",
            exist_ok=True,
            resume=resume,
            # Augmentation settings good for sports/action detection
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            degrees=5.0,
            translate=0.1,
            scale=0.3,
            fliplr=0.5,
            mosaic=0.5,  # Reduced from 1.0 to avoid shape mismatch with many objects
            mixup=0.1,
            # Performance
            patience=20,
            save=True,
            plots=True,
            verbose=True,
        )
        
        print("✅ Training complete!")
        print(f"📁 Results saved to: runs/baseball_detect")
        
        return results
    finally:
        # Restore original working directory
        os.chdir(original_cwd)


def validate_model(weights_path: str = None):
    """Validate the trained model."""
    # Change to project root directory so relative paths work correctly
    original_cwd = os.getcwd()
    os.chdir(BASE_DIR)
    
    try:
        if weights_path is None:
            weights_path = BASE_DIR / "runs" / "baseball_detect" / "weights" / "best.pt"
        else:
            # Convert to Path if it's a string
            weights_path = Path(weights_path)
            if not weights_path.is_absolute():
                weights_path = BASE_DIR / weights_path
        
        if not weights_path.exists():
            print(f"❌ Weights not found at {weights_path}")
            return None
        
        model = YOLO(str(weights_path))
        yaml_path = DATASET_DIR / "dataset.yaml"
        # Use relative path
        yaml_relative = str(yaml_path.relative_to(BASE_DIR))
        
        results = model.val(data=yaml_relative)
        return results
    finally:
        # Restore original working directory
        os.chdir(original_cwd)


def predict(source, weights_path: str = None, conf: float = 0.25):
    """Run inference on images or video."""
    # Change to project root directory so relative paths work correctly
    original_cwd = os.getcwd()
    os.chdir(BASE_DIR)
    
    try:
        if weights_path is None:
            weights_path = BASE_DIR / "runs" / "baseball_detect" / "weights" / "best.pt"
        else:
            # Convert to Path if it's a string
            weights_path = Path(weights_path)
            if not weights_path.is_absolute():
                weights_path = BASE_DIR / weights_path
        
        if not weights_path.exists():
            print(f"❌ Weights not found at {weights_path}")
            return None
        
        model = YOLO(str(weights_path))
        # Use relative path for project
        project_relative = "runs"
        results = model.predict(
            source=source,
            conf=conf,
            save=True,
            project=project_relative,
            name="predictions",
            exist_ok=True,
        )
        return results
    finally:
        # Restore original working directory
        os.chdir(original_cwd)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Baseball Vision YOLO Training")
    parser.add_argument("--prepare", action="store_true", help="Prepare dataset")
    parser.add_argument("--train", action="store_true", help="Train model")
    parser.add_argument("--validate", action="store_true", help="Validate model")
    parser.add_argument("--predict", type=str, help="Run prediction on source")
    parser.add_argument("--model", type=str, default="n", choices=["n", "s", "m", "l", "x"],
                       help="Model size: n(ano), s(mall), m(edium), l(arge), x(tra-large)")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--batch", type=int, default=8, help="Batch size (default: 8 to avoid shape mismatch errors)")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--resume", action="store_true", help="Resume training")
    parser.add_argument("--device", type=str, default="mps", help="Device: mps (Apple Metal), cpu, or cuda")
    
    args = parser.parse_args()
    
    if args.prepare:
        prepare_dataset()
    
    if args.train:
        if not (DATASET_DIR / "dataset.yaml").exists():
            print("Dataset not found. Preparing...")
            if not prepare_dataset():
                exit(1)
        train_model(
            model_size=args.model,
            epochs=args.epochs,
            batch=args.batch,
            imgsz=args.imgsz,
            resume=args.resume,
            device=args.device,
        )
    
    if args.validate:
        validate_model()
    
    if args.predict:
        predict(args.predict)
    
    # If no arguments, run both prepare and train
    if not any([args.prepare, args.train, args.validate, args.predict]):
        print("Baseball Vision - YOLO Training")
        print("=" * 40)
        print("\nUsage:")
        print("  python main.py --prepare          # Prepare dataset")
        print("  python main.py --train            # Train model")
        print("  python main.py --train --model s  # Train with YOLOv8s")
        print("  python main.py --validate         # Validate trained model")
        print("  python main.py --predict image.jpg  # Run inference")
        print("\nOr run both prepare and train:")
        print("  python main.py --prepare --train")

