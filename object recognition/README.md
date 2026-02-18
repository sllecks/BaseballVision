# Object Recognition

A YOLOv8-based object detection system for identifying baseball objects in videos and images. The system can detect four key objects: **Ball**, **Catcher's glove**, **Batter's bat**, and **Homeplate**.

## Features

- 🎯 **YOLOv8 Training** - Train custom models on baseball datasets
- 🔍 **Video Inference** - Run object detection on video files
- 🖼️ **Image Inference** - Run object detection on individual images
- 📊 **Dataset Management** - Automatic dataset preparation and validation
- 🎨 **Streamlit UI** - Web interface for video inference and MLB video search
- 📈 **Model Validation** - Evaluate model performance on validation sets

## Detected Objects

The model detects the following objects:

1. **Ball** 🏀
2. **Catcher's glove** 🧤
3. **Batter's bat** 🏏
4. **Homeplate** 🏠

## Requirements

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install ultralytics>=8.0.0 torch>=2.0.0 torchvision>=0.15.0 opencv-python>=4.8.0 numpy>=1.24.0 pillow>=10.0.0 pyyaml>=6.0 matplotlib>=3.7.0 requests>=2.31.0 yt-dlp>=2023.0.0 streamlit>=1.28.0
```

## Project Structure

```
object recognition/
├── main.py                 # Main training and inference script
├── run_inference.py        # Simple inference script
├── app.py                  # Streamlit web UI
├── requirements.txt        # Python dependencies
├── Frames/                 # Source images for training
├── annotation 1/           # Annotation files (directory 1)
├── annotation 2/           # Annotation files (directory 2)
├── dataset/                # Prepared YOLO dataset (auto-generated)
│   ├── train/
│   │   ├── images/
│   │   └── labels/
│   ├── val/
│   │   ├── images/
│   │   └── labels/
│   └── dataset.yaml
└── runs/                   # Training results and predictions
    ├── baseball_detect/
    │   ├── weights/
    │   │   ├── best.pt     # Best model weights
    │   │   └── last.pt     # Last epoch weights
    │   └── ...
    └── predictions/        # Inference output videos/images
```

## Quick Start

### 1. Prepare Dataset

Organize your images and annotations into the YOLO dataset format:

```bash
python main.py --prepare
```

This will:
- Validate annotation files
- Match images with their corresponding labels
- Split data into train/validation sets (80/20)
- Create `dataset/dataset.yaml` configuration file

### 2. Train Model

Train a YOLOv8 model on your dataset:

```bash
# Train with default settings (YOLOv8n, 100 epochs)
python main.py --train

# Train with specific model size
python main.py --train --model s  # Options: n, s, m, l, x

# Train with custom parameters
python main.py --train --model m --epochs 200 --batch 16
```

### 3. Run Inference

Run object detection on videos or images:

```bash
# Using the simple inference script
python run_inference.py path/to/video.mp4

# Using main.py
python main.py --predict path/to/video.mp4

# With custom confidence threshold
python run_inference.py path/to/video.mp4 --conf 0.3

# With custom model weights
python run_inference.py path/to/video.mp4 --weights runs/baseball_detect/weights/best.pt
```

### 4. Use Streamlit UI

Launch the web interface for easy video inference:

```bash
streamlit run app.py
```

The UI provides:
- 📁 Local video selection
- 📤 Video upload
- ⚾ MLB video search and download
- 🎯 Real-time inference with visualization
- ⚙️ Adjustable confidence thresholds

## Training Options

### Model Sizes

Choose the appropriate model size based on your needs:

| Model | Size | Speed | Accuracy | Use Case |
|-------|------|-------|----------|----------|
| `n` (nano) | Smallest | Fastest | Lower | Quick testing, edge devices |
| `s` (small) | Small | Fast | Good | Balanced performance |
| `m` (medium) | Medium | Moderate | Better | Recommended for most use cases |
| `l` (large) | Large | Slower | High | High accuracy needs |
| `x` (xlarge) | Largest | Slowest | Highest | Maximum accuracy |

### Training Parameters

```bash
python main.py --train \
    --model m \              # Model size: n, s, m, l, x
    --epochs 200 \           # Number of training epochs
    --batch 16 \             # Batch size (adjust based on GPU memory)
    --imgsz 640 \            # Image size (640, 1280, etc.)
    --device mps \           # Device: mps (Apple Metal), cuda, or cpu
    --resume                 # Resume from last checkpoint
```

### Device Options

- **`mps`** - Apple Metal (M1/M2 Macs) - Recommended for Mac users
- **`cuda`** - NVIDIA GPU - Recommended for NVIDIA GPUs
- **`cpu`** - CPU only - Slower but works everywhere

## Dataset Preparation

### Annotation Format

Annotations should be in YOLO format (`.txt` files) with one file per image:

```
class_id center_x center_y width height
```

All coordinates are normalized (0.0 to 1.0).

Example:
```
0 0.5 0.5 0.1 0.1  # Ball at center, 10% of image size
1 0.3 0.7 0.15 0.2  # Catcher's glove
```

### Class IDs

- `0` - Ball
- `1` - Catcher's glove
- `2` - Batter's bat
- `3` - Homeplate

### Directory Structure

Place your images in:
```
Frames/
  ├── image1.jpg
  ├── image2.jpg
  └── ...
```

Place your annotations in:
```
annotation 1/obj_train_data/
  ├── image1.txt
  ├── image2.txt
  └── ...

annotation 2/obj_train_data/
  ├── image3.txt
  └── ...
```

The script will automatically:
- Match images with annotations by filename (stem)
- Validate annotation format
- Filter out invalid annotations
- Split into train/validation sets

## Validation

Evaluate your trained model:

```bash
python main.py --validate

# Or with custom weights
python main.py --validate --weights path/to/weights.pt
```

This will compute metrics like:
- Precision
- Recall
- mAP (mean Average Precision)
- Confusion matrix

Results are saved in `runs/baseball_detect/`.

## Inference Options

### Confidence Threshold

Adjust the confidence threshold to control detection sensitivity:

```bash
# Lower threshold = more detections (may include false positives)
python run_inference.py video.mp4 --conf 0.15

# Higher threshold = fewer detections (more conservative)
python run_inference.py video.mp4 --conf 0.5
```

### Output Location

Inference results are saved to:
```
runs/predictions/
```

Videos are saved with the same filename as the input, with bounding boxes drawn on detected objects.

## Streamlit Web UI

The Streamlit interface (`app.py`) provides a user-friendly way to:

1. **Select Local Videos** - Choose from videos in the `video_downloader/videos/` directory
2. **Upload Videos** - Upload your own video files
3. **Search MLB Videos** - Search and download MLB clips directly from the interface
4. **Run Inference** - Process videos with adjustable settings
5. **View Results** - Watch annotated videos and download results

### Launch UI

```bash
streamlit run app.py
```

The UI will open in your browser at `http://localhost:8501`.

## Integration with Video Downloader

This module integrates with the `video_downloader` module:

- The video downloader can automatically run inference on downloaded clips
- The Streamlit UI can search and download MLB videos
- Shared video directory structure for easy workflow

## Model Weights

After training, model weights are saved to:
```
runs/baseball_detect/weights/
  ├── best.pt    # Best model (highest mAP)
  └── last.pt    # Last epoch checkpoint
```

Use `best.pt` for inference as it represents the best performing model.

## Troubleshooting

### "Dataset not prepared" Error

Run dataset preparation first:
```bash
python main.py --prepare
```

### Shape Mismatch Errors

If you encounter shape mismatch errors during training:
- Reduce batch size: `--batch 4` or `--batch 8`
- Reduce image size: `--imgsz 416`
- Check for images with too many objects (filtered during preparation)

### Out of Memory Errors

- Reduce batch size: `--batch 4`
- Use smaller model: `--model n` or `--model s`
- Reduce image size: `--imgsz 416`

### No Detections

- Lower confidence threshold: `--conf 0.15`
- Check that model weights exist and are valid
- Verify input video/image format is supported

### MPS (Apple Metal) Issues

If you encounter issues with MPS on Mac:
- Try using CPU: `--device cpu`
- Update PyTorch to latest version
- Check that you have a compatible Mac (M1/M2/M3)

## Performance Tips

1. **Use appropriate model size** - Start with `s` or `m` for good balance
2. **Optimize batch size** - Use largest batch that fits in memory
3. **Use GPU acceleration** - MPS for Mac, CUDA for NVIDIA
4. **Pre-process videos** - Extract frames if needed for training
5. **Data augmentation** - Already included in training (mosaic, mixup, etc.)

## Examples

### Complete Training Workflow

```bash
# 1. Prepare dataset
python main.py --prepare

# 2. Train model
python main.py --train --model m --epochs 200

# 3. Validate model
python main.py --validate

# 4. Run inference
python run_inference.py ../video_downloader/videos/my_video.mp4
```

### Quick Inference

```bash
# Simple inference on a video
python run_inference.py video.mp4

# Inference with custom settings
python run_inference.py video.mp4 --conf 0.3 --weights runs/baseball_detect/weights/best.pt
```

## License

This project uses YOLOv8 from Ultralytics, which is licensed under AGPL-3.0.
