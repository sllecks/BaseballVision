"""
Feature Extractor - Extract features from video frames using pre-trained models
Supports CLIP and ResNet50 models
"""

import os
# Suppress Hugging Face tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from typing import List, Optional
import warnings

try:
    from transformers import CLIPProcessor, CLIPModel
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    warnings.warn("transformers library not available. CLIP model will not work.")

try:
    import torchvision.models as models
    TORCHVISION_AVAILABLE = True
except ImportError:
    TORCHVISION_AVAILABLE = False
    warnings.warn("torchvision not available. ResNet model will not work.")


class FeatureExtractor:
    """Extract features from frames using pre-trained vision models."""
    
    def __init__(self, config: dict):
        """
        Initialize feature extractor.
        
        Args:
            config: Configuration dictionary with feature settings
        """
        self.config = config
        self.feature_config = config.get('features', {})
        self.model_name = self.feature_config.get('model', 'clip').lower()
        self.batch_size = self.feature_config.get('batch_size', 32)
        self.image_size = self.feature_config.get('image_size', 224)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
        
        # Initialize model
        self.model = None
        self.processor = None
        self._load_model()
    
    def _load_model(self):
        """Load the specified pre-trained model."""
        print(f"🔄 Loading {self.model_name.upper()} model on {self.device}...")
        
        if self.model_name == 'clip':
            if not CLIP_AVAILABLE:
                raise ImportError("transformers library required for CLIP. Install with: pip install transformers")
            
            self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
            self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            self.model.eval()
            self.feature_dim = 512
            print(f"✅ Loaded CLIP model (feature dim: {self.feature_dim})")
        
        elif self.model_name == 'resnet50':
            if not TORCHVISION_AVAILABLE:
                raise ImportError("torchvision required for ResNet. Install with: pip install torchvision")
            
            self.model = models.resnet50(pretrained=True).to(self.device)
            # Remove the final classification layer
            self.model = torch.nn.Sequential(*list(self.model.children())[:-1])
            self.model.eval()
            self.feature_dim = 2048
            print(f"✅ Loaded ResNet50 model (feature dim: {self.feature_dim})")
            
            # Set up transforms for ResNet
            self.transform = transforms.Compose([
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            raise ValueError(f"Unknown model: {self.model_name}. Choose 'clip' or 'resnet50'")
    
    def _frame_to_pil(self, frame: np.ndarray) -> Image.Image:
        """Convert OpenCV frame (BGR) to PIL Image (RGB)."""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(frame_rgb)
    
    def extract(self, frames: List[np.ndarray]) -> np.ndarray:
        """
        Extract features from frames.
        
        Args:
            frames: List of frames as numpy arrays (BGR format from OpenCV)
            
        Returns:
            Feature matrix of shape (n_frames, feature_dim)
        """
        print(f"🔍 Extracting features from {len(frames)} frames using {self.model_name.upper()}...")
        
        features = []
        
        # Process in batches
        for i in range(0, len(frames), self.batch_size):
            batch_frames = frames[i:i + self.batch_size]
            
            if self.model_name == 'clip':
                batch_features = self._extract_clip_features(batch_frames)
            elif self.model_name == 'resnet50':
                batch_features = self._extract_resnet_features(batch_frames)
            else:
                raise ValueError(f"Unknown model: {self.model_name}")
            
            features.append(batch_features)
            
            if (i // self.batch_size + 1) % 10 == 0:
                print(f"   Processed {min(i + self.batch_size, len(frames))}/{len(frames)} frames...")
        
        features = np.vstack(features)
        print(f"✅ Extracted features: shape {features.shape}")
        
        return features
    
    def _extract_clip_features(self, frames: List[np.ndarray]) -> np.ndarray:
        """Extract features using CLIP model."""
        # Convert frames to PIL Images
        images = [self._frame_to_pil(frame) for frame in frames]
        
        # Process images
        inputs = self.processor(images=images, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Extract features
        with torch.no_grad():
            outputs = self.model.get_image_features(**inputs)
            # Normalize features
            features = outputs / outputs.norm(dim=-1, keepdim=True)
        
        return features.cpu().numpy()
    
    def _extract_resnet_features(self, frames: List[np.ndarray]) -> np.ndarray:
        """Extract features using ResNet50 model."""
        # Convert frames to PIL Images and apply transforms
        images = [self.transform(self._frame_to_pil(frame)) for frame in frames]
        batch_tensor = torch.stack(images).to(self.device)
        
        # Extract features
        with torch.no_grad():
            features = self.model(batch_tensor)
            # Flatten spatial dimensions
            features = features.view(features.size(0), -1)
            # Normalize features
            features = features / features.norm(dim=1, keepdim=True)
        
        return features.cpu().numpy()
