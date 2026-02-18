"""
Angle Classifier - Use AI/ML to classify camera angles
Uses CLIP text-image matching to identify specific camera angles
"""

import os
# Suppress Hugging Face tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import numpy as np
from typing import List, Dict, Optional
from PIL import Image
import cv2

try:
    from transformers import CLIPProcessor, CLIPModel
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    raise ImportError("transformers library required for angle classification. Install with: pip install transformers")


class AngleClassifier:
    """Classify camera angles using CLIP text-image matching."""
    
    # Improved baseball camera angle descriptions with more specific terminology
    DEFAULT_ANGLE_DESCRIPTIONS = [
        # Main broadcast angles
        "center field camera high angle view from behind home plate showing the entire baseball diamond and outfield",
        "center field camera medium shot from behind home plate showing the pitcher, batter, and infield",
        "first base camera angle from the first base side showing the first baseman, batter, and pitcher",
        "third base camera angle from the third base side showing the third baseman, batter, and pitcher",
        "home plate camera closeup view of the batter at the plate with the catcher and umpire",
        "pitcher camera view focused on the pitcher's mound and delivery from the side",
        "behind pitcher camera view from behind the pitcher showing the batter and catcher",
        
        # Infield angles
        "first base line camera view along the first base line showing the runner and first baseman",
        "third base line camera view along the third base line showing the runner and third baseman",
        "second base camera view showing the middle infield and second base area",
        "shortstop camera view from the shortstop position showing the infield",
        
        # Outfield angles
        "left field camera view from left field showing the outfield and infield",
        "right field camera view from right field showing the outfield and infield",
        "center field outfield camera view from deep center field showing the entire field",
        
        # Special angles
        "dugout camera view from the team dugout showing players and coaches",
        "high home plate camera view from high above home plate looking down at the field",
        "low angle camera view from ground level looking up at the players",
        "replay camera angle showing slow motion closeup of a specific play or player",
        "aerial camera view from above showing the entire baseball stadium and field",
        "fan camera view from the stands showing the field from spectator perspective"
    ]
    
    def __init__(self, config: dict, angle_descriptions: Optional[List[str]] = None):
        """
        Initialize angle classifier.
        
        Args:
            config: Configuration dictionary
            angle_descriptions: Optional list of angle descriptions to classify
        """
        self.config = config
        self.clustering_config = config.get('clustering', {})
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
        
        # Confidence threshold for classification
        self.confidence_threshold = self.clustering_config.get('confidence_threshold', 0.2)
        
        # Load CLIP model (use larger model for better accuracy)
        model_name = self.clustering_config.get('clip_model', 'openai/clip-vit-base-patch32')
        print(f"🔄 Loading CLIP model ({model_name}) for angle classification on {self.device}...")
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.eval()
        
        # Set angle descriptions
        if angle_descriptions is None:
            # Use config or default descriptions
            config_descriptions = config.get('clustering', {}).get('angle_descriptions', None)
            if config_descriptions:
                self.angle_descriptions = config_descriptions
            else:
                self.angle_descriptions = self.DEFAULT_ANGLE_DESCRIPTIONS
        else:
            self.angle_descriptions = angle_descriptions
        
        # Pre-compute text embeddings for all angle descriptions
        self._precompute_text_embeddings()
        
        print(f"✅ Angle classifier ready with {len(self.angle_descriptions)} angle types")
    
    def _precompute_text_embeddings(self):
        """Pre-compute text embeddings for all angle descriptions."""
        print(f"📝 Pre-computing text embeddings for {len(self.angle_descriptions)} angle descriptions...")
        
        with torch.no_grad():
            text_inputs = self.processor(
                text=self.angle_descriptions,
                return_tensors="pt",
                padding=True,
                truncation=True
            )
            text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}
            self.text_embeddings = self.model.get_text_features(**text_inputs)
            # Normalize embeddings
            self.text_embeddings = self.text_embeddings / self.text_embeddings.norm(dim=-1, keepdim=True)
        
        print(f"✅ Text embeddings computed: shape {self.text_embeddings.shape}")
    
    def classify_frame(self, frame: np.ndarray) -> Dict:
        """
        Classify a single frame to determine its camera angle.
        
        Args:
            frame: Frame as numpy array (BGR format from OpenCV)
            
        Returns:
            Dictionary with angle_id, angle_name, confidence, and all scores
        """
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)
        
        # Process image
        image_input = self.processor(images=image, return_tensors="pt")
        image_input = {k: v.to(self.device) for k, v in image_input.items()}
        
        # Get image embedding
        with torch.no_grad():
            image_embedding = self.model.get_image_features(**image_input)
            image_embedding = image_embedding / image_embedding.norm(dim=-1, keepdim=True)
        
        # Compute similarity to all angle descriptions
        similarities = torch.matmul(image_embedding, self.text_embeddings.T)
        similarities = similarities.cpu().numpy().flatten()
        
        # Get best match
        angle_id = int(np.argmax(similarities))
        confidence = float(similarities[angle_id])
        
        return {
            'angle_id': angle_id,
            'angle_name': f"angle_{angle_id}",
            'angle_description': self.angle_descriptions[angle_id],
            'confidence': confidence,
            'all_scores': similarities.tolist()
        }
    
    def classify_frames(self, frames: List[np.ndarray], batch_size: int = 32) -> List[Dict]:
        """
        Classify multiple frames.
        
        Args:
            frames: List of frames as numpy arrays
            batch_size: Batch size for processing
            
        Returns:
            List of classification dictionaries
        """
        print(f"🔍 Classifying {len(frames)} frames using AI angle classifier...")
        
        results = []
        
        # Process in batches
        for i in range(0, len(frames), batch_size):
            batch_frames = frames[i:i + batch_size]
            
            # Convert frames to PIL Images
            images = [Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) for frame in batch_frames]
            
            # Process images
            image_inputs = self.processor(images=images, return_tensors="pt", padding=True)
            image_inputs = {k: v.to(self.device) for k, v in image_inputs.items()}
            
            # Get image embeddings
            with torch.no_grad():
                image_embeddings = self.model.get_image_features(**image_inputs)
                image_embeddings = image_embeddings / image_embeddings.norm(dim=-1, keepdim=True)
            
            # Compute similarities to all angle descriptions
            similarities = torch.matmul(image_embeddings, self.text_embeddings.T)
            similarities = similarities.cpu().numpy()
            
            # Get best matches for each frame
            for j, sim_scores in enumerate(similarities):
                angle_id = int(np.argmax(sim_scores))
                confidence = float(sim_scores[angle_id])
                
                results.append({
                    'angle_id': angle_id,
                    'angle_name': f"angle_{angle_id}",
                    'angle_description': self.angle_descriptions[angle_id],
                    'confidence': confidence,
                    'all_scores': sim_scores.tolist()
                })
            
            if (i // batch_size + 1) % 10 == 0:
                print(f"   Processed {min(i + batch_size, len(frames))}/{len(frames)} frames...")
        
        print(f"✅ Classified {len(results)} frames")
        
        return results
    
    def classify_features(self, features: np.ndarray) -> np.ndarray:
        """
        Classify pre-extracted features (for use with existing feature extraction).
        
        Args:
            features: Feature matrix of shape (n_frames, feature_dim)
            
        Returns:
            Array of angle labels
        """
        print(f"🔍 Classifying {len(features)} feature vectors using AI angle classifier...")
        print(f"   Confidence threshold: {self.confidence_threshold:.2f}")
        
        # Convert features to tensor
        features_tensor = torch.from_numpy(features).float().to(self.device)
        features_tensor = features_tensor / features_tensor.norm(dim=-1, keepdim=True)
        
        # Compute similarities to text embeddings
        with torch.no_grad():
            similarities = torch.matmul(features_tensor, self.text_embeddings.T)
            similarities = similarities.cpu().numpy()
        
        # Get best match for each frame
        angle_labels = np.argmax(similarities, axis=1)
        confidences = np.max(similarities, axis=1)
        
        # Filter low confidence classifications
        low_confidence_mask = confidences < self.confidence_threshold
        n_low_confidence = np.sum(low_confidence_mask)
        
        if n_low_confidence > 0:
            print(f"   ⚠️  {n_low_confidence} frames ({n_low_confidence/len(features)*100:.1f}%) below confidence threshold")
            # For low confidence, use second-best match if it's significantly better
            for i in np.where(low_confidence_mask)[0]:
                sorted_indices = np.argsort(similarities[i])[::-1]
                if len(sorted_indices) > 1:
                    best_score = similarities[i][sorted_indices[0]]
                    second_score = similarities[i][sorted_indices[1]]
                    # If second best is close to best, keep best; otherwise try second
                    if second_score > best_score * 0.9:  # Second is within 90% of best
                        angle_labels[i] = sorted_indices[1]
                        confidences[i] = second_score
        
        avg_confidence = np.mean(confidences)
        min_confidence = np.min(confidences)
        max_confidence = np.max(confidences)
        
        print(f"✅ Classified features: {len(angle_labels)} frames")
        print(f"   Confidence: avg={avg_confidence:.3f}, min={min_confidence:.3f}, max={max_confidence:.3f}")
        
        # Show angle distribution
        unique_angles, counts = np.unique(angle_labels, return_counts=True)
        angle_dist = dict(zip(unique_angles, counts))
        print(f"   Angle distribution: {angle_dist}")
        
        return angle_labels
    
    def get_angle_names(self) -> List[str]:
        """Get list of angle description names."""
        return self.angle_descriptions
    
    def get_angle_name(self, angle_id: int) -> str:
        """Get description for a specific angle ID."""
        return self.angle_descriptions[angle_id] if 0 <= angle_id < len(self.angle_descriptions) else f"angle_{angle_id}"
