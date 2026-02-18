"""
Angle Clusterer - Discover camera angles using unsupervised clustering
Supports DBSCAN, K-means, and hierarchical clustering
"""

import numpy as np
from sklearn.cluster import DBSCAN, KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Tuple, Optional
import warnings


class AngleClusterer:
    """Discover camera angles using unsupervised clustering."""
    
    def __init__(self, config: dict):
        """
        Initialize angle clusterer.
        
        Args:
            config: Configuration dictionary with clustering settings
        """
        self.config = config
        self.cluster_config = config.get('clustering', {})
        self.method = self.cluster_config.get('method', 'dbscan').lower()
        self.min_samples = self.cluster_config.get('min_samples', 5)
        self.eps = self.cluster_config.get('eps', 0.3)
        self.n_clusters = self.cluster_config.get('n_clusters', None)
        
        # Temporal smoothing parameters
        self.temporal_window = self.cluster_config.get('temporal_window', 5)
        
        # Fitted clusterer for global clustering
        self.fitted_clusterer = None
        self.cluster_centers_ = None
    
    def fit_global_angles(self, all_features: np.ndarray) -> None:
        """
        Fit clustering model on all video features to discover global camera angles.
        
        Args:
            all_features: Combined feature matrix from all videos (n_total_frames, feature_dim)
        """
        print(f"🎯 Training global angle model on {all_features.shape[0]} frames from all videos...")
        print(f"   Features shape: {all_features.shape}")
        
        # Apply clustering on all features
        if self.method == 'dbscan':
            self.fitted_clusterer = DBSCAN(eps=self.eps, min_samples=self.min_samples, metric='cosine')
            labels = self.fitted_clusterer.fit_predict(all_features)
            
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = list(labels).count(-1)
            print(f"   Found {n_clusters} global camera angles, {n_noise} noise points")
            
            # For DBSCAN, compute cluster centers from labeled points
            if n_clusters > 0:
                self.cluster_centers_ = []
                unique_labels = [l for l in set(labels) if l >= 0]
                for label in sorted(unique_labels):
                    cluster_features = all_features[labels == label]
                    center = np.mean(cluster_features, axis=0)
                    self.cluster_centers_.append(center)
                self.cluster_centers_ = np.array(self.cluster_centers_)
        
        elif self.method == 'kmeans':
            if self.n_clusters is None:
                n_clusters = self._find_optimal_clusters(all_features)
                print(f"   Auto-detected {n_clusters} global camera angles")
            else:
                n_clusters = self.n_clusters
                print(f"   Using {n_clusters} global camera angles")
            
            self.fitted_clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            self.fitted_clusterer.fit(all_features)
            self.cluster_centers_ = self.fitted_clusterer.cluster_centers_
        
        elif self.method == 'hierarchical':
            if self.n_clusters is None:
                n_clusters = self._find_optimal_clusters(all_features)
                print(f"   Auto-detected {n_clusters} global camera angles")
            else:
                n_clusters = self.n_clusters
                print(f"   Using {n_clusters} global camera angles")
            
            self.fitted_clusterer = AgglomerativeClustering(
                n_clusters=n_clusters,
                linkage='ward',
                metric='euclidean'
            )
            labels = self.fitted_clusterer.fit_predict(all_features)
            
            # Compute cluster centers
            self.cluster_centers_ = []
            for label in range(n_clusters):
                cluster_features = all_features[labels == label]
                center = np.mean(cluster_features, axis=0)
                self.cluster_centers_.append(center)
            self.cluster_centers_ = np.array(self.cluster_centers_)
        
        else:
            raise ValueError(f"Unknown clustering method: {self.method}")
        
        n_angles = len(self.cluster_centers_)
        print(f"✅ Global angle model trained with {n_angles} camera angles")
        
        if n_angles == 1:
            print("⚠️  WARNING: Only 1 angle discovered! This may indicate:")
            print("   - Clustering parameters are too strict (try lower eps for DBSCAN)")
            print("   - Not enough variation in camera angles across videos")
            print("   - Consider using kmeans with n_clusters=6-10 to force multiple angles")
    
    def predict_angles(self, features: np.ndarray, timestamps: List[float]) -> np.ndarray:
        """
        Assign angle labels to frames using the fitted global model.
        
        Args:
            features: Feature matrix of shape (n_frames, feature_dim)
            timestamps: List of timestamps for each frame
            
        Returns:
            Array of angle labels for each frame
        """
        if self.fitted_clusterer is None or self.cluster_centers_ is None:
            raise ValueError("Model not fitted. Call fit_global_angles() first.")
        
        print(f"🔍 Assigning angles to {features.shape[0]} frames...")
        
        # Assign labels based on nearest cluster center
        # Compute similarity to each cluster center
        similarities = cosine_similarity(features, self.cluster_centers_)
        labels = np.argmax(similarities, axis=1)
        
        # Show angle distribution before smoothing
        unique_before = np.unique(labels)
        print(f"   Angles before smoothing: {len(unique_before)} (angles: {sorted(unique_before)})")
        
        # Apply temporal smoothing
        labels = self.apply_temporal_smoothing(labels, timestamps)
        
        # Show angle distribution after smoothing
        unique_after = np.unique(labels)
        n_angles = len(unique_after)
        angle_counts = {angle: np.sum(labels == angle) for angle in unique_after}
        print(f"✅ Assigned frames to {n_angles} camera angles")
        print(f"   Angle distribution: {angle_counts}")
        
        if n_angles == 1:
            print("⚠️  WARNING: Only 1 angle assigned to this video!")
            print("   This may be due to:")
            print("   - Temporal smoothing being too aggressive (reduce temporal_window)")
            print("   - Global model only discovered 1 angle (check global clustering)")
        
        return labels
    
    def discover_angles(self, features: np.ndarray, timestamps: List[float]) -> np.ndarray:
        """
        Discover camera angles by clustering frame features (legacy method for single video).
        
        Args:
            features: Feature matrix of shape (n_frames, feature_dim)
            timestamps: List of timestamps for each frame
            
        Returns:
            Array of angle labels for each frame
        """
        print(f"🔍 Discovering camera angles using {self.method.upper()}...")
        print(f"   Features shape: {features.shape}")
        
        # Apply clustering
        if self.method == 'dbscan':
            labels = self._cluster_dbscan(features)
        elif self.method == 'kmeans':
            labels = self._cluster_kmeans(features)
        elif self.method == 'hierarchical':
            labels = self._cluster_hierarchical(features)
        else:
            raise ValueError(f"Unknown clustering method: {self.method}")
        
        # Apply temporal smoothing
        labels = self.apply_temporal_smoothing(labels, timestamps)
        
        # Relabel to ensure consecutive numbering starting from 0
        labels = self._relabel_clusters(labels)
        
        n_angles = len(np.unique(labels[labels >= 0]))
        print(f"✅ Discovered {n_angles} camera angles")
        
        return labels
    
    def _cluster_dbscan(self, features: np.ndarray) -> np.ndarray:
        """Cluster using DBSCAN (automatically determines number of clusters)."""
        print(f"   DBSCAN parameters: eps={self.eps}, min_samples={self.min_samples}")
        
        clusterer = DBSCAN(eps=self.eps, min_samples=self.min_samples, metric='cosine')
        labels = clusterer.fit_predict(features)
        
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        
        print(f"   Found {n_clusters} clusters, {n_noise} noise points")
        
        return labels
    
    def _cluster_kmeans(self, features: np.ndarray) -> np.ndarray:
        """Cluster using K-means."""
        if self.n_clusters is None:
            # Use elbow method to determine optimal number of clusters
            n_clusters = self._find_optimal_clusters(features)
            print(f"   Auto-detected {n_clusters} clusters using elbow method")
        else:
            n_clusters = self.n_clusters
            print(f"   Using {n_clusters} clusters")
        
        clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = clusterer.fit_predict(features)
        
        return labels
    
    def _cluster_hierarchical(self, features: np.ndarray) -> np.ndarray:
        """Cluster using hierarchical clustering."""
        if self.n_clusters is None:
            # Use elbow method to determine optimal number of clusters
            n_clusters = self._find_optimal_clusters(features)
            print(f"   Auto-detected {n_clusters} clusters using elbow method")
        else:
            n_clusters = self.n_clusters
            print(f"   Using {n_clusters} clusters")
        
        clusterer = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage='ward',
            metric='euclidean'
        )
        labels = clusterer.fit_predict(features)
        
        return labels
    
    def _find_optimal_clusters(self, features: np.ndarray, max_k: int = 10) -> int:
        """
        Find optimal number of clusters using elbow method and silhouette score.
        
        Args:
            features: Feature matrix
            max_k: Maximum number of clusters to try
            
        Returns:
            Optimal number of clusters
        """
        max_k = min(max_k, len(features) // 2)  # Don't exceed half the data points
        
        if max_k < 2:
            return 2
        
        scores = []
        k_range = range(2, max_k + 1)
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(features)
            score = silhouette_score(features, labels)
            scores.append(score)
        
        # Find elbow (maximum silhouette score)
        optimal_k = k_range[np.argmax(scores)]
        
        return optimal_k
    
    def apply_temporal_smoothing(self, labels: np.ndarray, timestamps: List[float]) -> np.ndarray:
        """
        Apply temporal smoothing to reduce noise in angle transitions.
        Uses a more conservative approach to preserve quick cuts.
        
        Args:
            labels: Original cluster labels
            timestamps: Frame timestamps
            
        Returns:
            Smoothed labels
        """
        if self.temporal_window <= 1:
            return labels
        
        smoothed_labels = labels.copy()
        window_size = min(self.temporal_window, len(labels) // 3)  # Reduced max window
        
        if window_size < 1:
            return labels
        
        # Apply conservative smoothing - only smooth if there's a clear majority
        for i in range(len(labels)):
            start = max(0, i - window_size // 2)
            end = min(len(labels), i + window_size // 2 + 1)
            window_labels = labels[start:end]
            
            # Get most common label in window (excluding noise -1)
            valid_labels = window_labels[window_labels >= 0]
            if len(valid_labels) > 0:
                # Only change if there's a clear majority (at least 60% of window)
                label_counts = np.bincount(valid_labels)
                most_common_label = label_counts.argmax()
                most_common_count = label_counts[most_common_label]
                
                # Only smooth if majority is strong enough
                if most_common_count >= len(valid_labels) * 0.6:
                    smoothed_labels[i] = most_common_label
                else:
                    # Keep original label if no clear majority
                    smoothed_labels[i] = labels[i]
            else:
                smoothed_labels[i] = labels[i]
        
        return smoothed_labels
    
    def _relabel_clusters(self, labels: np.ndarray) -> np.ndarray:
        """
        Relabel clusters to ensure consecutive numbering starting from 0.
        
        Args:
            labels: Original cluster labels (may have gaps or negative values)
            
        Returns:
            Relabeled clusters (0, 1, 2, ...)
        """
        unique_labels = np.unique(labels)
        
        # Handle noise points (-1)
        if -1 in unique_labels:
            # Assign noise points to nearest cluster or create new cluster
            noise_mask = labels == -1
            if noise_mask.sum() > 0:
                # For now, assign noise to the most common cluster
                valid_labels = labels[~noise_mask]
                if len(valid_labels) > 0:
                    most_common = np.bincount(valid_labels).argmax()
                    labels[noise_mask] = most_common
        
        # Relabel to consecutive integers
        unique_labels = np.unique(labels)
        label_map = {old_label: new_label for new_label, old_label in enumerate(unique_labels)}
        relabeled = np.array([label_map[label] for label in labels])
        
        return relabeled
    
    def get_angle_statistics(self, labels: np.ndarray, timestamps: List[float]) -> dict:
        """
        Get statistics about discovered angles.
        
        Args:
            labels: Angle labels for each frame
            timestamps: Frame timestamps
            
        Returns:
            Dictionary with angle statistics
        """
        unique_labels = np.unique(labels)
        stats = {
            'n_angles': len(unique_labels),
            'angles': {}
        }
        
        for angle_id in unique_labels:
            mask = labels == angle_id
            angle_timestamps = np.array(timestamps)[mask]
            
            stats['angles'][int(angle_id)] = {
                'n_frames': int(mask.sum()),
                'duration': float(angle_timestamps[-1] - angle_timestamps[0]) if len(angle_timestamps) > 1 else 0.0,
                'start_time': float(angle_timestamps[0]) if len(angle_timestamps) > 0 else 0.0,
                'end_time': float(angle_timestamps[-1]) if len(angle_timestamps) > 0 else 0.0,
                'percentage': float(mask.sum() / len(labels) * 100)
            }
        
        return stats
