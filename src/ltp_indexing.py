import os
import cv2
import numpy as np
import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple
import time


class LTPIndexingSystem:

    
    def __init__(self, database_path: str, index_path: str = "index_data", 
                 threshold: float = 0.05, cell_size: Tuple[int, int] = (4, 4)):
        self.database_path = Path(database_path)
        self.index_path = Path(index_path)
        self.threshold = threshold
        self.cell_size = cell_size
        self.index_path.mkdir(exist_ok=True)
        
        self.image_metadata = {}
        
    def preprocess_image(self, image: np.ndarray, target_size: Tuple[int, int] = (120, 120)) -> np.ndarray:
        img = cv2.resize(image, target_size)
        
        img = img.astype(np.float32) / 255.0
        
        img_uint8 = (img * 255).astype(np.uint8)
        img_eq = cv2.equalizeHist(img_uint8)
        img = img_eq.astype(np.float32) / 255.0
        
        img_uint8 = (img * 255).astype(np.uint8)
        img_bilateral = cv2.bilateralFilter(img_uint8, 9, 75, 75)
        img = img_bilateral.astype(np.float32) / 255.0
        
        kernel_size = 5
        local_mean = cv2.blur(img, (kernel_size, kernel_size))
        
        local_sq_mean = cv2.blur(img ** 2, (kernel_size, kernel_size))
        local_std = np.sqrt(np.maximum(local_sq_mean - local_mean ** 2, 0.0001))
        
        img = (img - local_mean) / (local_std + 1e-10)
        
        img = np.clip(img, -2, 2)
        
        return img
    
    def compute_ltp(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        h, w = image.shape
        
        ltp_positive = np.zeros((h-2, w-2), dtype=np.uint8)
        ltp_negative = np.zeros((h-2, w-2), dtype=np.uint8)
        
        neighbors = [(-1,-1), (-1,0), (-1,1), (0,1), (1,1), (1,0), (1,-1), (0,-1)]
        
        for i in range(1, h-1):
            for j in range(1, w-1):
                center = image[i, j]
                
                pos_code = 0
                neg_code = 0
                
                for idx, (dy, dx) in enumerate(neighbors):
                    neighbor_val = image[i+dy, j+dx]
                    diff = neighbor_val - center
                    
                    if diff >= self.threshold:
                        pos_code |= (1 << idx)
                    
                    if diff <= -self.threshold:
                        neg_code |= (1 << idx)
                
                ltp_positive[i-1, j-1] = pos_code
                ltp_negative[i-1, j-1] = neg_code
        
        return ltp_positive, ltp_negative
    
    def is_uniform(self, pattern: int) -> bool:
        binary = format(pattern, '08b')
        binary_circular = binary + binary[0]
        
        transitions = sum(binary_circular[i] != binary_circular[i+1] 
                        for i in range(8))
        
        return transitions <= 2
    
    def compute_uniform_ltp_histogram(self, ltp_pos: np.ndarray, ltp_neg: np.ndarray) -> np.ndarray:
        h, w = ltp_pos.shape
        cell_h, cell_w = self.cell_size
        
        n_cells_y = h // cell_h
        n_cells_x = w // cell_w
        
        n_bins = 59
        
        uniform_patterns = []
        for i in range(256):
            if self.is_uniform(i):
                uniform_patterns.append(i)
        
        pattern_to_bin = {}
        for idx, pattern in enumerate(uniform_patterns):
            pattern_to_bin[pattern] = idx
        
        histograms = []
        
        for ltp_channel in [ltp_pos, ltp_neg]:
            for cell_y in range(n_cells_y):
                for cell_x in range(n_cells_x):
                    y_start = cell_y * cell_h
                    x_start = cell_x * cell_w
                    cell = ltp_channel[y_start:y_start+cell_h, x_start:x_start+cell_w]
                    
                    hist = np.zeros(n_bins, dtype=np.float32)
                    
                    for val in cell.flatten():
                        if val in pattern_to_bin:
                            bin_idx = pattern_to_bin[val]
                        else:
                            bin_idx = n_bins - 1
                        hist[bin_idx] += 1
                    
                    hist_sum = np.sum(hist)
                    if hist_sum > 0:
                        hist = hist / hist_sum
                    
                    histograms.append(hist)
        
        feature_vector = np.concatenate(histograms)
        
        return feature_vector
    
    def extract_features(self, image_path: str) -> np.ndarray:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        if img is None:
            raise ValueError(f"Cannot read image: {image_path}")
        
        preprocessed = self.preprocess_image(img)
        
        ltp_pos, ltp_neg = self.compute_ltp(preprocessed)
        
        features = self.compute_uniform_ltp_histogram(ltp_pos, ltp_neg)
        
        return features
    
    def browse_directory(self) -> List[str]:
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        image_files = []
        
        print(f"Browsing directory: {self.database_path}")
        
        for root, dirs, files in os.walk(self.database_path):
            for file in files:
                if Path(file).suffix.lower() in image_extensions:
                    full_path = os.path.join(root, file)
                    image_files.append(full_path)
        
        print(f"Found {len(image_files)} images")
        return sorted(image_files)
    
    def build_index(self) -> Dict:
        print("\n" + "="*60)
        print("BUILDING LTP INDEX FOR IMAGE DATABASE")
        print("="*60)
        
        start_time = time.time()
        
        image_files = self.browse_directory()
        
        if not image_files:
            print("No images found in database!")
            return {}
        
        features_dict = {}
        metadata_dict = {}
        failed_images = []
        
        print(f"\nExtracting features from {len(image_files)} images...")
        print("-" * 60)
        
        for idx, img_path in enumerate(image_files):
            try:
                features = self.extract_features(img_path)
                
                image_id = f"img_{idx:06d}"
                
                features_dict[image_id] = features
                metadata_dict[image_id] = {
                    'path': img_path,
                    'filename': os.path.basename(img_path),
                    'size': os.path.getsize(img_path),
                    'feature_dim': len(features)
                }
                
                if (idx + 1) % 10 == 0:
                    print(f"Processed {idx + 1}/{len(image_files)} images", end='\r')
                    
            except Exception as e:
                print(f"\nError processing {img_path}: {str(e)}")
                failed_images.append(img_path)
        
        print(f"\nProcessed {len(image_files)}/{len(image_files)} images")
        
        print("\nSaving index files...")
        self._save_index_files(features_dict, metadata_dict)
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        # Statistics
        stats = {
            'total_images': len(image_files),
            'indexed_images': len(features_dict),
            'failed_images': len(failed_images),
            'feature_dimension': len(next(iter(features_dict.values()))) if features_dict else 0,
            'processing_time': elapsed_time,
            'avg_time_per_image': elapsed_time / len(image_files) if image_files else 0
        }
        
        self._print_statistics(stats)
        
        return stats
    
    def _save_index_files(self, features_dict: Dict, metadata_dict: Dict):
        features_file = self.index_path / "features.pkl"
        with open(features_file, 'wb') as f:
            pickle.dump(features_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"  - Saved features to: {features_file}")
        
        features_array_file = self.index_path / "features_array.npy"
        image_ids = sorted(features_dict.keys())
        features_array = np.array([features_dict[img_id] for img_id in image_ids])
        np.save(features_array_file, features_array)
        print(f"  - Saved feature array to: {features_array_file}")
        
        id_mapping_file = self.index_path / "id_mapping.json"
        with open(id_mapping_file, 'w') as f:
            json.dump(image_ids, f, indent=2)
        print(f"  - Saved ID mapping to: {id_mapping_file}")
        
        metadata_file = self.index_path / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata_dict, f, indent=2)
        print(f"  - Saved metadata to: {metadata_file}")
        
        config = {
            'threshold': self.threshold,
            'cell_size': self.cell_size,
            'target_size': (120, 120),
            'n_images': len(features_dict),
            'feature_dimension': len(next(iter(features_dict.values())))
        }
        config_file = self.index_path / "config.json"
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"  - Saved configuration to: {config_file}")
    
    def _print_statistics(self, stats: Dict):
        print("\n" + "="*60)
        print("INDEXING STATISTICS")
        print("="*60)
        print(f"Total images found:      {stats['total_images']}")
        print(f"Successfully indexed:    {stats['indexed_images']}")
        print(f"Failed images:           {stats['failed_images']}")
        print(f"Feature dimension:       {stats['feature_dimension']}")
        print(f"Total processing time:   {stats['processing_time']:.2f} seconds")
        print(f"Avg time per image:      {stats['avg_time_per_image']:.3f} seconds")
        print("="*60)
