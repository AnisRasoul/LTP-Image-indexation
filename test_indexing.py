import os
import json
import numpy as np
import pickle
from pathlib import Path
from typing import Dict, List, Tuple
import time
import cv2
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import sys

sys.path.insert(0, str(Path(__file__).parent / 'src'))
from ltp_indexing import LTPIndexingSystem


class LTPIndexTester:
    """Test and evaluate the LTP indexing system"""
    
    def __init__(self, index_path: str = "output/ltp_index", database_path: str = "data/image_database"):
        """
        Initialize the tester
        
        Args:
            index_path: Path to the index files
            database_path: Path to the image database
        """
        self.index_path = Path(index_path)
        self.database_path = Path(database_path)
        
        # Load index files
        self.features_dict = self._load_features()
        self.metadata = self._load_metadata()
        self.config = self._load_config()
        self.image_ids = sorted(self.features_dict.keys())
        
        if len(self.image_ids) == 0:
            raise ValueError("No indexed images found! Please run main.py first.")
        
        print(f"Loaded index with {len(self.image_ids)} images")
        print(f"Feature dimension: {len(self.features_dict[self.image_ids[0]])}")
    
    def _load_features(self) -> Dict:
        """Load feature vectors"""
        features_file = self.index_path / "features.pkl"
        if not features_file.exists():
            raise FileNotFoundError(f"Features file not found: {features_file}")
        
        with open(features_file, 'rb') as f:
            return pickle.load(f)
    
    def _load_metadata(self) -> Dict:
        """Load metadata"""
        metadata_file = self.index_path / "metadata.json"
        if not metadata_file.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_file}")
        
        with open(metadata_file, 'r') as f:
            return json.load(f)
    
    def _load_config(self) -> Dict:
        """Load configuration"""
        config_file = self.index_path / "config.json"
        if not config_file.exists():
            raise FileNotFoundError(f"Config file not found: {config_file}")
        
        with open(config_file, 'r') as f:
            return json.load(f)
    
    def search_similar_images(self, query_image_id: str, top_k: int = 10, metric: str = 'cosine') -> List[Tuple]:
        """
        Search for similar images to a query image
        
        Args:
            query_image_id: ID of the query image
            top_k: Number of top results to return
            metric: Distance metric ('cosine' or 'euclidean')
            
        Returns:
            List of (image_id, similarity_score, distance) tuples
        """
        if query_image_id not in self.features_dict:
            raise ValueError(f"Image ID not found: {query_image_id}")
        
        query_features = self.features_dict[query_image_id].reshape(1, -1)
        
        # Build feature matrix
        feature_matrix = np.array([
            self.features_dict[img_id] for img_id in self.image_ids
        ])
        
        # Compute distances
        if metric == 'cosine':
            # Normalize features for cosine similarity
            query_norm = normalize(query_features, norm='l2')
            features_norm = normalize(feature_matrix, norm='l2')
            distances = 1 - cosine_similarity(query_norm, features_norm)[0]
        elif metric == 'euclidean':
            distances = euclidean_distances(query_features, feature_matrix)[0]
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        # Get top-k (excluding query itself)
        sorted_indices = np.argsort(distances)
        results = []
        
        for idx in sorted_indices[:top_k + 1]:
            img_id = self.image_ids[idx]
            dist = distances[idx]
            
            # Skip the query image itself
            if img_id == query_image_id:
                continue
            
            similarity = 1 - dist if metric == 'cosine' else 1 / (1 + dist)
            results.append((img_id, similarity, dist))
        
        return results[:top_k]
    
    def benchmark_search_speed(self, num_queries: int = 10) -> Dict:
        """
        Benchmark the speed of similarity search
        
        Args:
            num_queries: Number of random queries
            
        Returns:
            Speed statistics
        """
        print(f"\nBenchmarking search speed ({num_queries} queries)...")
        print("-" * 60)
        
        # Randomly select query images
        query_indices = np.random.choice(len(self.image_ids), min(num_queries, len(self.image_ids)), replace=False)
        query_ids = [self.image_ids[i] for i in query_indices]
        
        times = []
        
        for query_id in query_ids:
            start = time.time()
            self.search_similar_images(query_id, top_k=10)
            elapsed = time.time() - start
            times.append(elapsed)
        
        times = np.array(times)
        
        stats = {
            'num_queries': len(query_ids),
            'avg_time': np.mean(times),
            'min_time': np.min(times),
            'max_time': np.max(times),
            'std_time': np.std(times)
        }
        
        print(f"Average search time: {stats['avg_time']*1000:.2f} ms")
        print(f"Min/Max time: {stats['min_time']*1000:.2f} / {stats['max_time']*1000:.2f} ms")
        
        return stats
    
    def test_retrieval_consistency(self) -> Dict:
        """
        Test if same query returns same results
        
        Args:
            None
            
        Returns:
            Consistency statistics
        """
        print(f"\nTesting retrieval consistency...")
        print("-" * 60)
        
        # Test with 5 random queries
        num_tests = 5
        query_indices = np.random.choice(len(self.image_ids), num_tests, replace=False)
        
        all_consistent = True
        consistency_count = 0
        
        for idx in query_indices:
            query_id = self.image_ids[idx]
            
            # Get results twice
            results1 = self.search_similar_images(query_id, top_k=10)
            results2 = self.search_similar_images(query_id, top_k=10)
            
            # Compare
            ids1 = [r[0] for r in results1]
            ids2 = [r[0] for r in results2]
            
            if ids1 == ids2:
                consistency_count += 1
            else:
                all_consistent = False
        
        consistency_ratio = consistency_count / num_tests
        
        print(f"Consistent retrievals: {consistency_count}/{num_tests} ({consistency_ratio*100:.1f}%)")
        print(f"Status: {'✓ PASS' if all_consistent else '✗ FAIL'}")
        
        return {
            'consistency_ratio': consistency_ratio,
            'num_tests': num_tests,
            'all_consistent': all_consistent
        }
    
    def test_cross_similarity(self) -> Dict:
        """
        Test similarity matrix properties
        
        Returns:
            Cross-similarity statistics
        """
        print(f"\nAnalyzing similarity distribution...")
        print("-" * 60)
        
        # Sample images for computational efficiency
        sample_size = min(50, len(self.image_ids))
        sample_indices = np.random.choice(len(self.image_ids), sample_size, replace=False)
        
        sample_ids = [self.image_ids[i] for i in sample_indices]
        sample_features = np.array([self.features_dict[img_id] for img_id in sample_ids])
        
        # Compute similarity matrix
        sample_features_norm = normalize(sample_features, norm='l2')
        similarity_matrix = cosine_similarity(sample_features_norm)
        
        # Get upper triangle (excluding diagonal)
        upper_triangle = similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)]
        
        stats = {
            'sample_size': sample_size,
            'mean_similarity': np.mean(upper_triangle),
            'std_similarity': np.std(upper_triangle),
            'min_similarity': np.min(upper_triangle),
            'max_similarity': np.max(upper_triangle),
            'median_similarity': np.median(upper_triangle)
        }
        
        print(f"Mean similarity: {stats['mean_similarity']:.4f}")
        print(f"Std deviation: {stats['std_similarity']:.4f}")
        print(f"Range: [{stats['min_similarity']:.4f}, {stats['max_similarity']:.4f}]")
        
        return stats
    
    def run_full_test_suite(self) -> Dict:
        """Run complete test suite"""
        print("\n" + "="*60)
        print("LTP INDEXING SYSTEM TEST SUITE")
        print("="*60)
        
        results = {}
        
        # Test 1: Index statistics
        print("\n[1/4] Index Statistics")
        print("-" * 60)
        results['index_stats'] = {
            'total_images': len(self.image_ids),
            'feature_dimension': len(self.features_dict[self.image_ids[0]]),
            'index_size_mb': sum(len(pickle.dumps(self.features_dict[img_id])) 
                                for img_id in self.image_ids) / (1024*1024)
        }
        print(f"Total images: {results['index_stats']['total_images']}")
        print(f"Feature dimension: {results['index_stats']['feature_dimension']}")
        print(f"Index size: {results['index_stats']['index_size_mb']:.2f} MB")
        
        # Test 2: Retrieval consistency
        print("\n[2/4] Retrieval Consistency")
        results['consistency'] = self.test_retrieval_consistency()
        
        # Test 3: Search speed
        print("\n[3/4] Search Speed Benchmark")
        results['speed'] = self.benchmark_search_speed(num_queries=10)
        
        # Test 4: Similarity distribution
        print("\n[4/4] Similarity Analysis")
        results['similarity'] = self.test_cross_similarity()
        
        # Print summary
        self._print_test_summary(results)
        
        return results
    
    def _print_test_summary(self, results: Dict):
        """Print test summary"""
        print("\n" + "="*60)
        print("TEST SUMMARY")
        print("="*60)
        
        print(f"\n✓ Index Statistics:")
        print(f"  - Total images: {results['index_stats']['total_images']}")
        print(f"  - Feature dim: {results['index_stats']['feature_dimension']}")
        print(f"  - Index size: {results['index_stats']['index_size_mb']:.2f} MB")
        
        print(f"\n✓ Consistency: {results['consistency']['consistency_ratio']*100:.1f}% consistent")
        
        print(f"\n✓ Search Performance:")
        print(f"  - Avg time: {results['speed']['avg_time']*1000:.2f} ms")
        print(f"  - Min time: {results['speed']['min_time']*1000:.2f} ms")
        print(f"  - Max time: {results['speed']['max_time']*1000:.2f} ms")
        
        print(f"\n✓ Similarity Distribution:")
        print(f"  - Mean: {results['similarity']['mean_similarity']:.4f}")
        print(f"  - Std: {results['similarity']['std_similarity']:.4f}")
        print(f"  - Range: [{results['similarity']['min_similarity']:.4f}, {results['similarity']['max_similarity']:.4f}]")
        
        print("\n" + "="*60)


class LTPVisualizer:
    """Visualize retrieval results"""
    
    def __init__(self, tester: LTPIndexTester):
        """Initialize visualizer"""
        self.tester = tester
        self.database_path = tester.database_path
    
    def visualize_retrieval(self, query_image_id: str, top_k: int = 9, save_path: str = None):
        """
        Visualize query image and top-k similar results
        
        Args:
            query_image_id: ID of query image
            top_k: Number of results to show
            save_path: Path to save visualization (optional)
        """
        results = self.tester.search_similar_images(query_image_id, top_k=top_k)
        
        # Get query image
        query_path = self.tester.metadata[query_image_id]['path']
        query_img = cv2.imread(query_path)
        
        if query_img is None:
            print(f"Cannot read query image: {query_path}")
            return
        
        # Create grid
        fig, axes = plt.subplots(2, 5, figsize=(15, 6))
        fig.suptitle(f'Query Image (ID: {query_image_id}) and Top-{top_k} Similar Images', fontsize=14)
        
        # Plot query image
        query_img_rgb = cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB)
        axes[0, 0].imshow(query_img_rgb)
        axes[0, 0].set_title('Query Image', fontweight='bold')
        axes[0, 0].axis('off')
        
        # Plot results
        for idx, (img_id, similarity, distance) in enumerate(results):
            row = (idx + 1) // 5
            col = (idx + 1) % 5
            
            result_path = self.tester.metadata[img_id]['path']
            result_img = cv2.imread(result_path)
            
            if result_img is not None:
                result_img_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
                axes[row, col].imshow(result_img_rgb)
                axes[row, col].set_title(f'Rank #{idx+1}\nSim: {similarity:.3f}', fontsize=9)
            else:
                axes[row, col].text(0.5, 0.5, 'Image not found', ha='center', va='center')
            
            axes[row, col].axis('off')
        
        # Hide unused subplots
        for idx in range(len(results) + 1, 10):
            row = idx // 5
            col = idx % 5
            axes[row, col].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            print(f"Saved visualization to: {save_path}")
        else:
            plt.show()
        
        return fig
    
    def visualize_similarity_distribution(self, save_path: str = None):
        """Plot similarity distribution"""
        # Get similarity stats
        stats = self.tester.test_cross_similarity()
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Create histogram
        sample_size = min(50, len(self.tester.image_ids))
        sample_indices = np.random.choice(len(self.tester.image_ids), sample_size, replace=False)
        sample_ids = [self.tester.image_ids[i] for i in sample_indices]
        sample_features = np.array([self.tester.features_dict[img_id] for img_id in sample_ids])
        
        sample_features_norm = normalize(sample_features, norm='l2')
        similarity_matrix = cosine_similarity(sample_features_norm)
        upper_triangle = similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)]
        
        ax.hist(upper_triangle, bins=30, alpha=0.7, edgecolor='black')
        ax.axvline(np.mean(upper_triangle), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(upper_triangle):.3f}')
        ax.axvline(np.median(upper_triangle), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(upper_triangle):.3f}')
        
        ax.set_xlabel('Cosine Similarity', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('Distribution of Cross-Image Similarity Scores', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            print(f"Saved histogram to: {save_path}")
        else:
            plt.show()
        
        return fig


if __name__ == "__main__":
    try:
        # Initialize tester
        tester = LTPIndexTester()
        
        # Run full test suite
        results = tester.run_full_test_suite()
        
        # Interactive retrieval demo
        print("\n" + "="*60)
        print("INTERACTIVE RETRIEVAL DEMO")
        print("="*60)
        
        if len(tester.image_ids) > 0:
            # Test with first image
            test_id = tester.image_ids[0]
            print(f"\nSearching for images similar to: {test_id}")
            print(f"Path: {tester.metadata[test_id]['path']}")
            
            similar = tester.search_similar_images(test_id, top_k=5)
            print(f"\nTop 5 similar images:")
            for rank, (img_id, similarity, distance) in enumerate(similar, 1):
                print(f"  {rank}. {img_id} (similarity: {similarity:.4f})")
        
        # Try to visualize if matplotlib available
        try:
            print("\nGenerating visualization...")
            visualizer = LTPVisualizer(tester)
            
            # Visualize random retrieval
            random_idx = np.random.randint(len(tester.image_ids))
            random_id = tester.image_ids[random_idx]
            visualizer.visualize_retrieval(random_id, top_k=9, save_path='output/retrieval_result.png')
            
            # Visualize similarity distribution
            visualizer.visualize_similarity_distribution(save_path='output/similarity_distribution.png')
        except ImportError:
            print("Note: Matplotlib not available. Skipping visualizations.")
            print("Install with: pip install matplotlib")
    
    except Exception as e:
        print(f"Error: {e}")
        print("\nMake sure you've run main.py first to build the index!")
        sys.exit(1)
