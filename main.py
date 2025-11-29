import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from ltp_indexing import LTPIndexingSystem


def main():
    database_path = "data/image_database"
    index_path = "output/ltp_index"
    
    indexer = LTPIndexingSystem(
        database_path=database_path,
        index_path=index_path,
        threshold=0.05,
        cell_size=(4, 4)
    )
    
    stats = indexer.build_index()
    
    print("\nIndexing complete! Files saved to:", index_path)
    print("\nGenerated files:")
    print("  - features.pkl: Binary feature data (pickle format)")
    print("  - features_array.npy: Feature matrix (numpy format)")
    print("  - id_mapping.json: Image ID to array index mapping")
    print("  - metadata.json: Image metadata (paths, filenames, etc.)")
    print("  - config.json: System configuration parameters")


if __name__ == "__main__":
    main()
