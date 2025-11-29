# LTP Image Retrieval System

A Local Ternary Pattern (LTP) based offline indexing system for image databases.

## Project Structure

```
LTP-image-retreival/
├── src/
│   └── ltp_indexing.py          # Main LTP indexing class
├── data/
│   └── image_database/          # Place your images here
├── output/
│   └── ltp_index/               # Generated index files
├── main.py                       # Runner script
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## Installation

1. **Clone or navigate to the project directory:**
   ```bash
   cd /Users/tayeb/Desktop/TP/LTP-image-retreival
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Step 1: Prepare Your Images

Place your image files in the `data/image_database/` directory. Supported formats:
- `.jpg`, `.jpeg`
- `.png`
- `.bmp`
- `.tiff`, `.tif`

Example:
```
data/image_database/
├── image1.jpg
├── image2.png
├── subfolder/
│   └── image3.jpg
```

### Step 2: Run the Indexing System

```bash
python main.py
```

### Output

The system will generate the following files in `output/ltp_index/`:

- **features.pkl**: Pickled dictionary of image features
- **features_array.npy**: NumPy array format (faster loading)
- **id_mapping.json**: Maps image IDs to array indices
- **metadata.json**: Image metadata (paths, filenames, sizes)
- **config.json**: System configuration parameters

## Configuration

Edit `main.py` to customize parameters:

```python
indexer = LTPIndexingSystem(
    database_path="data/image_database",
    index_path="output/ltp_index",
    threshold=5.0,              # LTP threshold (higher = more selective)
    cell_size=(8, 8)            # Histogram cell size
)
```

## Parameters Explanation

- **threshold**: Controls the sensitivity of the LTP operator
  - Default: 5.0
  - Range: Typically 5-15
  - Higher values produce more selective patterns

- **cell_size**: Divides image into cells for histogram computation
  - Default: (8, 8)
  - Larger cells = fewer features but faster processing
  - Smaller cells = more features but slower processing

## System Requirements

- Python 3.7+
- OpenCV (cv2)
- NumPy
- 2GB+ RAM recommended for large image databases

## Features

- Automatic image discovery with recursive directory traversal
- Preprocessing: gamma correction and Difference of Gaussian (DoG) filtering
- Local Ternary Pattern computation with positive/negative channels
- Uniform pattern detection (58 uniform + 1 non-uniform pattern)
- Spatial histogram computation with cell-based division
- Error handling and progress reporting
- Multiple output formats (pickle, numpy, JSON)

## Example Output

```
============================================================
BUILDING LTP INDEX FOR IMAGE DATABASE
============================================================
Browsing directory: data/image_database
Found 100 images

Extracting features from 100 images...
------------------------------------------------------------
Processed 100/100 images

Saving index files...
  - Saved features to: output/ltp_index/features.pkl
  - Saved feature array to: output/ltp_index/features_array.npy
  - Saved ID mapping to: output/ltp_index/id_mapping.json
  - Saved metadata to: output/ltp_index/metadata.json
  - Saved configuration to: output/ltp_index/config.json

============================================================
INDEXING STATISTICS
============================================================
Total images found:      100
Successfully indexed:    100
Failed images:           0
Feature dimension:       7544
Total processing time:   45.23 seconds
Avg time per image:      0.452 seconds
============================================================
```

## Notes

- First run will be slower as images are processed sequentially
- Feature dimension depends on image size and cell size (120x120 / 8x8 = 15x15 cells)
- With 2 channels and 59 bins: 15×15×2×59 = 26550 features (default setup)
# LTP-Image-indexation
