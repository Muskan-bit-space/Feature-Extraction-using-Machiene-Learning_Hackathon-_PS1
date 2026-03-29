# AI Feature Extraction from Orthophotos

## Overview
This repository implements an **end-to-end deep learning pipeline for automatic feature extraction from orthophoto images**.  
The system uses a **U-Net semantic segmentation model with a ResNet34 encoder** to detect geospatial features such as roads, bridges, water bodies, railways, and utilities from drone or satellite imagery.

The pipeline integrates **geospatial data processing and deep learning** to convert raster imagery and vector training labels into a trained segmentation model capable of producing **GIS-ready shapefiles**.

The predicted segmentation masks are converted into **vector geometries**, allowing the outputs to be directly used in GIS platforms such as **QGIS or ArcGIS**.

---

# Features

- End-to-end geospatial deep learning pipeline
- Multi-class semantic segmentation
- Automatic rasterization of shapefile labels
- Patch-based training for large orthophotos
- U-Net with ResNet34 encoder
- Tile-based inference for very large raster files
- Conversion of segmentation masks to GIS shapefiles
- Integration with geospatial libraries (Rasterio, GeoPandas)

---

# Classes Detected

| Class | ID |
|------|------|
| Bridge | 0 |
| Built Up Area | 1 |
| Railway | 2 |
| Road | 3 |
| Road Centre Line | 4 |
| Utility | 5 |
| Utility Polygon | 6 |
| Water Body | 7 |
| Water Body Line | 8 |
| Waterbody Point | 9 |

---

# Project Pipeline

```
Orthophoto (.tif) + Vector Labels (.shp)
            │
            ▼
      Data Preprocessing
            │
            ▼
    Rasterization of Labels
            │
            ▼
      Patch Generation
        (256 × 256)
            │
            ▼
        Model Training
       (U-Net + ResNet34)
            │
            ▼
        Model Inference
            │
            ▼
   Segmentation Mask Output
            │
            ▼
 Raster → Vector Conversion
            │
            ▼
      Output Shapefile
```

---

# Project Structure

```
project/
│
├── config.py
├── preprocess.py
├── dataset.py
├── utils.py
├── train.py
├── predict.py
├── predict_tile_by_tile.py
├── main.py
│
├── data/
│   ├── orthophotos/
│   └── shapefiles/
│
├── output/
│   ├── patches/
│   │   ├── images/
│   │   └── masks/
│   │
│   ├── models/
│   └── predictions/
```

---

# Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/feature-extraction-orthophotos.git
cd feature-extraction-orthophotos
```

Install dependencies:

```bash
pip install torch
pip install segmentation-models-pytorch
pip install rasterio
pip install geopandas
pip install opencv-python
pip install numpy
pip install matplotlib
```

---

# Data Preparation

### 1. Orthophoto Image

Input raster file:

```
GeoTIFF (.tif)
```

Example:

```
NADALA_28996_ORTHO.tif
```

### 2. Vector Shapefiles

Training labels provided as shapefiles.

Examples:

```
Bridge.shp
Road.shp
Water_Body.shp
Railway.shp
Utility.shp
```

These shapefiles are converted into a **segmentation mask** during preprocessing.

---

# Preprocessing

Run preprocessing to create segmentation masks and training patches.

```bash
python main.py
```

This step performs:

1. Load orthophoto raster
2. Load vector shapefiles
3. Convert shapefiles into raster masks
4. Generate image and mask patches

Output directories:

```
output/patches/images/
output/patches/masks/
```

---

# Model Training

Train the segmentation model:

```bash
python train.py
```

Training configuration:

- Model: **U-Net**
- Encoder: **ResNet34**
- Loss Function: **CrossEntropyLoss**
- Optimizer: **Adam**
- Epochs: **45**
- Batch Size: **4**

Trained model will be saved in:

```
output/models/unet_with_45_Epochs.pth
```

---

# Prediction

Run inference on an image:

```bash
python predict.py --predict_file image_name.tif
```

This step will:

1. Load the trained model
2. Generate segmentation mask
3. Convert mask into shapefile

Output:

```
output/predictions/output.shp
```

---

# Large Image Prediction

For very large orthophotos (1GB+), tile-based prediction can be used:

```bash
python predict_tile_by_tile.py
```

This splits the raster into tiles and predicts each tile separately.

---

# Output

The pipeline generates two types of outputs:

### Segmentation Mask

```
prediction.tif
```

Pixel-wise classified raster.

### Vector Shapefile

```
output.shp
```

Vector geometries for detected GIS features.

These outputs can be directly loaded into:

- QGIS
- ArcGIS
- PostGIS

---

# Technologies Used

- Python
- PyTorch
- segmentation_models_pytorch
- Rasterio
- GeoPandas
- OpenCV
- NumPy
- Matplotlib

---

# Applications

This project can be used for:

- Drone-based mapping
- Automated road extraction
- Water body detection
- Infrastructure mapping
- Smart city planning
- Land use analysis
- Geospatial AI pipelines

---

# Future Improvements

- Add data augmentation
- Implement Dice Loss / Focal Loss
- Improve tile merging for large images
- Add model evaluation metrics (IoU, Dice score)
- Support additional encoder architectures

---

# License

MIT License
