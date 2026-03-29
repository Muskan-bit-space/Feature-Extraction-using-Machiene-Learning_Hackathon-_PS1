# main.py

from preprocess import *
from dataset import *
from config import *

raster_path = DATA_DIR+"NADALA_28996_ORTHO.tif"

vector_files = {
    "Bridge":               DATA_DIR+"shp-file\/Bridge.shp",
    "Built_Up_Area_type":   DATA_DIR+"shp-file\/Built_Up_Area_typ.shp",
    "Railway":              DATA_DIR+"shp-file\/Railway.shp",
    "Road":                 DATA_DIR+"shp-file\/Road.shp",
    "Roal_Centre_Line":     DATA_DIR+"shp-file\/Road_Centre_Line.shp",
    "Utlity":               DATA_DIR+"shp-file\/Utility.shp",
    "Utility_Poly_":        DATA_DIR+"shp-file\/Utility_Poly_.shp",
    "Water_Body":           DATA_DIR+"shp-file\/Water_Body.shp",
    "Water_Body_Line":      DATA_DIR+"shp-file\/Water_Body_Line.shp",
    "Waterbody_Point":      DATA_DIR+"shp-file\/Waterbody_Point.shp"
}

raster = load_raster(raster_path)

mask = create_combined_mask(raster, vector_files)

save_mask(mask, raster, OUTPUT_DIR+"masks/mask.tif")

image = raster_to_array(raster)

create_patches(image, mask, OUTPUT_DIR+"patches")