# preprocess.py

import os
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from config import CLASS_MAP, TARGET_CRS


def reproject_vector(shp_path, target_crs):
    gdf = gpd.read_file(shp_path)
    return gdf.to_crs(target_crs)


def load_raster(raster_path):
    return rasterio.open(raster_path)


def create_combined_mask(raster, vector_files):
    shapes = []

    for class_name, shp_path in vector_files.items():
        gdf = gpd.read_file(shp_path)
        gdf = gdf.to_crs(raster.crs)
        #print("Printing gdf")
        #print(gdf)

        class_id = CLASS_MAP[class_name]
        class_Name = class_name
        #print("Class ID", class_id, "Class Name", class_name)

        for geom in gdf.geometry:
            #print("Class Name",class_name)
            shapes.append((geom, class_id))

    mask = rasterize(
        shapes,
        out_shape=(raster.height, raster.width),
        transform=raster.transform,
        fill=0,
        dtype="uint8"
    )

    return mask


def save_mask(mask, raster, out_path):
    with rasterio.open(
        out_path,
        'w',
        driver='GTiff',
        height=mask.shape[0],
        width=mask.shape[1],
        count=1,
        dtype=mask.dtype,
        crs=raster.crs,
        transform=raster.transform
    ) as dst:
        dst.write(mask, 1)