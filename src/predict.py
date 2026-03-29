# predict.py

import torch
import rasterio
import numpy as np
import geopandas as gpd
from rasterio.features import shapes
import segmentation_models_pytorch as smp
from config import *
import glob
import argparse
#import tifffile # Use pip install tifffile
import os
import sys
device = "cuda" if torch.cuda.is_available() else "cpu"

model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights=None,
    in_channels=3,
    classes=len(CLASS_MAP)
)

#model.load_state_dict(torch.load(OUTPUT_DIR+"models/unet.pth"))
#model.load_state_dict(torch.load(OUTPUT_DIR+"models/unet_25.pth"))
model.load_state_dict(torch.load(OUTPUT_DIR+"models/unet_with_45_Epochs.pth"))
model.to(device)
model.eval()


def predict_raster(raster_path):
    raster = rasterio.open(raster_path)
    #print("Input Raster height:", raster.height, "Input Raster width:", raster.width)
    img = raster.read([1, 2, 3])
    img = np.transpose(img, (1, 2, 0))

    img_tensor = torch.tensor(img).permute(2,0,1).float().unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(img_tensor)
        #print("Here we are seeing what is model predict:",torch.max(pred, 1))
        pred_mask = torch.argmax(pred, dim=1).squeeze().cpu().numpy()
        #print("Here we are seeing what is model predict Mask:",pred_mask)
        #print(pred_mask)

    return pred_mask, raster

# s and v for raster value is s--> Shave and v-->Pixel value
def mask_to_shapefile(mask, raster, out_path):
    results = (
        {'properties': {'class': int(v)}, 'geometry': s}
        for s, v in shapes(mask.astype('uint8'), transform=raster.transform)
    )
    print("Results are:",results)
    gdf = gpd.GeoDataFrame.from_features(list(results))
    gdf.set_crs('EPSG:4326', inplace=True)
    print("GDF Heads are:")
    print(gdf.head())
    gdf.to_file(out_path)

#---------------------------------------------------------------------
def predict_tile_by_tile(raster_path):
    model.eval()

    pred_mask = np.zeros((dataset.height, dataset.width), dtype=np.uint8)

    for tile, x, y in generate_tiles(dataset):
    
        tile = tile / 255.0
        tile = torch.tensor(tile).permute(2, 0, 1).unsqueeze(0).float().cuda()

        with torch.no_grad():
            pred = model(tile)
            pred = torch.argmax(pred, dim=1).squeeze().cpu().numpy()

        h, w = pred.shape
        pred_mask[y:y+h, x:x+w] = pred

def save_predicted_output_GeoTIFF():
    with rasterio.open(
        "prediction.tif",
        "w",
        driver="GTiff",
        height=pred_mask.shape[0],
        width=pred_mask.shape[1],
        count=1,
        dtype=pred_mask.dtype,
        crs=dataset.crs,
        transform=dataset.transform,
    ) as dst:
        dst.write(pred_mask, 1)

def generate_tiles(dataset, tile_size=512, stride=512):
    for y in range(0, dataset.height, stride):
        for x in range(0, dataset.width, stride):
            window = rasterio.windows.Window(x, y, tile_size, tile_size)
            transform = dataset.window_transform(window)

            tile = dataset.read(window=window)  # shape: (C, H, W)
            tile = np.transpose(tile, (1, 2, 0))  # to HWC

            yield tile, x, y

# 2.78 GB files in live demo2
#mask, raster = predict_raster(PREDICT_DIR+"ANAITPURA_FATEHGARH SAHIB_32705_ORTHO.tif")

# 1.7 GB files in live demo3
#mask, raster = predict_raster(PREDICT_DIR+"BUTTAR_SIVIYA_AMRITSAR_37810_ORTHO.tif")
#file_list= glob.glob(os.path.join(OUTPUT_DIR+"tiles_to_test\/images\/", '*.tif'))
#for filepath in glob.glob(os.path.join(OUTPUT_DIR+"tiles_to_test\/images\/", '*.tif')) :
#    print("Raster file to predit:",filepath)
#    mask, raster = predict_raster(filepath)

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--predict_file", required=True,    help="path to input dataset")
#ap.add_argument("-m", "--model", required=True,    help="path to output model")
#ap.add_argument("-p", "--plot", type=str, default="plot.png",    help="path to output loss/accuracy plot")
args = vars(ap.parse_args())
if __name__ == "__main__":
    print("Print Class Map", CLASS_MAP)
    #print(f"Please provide file name to predict what is this shape {sys.argv[1:]}")
    file =sys.argv[1:]
    mask, raster = predict_raster(OUTPUT_DIR+"tiles_to_test\/images\/"+args["predict_file"])
    mask_to_shapefile(mask, raster, OUTPUT_DIR+"predictions/output.shp")
#mask, raster = predict_raster("C:\/My_D_Drive\/Customers\/Source_Code\/DataScience_Machine_Learning\/feature_extraction_from_tif\/ortho\/output\/patches\/images\/144.png")
#mask, raster = predict_raster("C:\My_D_Drive\Customers\Source_Code\DataScience_Machine_Learning\AI-Feature-Extraction-From-Drone-Orthophotos-main\/pexels-photo-417351.jpeg")
#mask, raster = predict_raster("C:\/My_D_Drive\/DIWANA_BARNALA_40082_ORTHO_01_01.tif")
