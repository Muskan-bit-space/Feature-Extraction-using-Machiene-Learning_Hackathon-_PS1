# predict.py

import torch
import rasterio
import numpy as np
import geopandas as gpd
from rasterio.features import shapes
import segmentation_models_pytorch as smp
from config import *
from rasterio.transform import from_origin

import os
import cv2
from config import PATCH_SIZE, STRIDE,TEST_PATCH_SIZE, TEST_STRIDE

device = "cuda" if torch.cuda.is_available() else "cpu"

model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights=None,
    in_channels=3,
    classes=len(CLASS_MAP)
)

#model.load_state_dict(torch.load(OUTPUT_DIR+"models/unet.pth"))
model.load_state_dict(torch.load(OUTPUT_DIR+"models/unet_25.pth"))
model.to(device)
model.eval()


def predict_raster(raster_path):
    raster = rasterio.open(raster_path)
    print("Input Raster height:", raster.height, "Input Raster width:", raster.width)
    img = raster.read([1, 2, 3])
    img = np.transpose(img, (1, 2, 0))

    img_tensor = torch.tensor(img).permute(2,0,1).float().unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(img_tensor)
        #print("Here we are seeing what is model predict:",torch.max(pred, 1))
        pred_mask = torch.argmax(pred, dim=1).squeeze().cpu().numpy()
        print("Here we are seeing what is model predict Mask:",pred_mask)

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

#------------------------------------------------------
def load_raster(raster_path):
    print("I am here in load_raster----------------")
    return rasterio.open(raster_path)

def raster_to_array(raster):
    print("I am here in raster_to_array----------------")
    img = raster.read([1, 2, 3])
    img = np.transpose(img, (1, 2, 0))
    return img


def create_patches(image,  out_dir):
    print("I am here in create patches----------------",image,out_dir)
    img_dir = os.path.join(out_dir, "images")
    #mask_dir = os.path.join(out_dir, "masks")

    os.makedirs(img_dir, exist_ok=True)
    #os.makedirs(mask_dir, exist_ok=True)

    idx = 0

    for i in range(0, image.shape[0] - TEST_PATCH_SIZE, TEST_STRIDE):
        for j in range(0, image.shape[1] - TEST_PATCH_SIZE, TEST_STRIDE):

            img_patch = image[i:i+TEST_PATCH_SIZE, j:j+TEST_PATCH_SIZE]
            #mask_patch = mask[i:i+PATCH_SIZE, j:j+PATCH_SIZE]
            

            #if np.sum(mask_patch) == 0:
                #continue  # skip empty patches
            
            #mask_patch = mask_patch * 255
            cv2.imwrite(f"{img_dir}/{idx}.tif", img_patch)
            #cv2.imwrite(f"{mask_dir}/{idx}.png", mask_patch)

            idx += 1
#-------------------------------------------------------------
def predict_tile_by_tile(raster_path):
    print("Inside----------predict_tile_by_tile")
    raster = rasterio.open(raster_path)
    model.eval()
    
    pred_mask = np.zeros((raster.height, raster.width), dtype=np.uint8)

    for tile, x, y in generate_tiles(raster):    
        tile = tile / 255.0
        tile = torch.tensor(tile).permute(2, 0, 1).unsqueeze(0).float().to(device)
        with torch.no_grad():
            pred = model(tile)
            pred = torch.argmax(pred, dim=1).squeeze().cpu().numpy()

        h, w = pred.shape
        pred_mask[y:y+h, x:x+w] = pred



def generate_tiles(dataset, tile_size=1024, stride=1024):
    #dataset = rasterio.open(dataset_path)
    print("Inside----------predict_tile_by_tile")
    for y in range(0, dataset.height, stride):
        for x in range(0, dataset.width, stride):
            window = rasterio.windows.Window(x, y, tile_size, tile_size)
            transform = dataset.window_transform(window)

            tile = dataset.read(window=window)  # shape: (C, H, W)
            tile = np.transpose(tile, (1, 2, 0))  # to HWC

            yield tile, x, y

def save_predicted_output_GeoTIFF(raster, out_path):
    print("Inside----------save predicted output")
    with rasterio.open(
        "prediction.tif",
        "w",
        driver="GTiff",
        height=pred_mask.shape[0],
        width=pred_mask.shape[1],
        count=1,
        dtype=pred_mask.dtype,
        crs=dataset.crs,
        transform=raster.transform,
    ) as dst:
        dst.write(pred_mask, 1)

raster = load_raster(PREDICT_DIR+"BUTTAR_SIVIYA_AMRITSAR_37810_ORTHO.tif")
image = raster_to_array(raster)
create_patches(image, OUTPUT_DIR+"tiles_to_test")

#generate_tiles(PREDICT_DIR+"BUTTAR_SIVIYA_AMRITSAR_37810_ORTHO.tif",tile_size=512, stride=512)
#predict_tile_by_tile(PREDICT_DIR+"BUTTAR_SIVIYA_AMRITSAR_37810_ORTHO.tif")
# 2.78 GB files in live demo2
#mask, raster = predict_raster(PREDICT_DIR+"ANAITPURA_FATEHGARH SAHIB_32705_ORTHO.tif")

# 1.7 GB files in live demo3
#mask, raster = predict_raster(OUTPUT_DIR+"tiles_to_test"+"\/388.tif")

#mask, raster = predict_raster("C:\/My_D_Drive\/Customers\/Source_Code\/DataScience_Machine_Learning\/feature_extraction_from_tif\/ortho\/output\/patches\/images\/144.png")
#mask, raster = predict_raster("C:\My_D_Drive\Customers\Source_Code\DataScience_Machine_Learning\AI-Feature-Extraction-From-Drone-Orthophotos-main\/pexels-photo-417351.jpeg")
#mask, raster = predict_raster("C:\/My_D_Drive\/DIWANA_BARNALA_40082_ORTHO_01_01.tif")
#mask_to_shapefile(mask, raster, OUTPUT_DIR+"predictions/output.shp")
#

#save_predicted_output_GeoTIFF(raster, OUTPUT_DIR+"predictions/prediction.tif")