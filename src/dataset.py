# dataset.py

import numpy as np
import os
import cv2
from config import PATCH_SIZE, STRIDE,TEST_PATCH_SIZE, TEST_STRIDE


def raster_to_array(raster):
    img = raster.read([1, 2, 3])
    img = np.transpose(img, (1, 2, 0))
    return img


def create_patches(image, mask, out_dir):
    img_dir = os.path.join(out_dir, "images")
    mask_dir = os.path.join(out_dir, "masks")

    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    idx = 0

    for i in range(0, image.shape[0] - PATCH_SIZE, STRIDE):
        for j in range(0, image.shape[1] - PATCH_SIZE, STRIDE):

            img_patch = image[i:i+PATCH_SIZE, j:j+PATCH_SIZE]
            mask_patch = mask[i:i+PATCH_SIZE, j:j+PATCH_SIZE]
            

            if np.sum(mask_patch) == 0:
                continue  # skip empty patches
            
            #mask_patch = mask_patch * 255
            cv2.imwrite(f"{img_dir}/{idx}.png", img_patch)
            cv2.imwrite(f"{mask_dir}/{idx}.png", mask_patch)

            idx += 1