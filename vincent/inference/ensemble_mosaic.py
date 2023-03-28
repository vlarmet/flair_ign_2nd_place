import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import backend as K
from osgeo import gdal
from tqdm import tqdm


# Import functions defined in utils/inference.py
file_path = os.path.realpath(__file__)
file_root = "/".join(file_path.replace("\\", "/").split("/")[:-2])
sys.path.append(file_root + "/utils")

import inference as inf

# Global parameters

DATA_DIR = "C:/Users/vincent/Documents/flair/test/"
OUTPUT_DIR = "C:/Users/vincent/Documents/flair/predictions_test/"
VRT_FOLDER = "C:/Users/vincent/Documents/flair/mosaic/"
MOSAIC_PREDS = "C:/Users/vincent/Documents/flair/predictions_mosaic/"
IMAGE_SIZE = 512
STRIDE = 128

# Create working folder
for folder in [OUTPUT_DIR, VRT_FOLDER, MOSAIC_PREDS]:
    if not os.path.exists(folder):
        os.makedirs(folder)

# Test image paths
img_paths = []
for dep in os.listdir(DATA_DIR):
    for zone in os.listdir("/".join([DATA_DIR, dep])):
        for img in os.listdir("/".join([DATA_DIR, dep, zone, "img"])):
            if img.__contains__("xml"):
                continue
            img_path = "/".join([DATA_DIR, dep, zone, "img", img])
            img_paths.append(img_path)


patches = list(set(["/".join(img.split("/")[-4:-2]) for img in img_paths]))

# Build all vrts
for patch in patches:
    output = VRT_FOLDER + patch.replace('/', '_') + ".vrt"
    img_list = [img for img in img_paths if patch in img]
    my_vrt = gdal.BuildVRT(output, img_list)
    my_vrt = None


# dictionnaire qui stocke les emprises de chaque image
emprise = {}
for patch in patches:
    img_list = [img for img in img_paths if patch in img]
    emprise2 = {}
    for img in img_list:
        file_name = img.split("/")[-1][:-4]
        ds = gdal.Open(img)
        ulx, _,_, uly, _,_ = ds.GetGeoTransform()
        emprise2[file_name] = (ulx, uly)
    emprise[patch] = emprise2

################################################## LOAD MODELS
models_to_load = {
    'unet_efficientnetv2s' : True,
    'unetpp_convnext' : True,
    'segformerb0_5c' : True,
    'segformerb1_5c':True,
    'segformerb0_rgb' : True,
    'segformerb1_rgb' : True,
    'segformerb2_rgb' : True,
    'segformerb3_rgb' : True,
    'segformerb4_rgb' : True
}

list_of_models = inf.load_models(**models_to_load)

# Predict tiles 

for i, patch in tqdm(enumerate(patches)):

    path = MOSAIC_PREDS + patch.replace('/', '_') + '/' 
    if not os.path.exists(path):
        os.makedirs(path)
    # predictions 
    vrt_path = VRT_FOLDER + patch.replace('/', '_') + '.vrt'
    a = 0
    for i in range(int(IMAGE_SIZE/STRIDE)):
        for j in range(int(IMAGE_SIZE/STRIDE)):
            a = a+1
            inf.pred_tif(vrt_path = vrt_path,
            stride = (i*STRIDE, j*STRIDE), 
            size= IMAGE_SIZE, 
            tile_path = path, output_name = "all" + str(a),
            predict_func = inf.predict_ensemble,
            models = list_of_models)
    
    
    # majority voting
    tiles = os.listdir(path)
    tiles = [path + tile for tile in tiles if ".tif"  in tile and "all" in tile]
    inf.merge_all(file_list = tiles, output = path + "final.tif")


# Split prediction tiles into original 512*512 patches

for i, patch in enumerate(patches):
    path = MOSAIC_PREDS + patch.replace('/', '_') + '/'
    bigimg = gdal.Open(path + 'final.tif')
    ULX,_,_,ULY,_,_ = bigimg.GetGeoTransform()
    
    for img in emprise[patch].keys():
        ulx, uly = emprise[patch][img]
        arr = bigimg.ReadAsArray(
            xoff = round((ulx-ULX)/0.2),
            yoff = abs(round((uly-ULY)/0.2)),
            xsize = 512, ysize = 512
        )

        driver = gdal.GetDriverByName('GTiff')
        dataset = driver.Create(OUTPUT_DIR + img.replace('IMG', 'PRED') + '.tif', 512, 512, 1, gdal.GDT_Byte)
        dataset.GetRasterBand(1).WriteArray(arr)
        proj = bigimg.GetProjection() 
        dataset.SetProjection(proj)
        dataset.FlushCache()
        dataset=None