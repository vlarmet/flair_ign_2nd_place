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
BATCH_SIZE = 4

# Create working folder
for folder in [OUTPUT_DIR]:
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



################################################## LOAD MODELS
models_to_load = {
    'unet_efficientnetv2s' : False,
    'unetpp_convnext' : True,
    'segformerb0_5c' : False,
    'segformerb1_5c':False,
    'segformerb0_rgb' : True,
    'segformerb1_rgb' : False,
    'segformerb2_rgb' : False,
    'segformerb3_rgb' : False,
    'segformerb4_rgb' : False
}

list_of_models = inf.load_models(**models_to_load)

# Read first image for getting projection and geotransform
im1 = gdal.Open(img_paths[0])

# Predict all tiles
for i in tqdm(range(0, len(img_paths), BATCH_SIZE)):

    images = [np.expand_dims(gdal.Open(img_path).ReadAsArray().transpose((1,2,0)) , axis = 0) for img_path in img_paths[i:i+BATCH_SIZE]]
    images = np.concatenate(images, axis = 0)
    names = [img_path.split("/")[-1][4:] for img_path in img_paths[i:i+BATCH_SIZE]]
    pred = inf.predict_ensemble(list_of_models, images, target_size = 512)
    pred = np.argmax(pred, axis = -1)
    # export
    for index, name in enumerate(names):
        driver = gdal.GetDriverByName('GTiff')
        dataset = driver.Create(OUTPUT_DIR + "PRED_" + name, 512, 512, 1, gdal.GDT_Byte)
        dataset.GetRasterBand(1).WriteArray(pred[index,:,:])
        proj = im1.GetProjection() #you can get from a exsited tif or import 
        dataset.SetGeoTransform(list(im1.GetGeoTransform() ))
        dataset.SetProjection(proj)
        dataset.FlushCache()
        dataset=None
