# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np
from glob import glob
from scipy.io import loadmat
import matplotlib.pyplot as plt
import matplotlib
import torch
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import backend as K
from osgeo import gdal
import pandas as pd
import gc
import math
from transformers import TFSegformerForSemanticSegmentation, SegformerConfig


"""## Read Data"""

DATA_DIR = "flaire-one/data/test"
np.random.seed(123)
def to_categorical(a):
  classes = np.arange(1,14)
  a_array = [(a == v) for v in classes]  #extract
  a = np.stack(a_array,axis=-1).astype("float")  #stack
  return a

def read_image(image_path, mask=False, resize = True):
    
    im = gdal.Open(image_path)
    if mask:
        image = im.ReadAsArray().transpose()
        image = np.where(np.isin(image, [19,13,14,15,16,17,18]), 13, image) - 1
        if resize:
             image = cv2.resize(image, (256,256))
    else:
        image = im.ReadAsArray().transpose().astype(np.float32)        
        image = image / 255.0
        if resize:
             image = cv2.resize(image, (256,256))
    im = None
    return image

img_paths = []
for dep in os.listdir(DATA_DIR):
    for zone in os.listdir("/".join([DATA_DIR, dep])):
        for img in os.listdir("/".join([DATA_DIR, dep, zone, "img"])):
            if img.__contains__("xml"):
                continue
            img_path = "/".join([DATA_DIR, dep, zone, "img", img])
            img_paths.append(img_path)


"""## Create Patches"""

patches = list(set(["/".join(img.split("/")[3:5]) for img in img_paths]))

# On crée les gros patchs grace au géoréférencement des petits patchs 512*512
PATH = "data/mosaic/"
# vrts
for patch in patches:
    output = PATH + patch.replace('/', '_') + ".vrt"
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

"""## Import models"""

segformer_b0_5c_model_name = "models/segformer_b0_5c/segformer"
segformer_b5_5c_model_name = "models/segformer_b5_5c/segformer"
segformer_b0_rgb_model_name = "models/segformer_b0_rgb"
segformer_b5_rgb_model_name = "models/segformer_b5_rgb"
mask2former-large-ade-semantic_model_name = "models/mask2former-swin-large-ade-semantic/mask2former-large-ade-semantic.zip"
mask2former-swin-base-ade-semantic_model_name = "models/mask2former-swin-base-ade-semantic/mask2former-swin-base-ade-semantic"

classes = ['None','building','pervious surface','impervious surface','bare soil','water','coniferous','deciduous','brushwood','vineyard','herbaceous vegetation','agricultural land','plowed land']
id2label = pd.DataFrame(classes)[0].to_dict()
label2id = {v: k for k, v in id2label.items()}

num_labels = len(id2label)
num_labels


"""### Segformer 5C"""

model_temp = TFSegformerForSemanticSegmentation.from_pretrained(
    "nvidia/mit-b0",
    num_labels=13
)
new_config = model_temp.config
# print(new_config)
new_config.num_channels = 5
segformer_b0_5c_model = TFSegformerForSemanticSegmentation(new_config)
segformer_b0_5c_model.build(input_shape=(1,5,512,512))
segformer_b0_5c_model.load_weights(segformer_b0_5c_model_name)
del model_temp


model_temp = TFSegformerForSemanticSegmentation.from_pretrained(
    "nvidia/mit-b5",
    num_labels=13
)
new_config = model_temp.config
# print(new_config)
new_config.num_channels = 5
segformer_b5_5c_model = TFSegformerForSemanticSegmentation(new_config)
segformer_b5_5c_model.build(input_shape=(1,5,512,512))
segformer_b5_5c_model.load_weights(segformer_b5_5c_model_name)
del model_temp


"""### Segformers rgb"""

from transformers import SegformerForSemanticSegmentation, SegformerFeatureExtractor
segformer_b0_rgb_model = SegformerForSemanticSegmentation.from_pretrained(segformer_b0_rgb_model_name,
                                                                            num_labels=len(id2label),
                                                                            id2label=id2label,
                                                                            label2id=label2id)

from transformers import SegformerForSemanticSegmentation, SegformerFeatureExtractor
segformer_b5_rgb_model = SegformerForSemanticSegmentation.from_pretrained(segformer_b5_rgb_model_name,
                                                                            num_labels=len(id2label),
                                                                            id2label=id2label,
                                                                            label2id=label2id)

segformer_rgb_feature_extractor = SegformerFeatureExtractor(ignore_index=0, reduce_labels=False, do_resize=False, do_rescale=False, do_normalize=False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

segformer_b0_rgb_model = segformer_b0_rgb_model.to(device)
segformer_b5_rgb_model = segformer_b5_rgb_model.to(device)


"""## Mask2former"""

from transformers import MaskFormerImageProcessor
# Create a preprocessor
preprocessor = MaskFormerImageProcessor(ignore_index=0, reduce_labels=False, do_resize=False, do_rescale=False, do_normalize=False)

# Replace the head of the pre-trained model
from transformers import Mask2FormerForUniversalSegmentation
mask2former_swin_large_ade_semantic_model = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-large-cityscapes-semantic",
                                                          id2label=id2label,
                                                          ignore_mismatched_sizes=True)
mask2former_swin_large_ade_semantic_model.load_state_dict(torch.load(mask2former-large-ade-semantic_model_name))   
mask2former_swin_large_ade_semantic_model = mask2former_swin_large_ade_semantic_model.to(device)

from transformers import Mask2FormerForUniversalSegmentation
# Replace the head of the pre-trained model
mask2former_swin_base_ade_semantic_model = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-base-IN21k-ade-semantic",
                                                          id2label=id2label,
                                                          ignore_mismatched_sizes=True)
mask2former_swin_base_ade_semantic_model.load_state_dict(torch.load(mask2former-swin-base-ade-semantic_model_name))  
mask2former_swin_base_ade_semantic_model = mask2former_swin_base_ade_semantic_model.to(device)


"""## Inference sur la grille"""

def pred_tif_sf(rgb_path, stride = (256,256), size = 512, tile_path = "data/preds/", output_name = "all"):
    stride_x, stride_y = stride
    rgb = gdal.Open(rgb_path)
    gt = list(rgb.GetGeoTransform())
    originY = gt[3]
    originX = gt[0]
    width, height = rgb.RasterXSize, rgb.RasterYSize
    for row in range(stride_y, height, size): #range(0, height, stride)
        # print(row)
        res = []
        res_5C = []
        if row + size > height:
            break
        for col in range(stride_x, width, size):
            if col + size > width:
                break
            arr = rgb.ReadAsArray(xoff=col, yoff=row, xsize=size, ysize=size).astype(np.float32)
            arr = arr/255.0
            arr = np.expand_dims(arr, axis = 0) # 1,256,256,5
            res.append(arr)
        res = np.concatenate(res, axis = 0)
        for channel,avg,std in zip(
            [0,1,2,3,4],
            [0.44050665, 0.45704361, 0.42254708, 0.40987858, 0.06875153],
            [0.20264351, 0.1782405 , 0.17575739, 0.15510736, 0.11867123]):
            res[:,channel,:,:] = (res[:,channel,:,:] - avg)/std
        res_rgb = res[:,:3,:,:] 


        ######################################################################################################
        # MASK2FORMER large
        results = []
        for image in res_rgb:
            pixel_values = preprocessor (image, return_tensors="pt").pixel_values.to(device)
            with torch.no_grad():
                outputs = mask2former_swin_large_ade_semantic_model(pixel_values=pixel_values)
            target_sizes = [(512, 512)]

            pred_mask2former_ade_large_tp =  preprocessor.post_process_semantic_segmentation(outputs, target_sizes=target_sizes)
            pred_mask2former_ade_large_tp = np.array(pred_mask2former_ade_large_tp[0].cpu().detach().numpy()) - 1 

            class_queries_logits = outputs.class_queries_logits  # [batch_size, num_queries, num_classes+1]
            masks_queries_logits = outputs.masks_queries_logits  # [batch_size, num_queries, height, width]

            # Scale back to preprocessed image size - (384, 384) for all models
            masks_queries_logits = torch.nn.functional.interpolate(
                masks_queries_logits, size=(384, 384), mode="bilinear", align_corners=False)

            # Remove the null class `[..., :-1]`
            masks_classes = class_queries_logits.softmax(dim=-1)[..., :-1]
            masks_probs = masks_queries_logits.sigmoid()  # [batch_size, num_queries, height, width]

            # Semantic segmentation logits of shape (batch_size, num_classes, height, width)
            segmentation = torch.einsum("bqc, bqhw -> bchw", masks_classes, masks_probs)
            batch_size = class_queries_logits.shape[0]

            # Resize logits and compute semantic segmentation maps
            if target_sizes is not None:
                semantic_segmentation = []
                for idx in range(batch_size):
                    resized_logits = torch.nn.functional.interpolate(segmentation[idx].unsqueeze(dim=0), size=target_sizes[idx], mode="bilinear", align_corners=False
                        )
                    semantic_map = resized_logits[0]   #.argmax(dim=0)
                    semantic_segmentation.append(semantic_map)
            pred_mask2former_ade_large =  semantic_segmentation # preprocessor.post_process_semantic_segmentation(outputs, target_sizes=target_sizes)
            pred_mask2former_ade_large = np.array(pred_mask2former_ade_large[0].cpu().detach().numpy())
            pred_mask2former_ade_large = tf.transpose(np.expand_dims(pred_mask2former_ade_large, axis = 0), perm=[0,2,3,1])
            results.append(np.squeeze(pred_mask2former_ade_large))
        pred_mask2former_ade_large = results


        ######################################################################################################
        # MASK2FORMER base
        results = []
        for image in res_rgb:
            pixel_values = preprocessor (image, return_tensors="pt").pixel_values.to(device)
            with torch.no_grad():
                outputs = mask2former_swin_base_ade_semantic_model(pixel_values=pixel_values)
            target_sizes = [(512, 512)]

            pred_mask2former_ade_base_tp =  preprocessor.post_process_semantic_segmentation(outputs, target_sizes=target_sizes)
            pred_mask2former_ade_base_tp = np.array(pred_mask2former_ade_base_tp[0].cpu().detach().numpy()) - 1 

            class_queries_logits = outputs.class_queries_logits  # [batch_size, num_queries, num_classes+1]
            masks_queries_logits = outputs.masks_queries_logits  # [batch_size, num_queries, height, width]

            # Scale back to preprocessed image size - (384, 384) for all models
            masks_queries_logits = torch.nn.functional.interpolate(
                masks_queries_logits, size=(384, 384), mode="bilinear", align_corners=False)

            # Remove the null class `[..., :-1]`
            masks_classes = class_queries_logits.softmax(dim=-1)[..., :-1]
            masks_probs = masks_queries_logits.sigmoid()  # [batch_size, num_queries, height, width]

            # Semantic segmentation logits of shape (batch_size, num_classes, height, width)
            segmentation = torch.einsum("bqc, bqhw -> bchw", masks_classes, masks_probs)
            batch_size = class_queries_logits.shape[0]

            # Resize logits and compute semantic segmentation maps
            if target_sizes is not None:
                semantic_segmentation = []
                for idx in range(batch_size):
                    resized_logits = torch.nn.functional.interpolate(segmentation[idx].unsqueeze(dim=0), size=target_sizes[idx], mode="bilinear", align_corners=False
                        )
                    semantic_map = resized_logits[0]   #.argmax(dim=0)
                    semantic_segmentation.append(semantic_map)
            pred_mask2former_ade_base =  semantic_segmentation # preprocessor.post_process_semantic_segmentation(outputs, target_sizes=target_sizes)
            pred_mask2former_ade_base = np.array(pred_mask2former_ade_base[0].cpu().detach().numpy())
            pred_mask2former_ade_base = tf.transpose(np.expand_dims(pred_mask2former_ade_base, axis = 0), perm=[0,2,3,1])
            results.append(np.squeeze(pred_mask2former_ade_base))
        pred_mask2former_ade_base = results


        ######################################################################################################
        # Segformer b0 rgb
        results = []
        for image in res_rgb:
            pixel_values = segformer_rgb_feature_extractor(image, return_tensors="pt").pixel_values.to(device)
            outputs_segformer_b0_rgb = segformer_b0_rgb_model(pixel_values=pixel_values)# logits are of shape (batch_size, num_labels, height/4, width/4)
            pred_segformer_b0_rgb = outputs_segformer_b0_rgb.logits.cpu().detach().numpy()
            pred_segformer_b0_rgb = tf.image.resize(tf.transpose(pred_segformer_b0_rgb, perm=[0,2,3,1]), size = [512,512], method="bilinear") # resize to 512*512  
            results.append(np.squeeze(pred_segformer_b0_rgb))
        pred_segformer_b0_rgb = results


        ######################################################################################################
        # Segformer b5 rgb
        results = []
        for image in res_rgb:
            pixel_values = segformer_rgb_feature_extractor(image, return_tensors="pt").pixel_values.to(device)
            outputs_segformer_b5_rgb = segformer_b5_rgb_model(pixel_values=pixel_values)# logits are of shape (batch_size, num_labels, height/4, width/4)
            pred_segformer_b5_rgb = outputs_segformer_b5_rgb.logits.cpu().detach().numpy()
            pred_segformer_b5_rgb = tf.image.resize(tf.transpose(pred_segformer_b5_rgb, perm=[0,2,3,1]), size = [512,512], method="bilinear") # resize to 512*512
            results.append(np.squeeze(pred_segformer_b5_rgb))
        pred_segformer_b5_rgb = results


        ######################################################################################################
        # Segformer b0 5c
        pred_segformer_b0_5c = segformer_b0_5c_model.predict(res, batch_size=1)
        pred_segformer_b0_5c = list(pred_segformer_b0_5c.values())[0]
        pred_segformer_b0_5c = pred_segformer_b0_5c[:,[12,0,1,2,3,4,5,6,7,8,9,10,11],:,:]
        pred_segformer_b0_5c = tf.image.resize(tf.transpose(pred_segformer_b0_5c, perm=[0,2,3,1]), size = [512,512], method="bilinear") # resize to 512*512


        ######################################################################################################
        # Segformer b5 5c
        pred_segformer_b5_5c = segformer_b5_5c_model.predict(res, batch_size=1)
        pred_segformer_b5_5c = list(pred_segformer_b5_5c.values())[0]
        pred_segformer_b5_5c = pred_segformer_b5_5c[:,[12,0,1,2,3,4,5,6,7,8,9,10,11],:,:]
        pred_segformer_b5_5c = tf.image.resize(tf.transpose(pred_segformer_b5_5c, perm=[0,2,3,1]), size = [512,512], method="bilinear") # resize to 512*512


        ######################################################################################################
        # Mean predictio,
        preds = np.mean(np.array([pred_segformer_b0_rgb,
                                  pred_segformer_b5_5c,
                                  pred_segformer_b5_rgb,
                                  pred_segformer_b0_5c,
                                  pred_mask2former_ade_large,
                                  pred_mask2former_ade_base,
                                ]), axis = 0)

        preds = [np.argmax(preds[index,:,:,:], axis = -1).transpose((0,1)) for index in range(preds.shape[0])]
        preds = np.array(preds)-1
        #print(np.array(preds).shape)

        outfile = tile_path + str(row) + ".tif"
        driver = gdal.GetDriverByName('GTiff')
        dataset = driver.Create(outfile, col + size, size, 1, gdal.GDT_Byte)
        preds = np.hstack(preds)
        dataset.GetRasterBand(1).WriteArray(preds + 1) # +1 pour reconnaitre les zones non predites(=0)

        # follow code is adding GeoTranform and Projection
        gt[3] = originY + row * gt[5]
        gt[0] = originX + stride_x * gt[1]
        proj = rgb.GetProjection() #you can get from a exsited tif or import 
        dataset.SetGeoTransform(gt)
        dataset.SetProjection(proj)
        dataset.FlushCache()
        dataset=None
        _ = gc.collect()
    
    rgb =  None
    #print( ["".join([tile_path,i]) for i in os.listdir(tile_path) if "al" not in i and "rf" not in i])
    ds = gdal.BuildVRT(srcDSOrSrcDSTab = ["".join([tile_path,i]) for i in os.listdir(tile_path) if "al" not in i and "rf" not in i], destName = tile_path + output_name + ".vrt")
    ds = None

    ds = gdal.Warp(srcDSOrSrcDSTab=tile_path + output_name + ".vrt", 
    destNameOrDestDS=tile_path + output_name + ".tif", 
    outputType  = gdal.gdalconst.GDT_Byte, 
    multithread =True, srcSRS = "+proj=lcc +lat_1=49 +lat_2=44 +lat_0=46.5 +lon_0=3 +x_0=700000 +y_0=6600000 +ellps=GRS80 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs",
    outputBounds = (originX, originY + height * gt[5], originX + width * gt[1], originY))
    ds = None

    # effacer tuiles
    tiles = [os.remove(tile_path + i) for i in os.listdir(tile_path) if "al" not in i and "rf" not in i]
    return 



def merge_all(file_list, output):
    
    driver = gdal.GetDriverByName('GTiff')
    ds1 = gdal.Open(file_list[0])

    dataset = driver.Create(output, ds1.RasterXSize, ds1.RasterYSize, 1, gdal.GDT_Byte)
        
    step = int(ds1.RasterXSize/10)
    for col in range(0, ds1.RasterXSize, step):
        if col + step > ds1.RasterXSize:
            step = ds1.RasterXSize - col
        #print(col)
        arrs = np.concatenate(list(map(lambda x: np.expand_dims(gdal.Open(x).ReadAsArray(xoff = col, yoff= 0, xsize= step), axis = 2), file_list)), axis = -1)
        
        arr2 = np.concatenate([np.expand_dims((arrs == i).sum(axis = 2), axis = 2) for i in range(1,14)], axis = -1)
        arr2 = np.argmax(arr2, axis = -1)

        dataset.GetRasterBand(1).WriteArray(arr2, xoff = col, yoff= 0)
        dataset.FlushCache()
        gc.collect()


    dataset.SetGeoTransform(ds1.GetGeoTransform())
    dataset.SetProjection(ds1.GetProjection())
    
    dataset=ds1 = None

len(patches)

import os
subdirs = [x[0] for x in os.walk('data/predictions_mosaic/')]
print(len(subdirs))

PATH = "data/mosaic/"
vrts = [PATH + file_name for file_name in os.listdir(PATH)]
PRED_PATH = "data/predictions_mosaic/"


for i, patch in enumerate(patches):
    check = PRED_PATH + patch.replace('/', '_')
    print("patch: ", patch," " , i)
    if check in subdirs:
        print('pass')
        pass
        
    else:
        # print("patch: ", patch," " , i)
        path = PRED_PATH + patch.replace('/', '_') + '/' 
        if not os.path.exists(path):
            os.makedirs(path)
        # predictions 
        vrt_path = PATH + patch.replace('/', '_') + '.vrt'
        a = 0
        for i in range(4):
            for j in range(4):
                a = a+1
                pred_tif_sf(rgb_path = vrt_path,
                stride = (i*128, j*128), size= 512, 
                tile_path = path, output_name = "all" + str(a))
        
        
        # merge
        tiles = os.listdir(path)
        tiles = [path + tile for tile in tiles if ".tif"  in tile and "all" in tile]
        merge_all(file_list = tiles, output = path + "final.tif")
        #clear_output()

# unpatch
PATH2 = 'flair-one/data/predictions/'

for i, patch in enumerate(patches):
    path = PRED_PATH + patch.replace('/', '_') + '/'
    print(patch, path)
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
        dataset = driver.Create(PATH2 + img.replace('IMG', 'PRED') + '.tif', 512, 512, 1, gdal.GDT_Byte)
        dataset.GetRasterBand(1).WriteArray(arr)
        proj = bigimg.GetProjection() #you can get from a exsited tif or import 
        dataset.SetProjection(proj)
        dataset.FlushCache()
        dataset=None
