# -*- coding: utf-8 -*-

import os
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
import matplotlib
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
import pandas as pd
import gc
import random
import math
import glob
import torch
import gradio as gr
from PIL import Image
import cv2


classes = ['None','building','pervious surface','impervious surface','bare soil','water','coniferous','deciduous','brushwood','vineyard','herbaceous vegetation','agricultural land','plowed land']
id2label = pd.DataFrame(classes)[0].to_dict()
print(id2label)
label2id = {v: k for k, v in id2label.items()}
num_labels = len(id2label)

from transformers import SegformerForSemanticSegmentation, SegformerFeatureExtractor

segformer_b0_rgb_model = SegformerForSemanticSegmentation.from_pretrained("alanoix/segformer_b0_flair_one",
                                                                            num_labels=len(id2label),
                                                                            id2label=id2label,
                                                                            label2id=label2id)

segformer_rgb_feature_extractor = SegformerFeatureExtractor(ignore_index=0, reduce_labels=False, do_resize=False, do_rescale=False, do_normalize=False)
segformer_b0_rgb_model= torch.quantization.quantize_dynamic(segformer_b0_rgb_model, {torch.nn.Linear}, dtype=torch.qint8)


import albumentations as aug
MEAN = np.array([0.44050665, 0.45704361, 0.42254708])
STD = np.array([0.20264351, 0.1782405 , 0.17575739])

test_transform = aug.Compose([
    aug.Normalize(mean=MEAN, std=STD),
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
segformer_b0_rgb_model = segformer_b0_rgb_model.to(device)

class_colors = [(random.randint(0, 255), random.randint(
    0, 255), random.randint(0, 255)) for _ in range(5000)]


# Default IMAGE_ORDERING = channels_last
IMAGE_ORDERING = "channels_last"


def get_colored_segmentation_image(seg_arr, n_classes, colors=class_colors):
    output_height = seg_arr.shape[0]
    output_width = seg_arr.shape[1]

    seg_img = np.zeros((output_height, output_width, 3))

    for c in range(n_classes):
        seg_arr_c = seg_arr[:, :] == c
        seg_img[:, :, 0] += ((seg_arr_c)*(colors[c][0])).astype('uint8')
        seg_img[:, :, 1] += ((seg_arr_c)*(colors[c][1])).astype('uint8')
        seg_img[:, :, 2] += ((seg_arr_c)*(colors[c][2])).astype('uint8')

    return seg_img


def get_legends(class_names, colors=class_colors):

    n_classes = len(class_names)
    legend = np.zeros(((len(class_names) * 25) + 25, 125, 3),
                      dtype="uint8") + 255

    class_names_colors = enumerate(zip(class_names[:n_classes],
                                       colors[:n_classes]))

    for (i, (class_name, color)) in class_names_colors:
        color = [int(c) for c in color]
        cv2.putText(legend, class_name, (5, (i * 25) + 17),
                    cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1)
        cv2.rectangle(legend, (100, (i * 25)), (125, (i * 25) + 25),
                      tuple(color), -1)

    return legend


def overlay_seg_image(inp_img, seg_img):
    orininal_h = inp_img.shape[0]
    orininal_w = inp_img.shape[1]
    seg_img = cv2.resize(seg_img, (orininal_w, orininal_h), interpolation=cv2.INTER_NEAREST)

    fused_img = (inp_img/2 + seg_img/2).astype('uint8')
    return fused_img


def concat_lenends(seg_img, legend_img):

    new_h = np.maximum(seg_img.shape[0], legend_img.shape[0])
    new_w = seg_img.shape[1] + legend_img.shape[1]

    out_img = np.zeros((new_h, new_w, 3)).astype('uint8') + legend_img[0, 0, 0]

    out_img[:legend_img.shape[0], :  legend_img.shape[1]] = np.copy(legend_img)
    out_img[:seg_img.shape[0], legend_img.shape[1]:] = np.copy(seg_img)

    return out_img


def visualize_segmentation(seg_arr, inp_img=None, n_classes=None,
                           colors=class_colors, class_names=None,
                           overlay_img=False, show_legends=False,
                           prediction_width=None, prediction_height=None):

    if n_classes is None:
        n_classes = np.max(seg_arr)

    seg_img = get_colored_segmentation_image(seg_arr, n_classes, colors=colors)

    if inp_img is not None:
        original_h = inp_img.shape[0]
        original_w = inp_img.shape[1]
        seg_img = cv2.resize(seg_img, (original_w, original_h), interpolation=cv2.INTER_NEAREST)

    if (prediction_height is not None) and (prediction_width is not None):
        seg_img = cv2.resize(seg_img, (prediction_width, prediction_height), interpolation=cv2.INTER_NEAREST)
        if inp_img is not None:
            inp_img = cv2.resize(inp_img,
                                 (prediction_width, prediction_height))

    if overlay_img:
        assert inp_img is not None
        seg_img = overlay_seg_image(inp_img, seg_img)

    if show_legends:
        assert class_names is not None
        legend_img = get_legends(class_names, colors=colors)

        seg_img = concat_lenends(seg_img, legend_img)

    return seg_img

def query_image(img):
    image_to_pred = test_transform(image=img)['image']

    pixel_values = segformer_rgb_feature_extractor(image_to_pred, return_tensors="pt").pixel_values.to(device)

    outputs_segformer_b0_rgb = segformer_b0_rgb_model(pixel_values=pixel_values)
    pred_segformer_b0_rgb = outputs_segformer_b0_rgb.logits.cpu().detach().numpy()

    pred = np.mean(np.array([K.softmax(pred_segformer_b0_rgb, axis = 1)]), axis = 0)
    pred = tf.image.resize(tf.transpose(pred, perm=[0,2,3,1]), size = [512,512], method="bilinear") # resize to 512*512
    pred = np.argmax(pred, axis = -1)
    pred =np.squeeze(pred)
    result = pred.astype(np.uint8)

    class_names = [ 'None', 'building', 'pervious surface', 'impervious surface', 'bare soil','water','coniferous','deciduous','brushwood','vineyard', 'herbaceous vegetation', 'agricultural land', 'plowed land']
    seg_img = visualize_segmentation(result, img, n_classes=13,
                                     colors=class_colors , overlay_img=True,
                                     show_legends=True,
                                     class_names=class_names,
                                     prediction_width=512,
                                     prediction_height=512)  
    
    return seg_img

demo = gr.Interface(
    
    query_image, 
    inputs=[gr.Image()], 
    outputs="image",
    title="Image Segmentation Demo",
    description = "Please upload an image to see segmentation capabilities of this model",
    examples=["examples/IMG_011942.jpeg","examples/IMG_005339.jpeg","examples/IMG_004753.jpeg","examples/IMG_011617.jpeg","examples/IMG_003022.jpeg"]
)

demo.launch() #debug=True