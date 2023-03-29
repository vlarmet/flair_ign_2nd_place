# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np
from glob import glob
from scipy.io import loadmat
import matplotlib.pyplot as plt
import matplotlib
import albumentations as aug
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import backend as K
from osgeo import gdal
import pandas as pd
import gc
import math
import json
from transformers import TFSegformerForSemanticSegmentation, SegformerConfig
import shutil
import splitfolders
from skimage import img_as_float


"""## data augmentation"""

MEAN = np.array([0.44050665, 0.45704361, 0.42254708, 0.40987858, 0.06875153])
STD = np.array([0.20264351, 0.1782405 , 0.17575739, 0.15510736, 0.11867123])

train_transform = aug.Compose([
    aug.VerticalFlip(p=0.5),
    aug.HorizontalFlip(p=0.5),
    aug.RandomRotate90(p=0.5),
    aug.Normalize(mean=MEAN, std=STD),

])

test_transform = aug.Compose([
    aug.Normalize(mean=MEAN, std=STD),
])

with open(metadata) as f:
    data = json.load(f)

np.random.seed(random_state)

def to_categorical(a):
  classes = np.arange(13)
  a_array = [(a == v) for v in classes]  #extract
  a = np.stack(a_array,axis=-1).astype("float")  #stack
  return a

def read_image(image_path, mask=False):
    im = gdal.Open(image_path)
    if mask:
        image = im.ReadAsArray()
        image = np.where(np.isin(image, [19,13,14,15,16,17,18]), 13, image) - 1
    else:
        image = im.ReadAsArray().astype(np.float32)     
    im = None
    return image


def get_metadata(img_id):
  CAMERAS = ["CAMERA#030", "UCE-M3-f120-s06", "UCE-M3-f120-s08", "UCE-M3-f120-s07", "CAMERA#020", "CAMERA#017", "UCE-M3-f120", "CAMERA#034", "UCE-M3-f100"]
  res = []
  dat = data[img_id]
  # x, y, z
  res.append((dat["patch_centroid_x"] -  1.352762e+05)/ (1.079966e+06 - 1.352762e+05))
  res.append((dat["patch_centroid_y"] -  6.157188e+06)/ (17.030514e+06 - 6.157188e+06))
  res.append((dat["patch_centroid_z"] -  -109.949997)/ (3164.909912 - -109.949997))
  # day
  x = np.sin(2 * np.pi * float(dat["date"][8:])/31.0)
  y = np.cos(2 * np.pi * float(dat["date"][8:])/31.0)
  res.append(x)
  res.append(y)
  #month
  x = np.sin(2 * np.pi * float(dat["date"][5:7])/12.0)
  y = np.cos(2 * np.pi * float(dat["date"][5:7])/12.0)
  res.append(x)
  res.append(y)
  #hour
  x = np.sin(2 * np.pi * float(dat["time"][:2])/24.0)
  y = np.cos(2 * np.pi * float(dat["time"][:2])/24.0)
  res.append(x)
  res.append(y)
  #camera
  cam = [1 if dat["camera"] == cam else 0 for cam in CAMERAS]
  res = res + cam
  return np.array(res).astype(np.float32)


"""## Define a class for the image segmentation dataset"""

train_img_paths = []
for dep in os.listdir("data/train"):
    for img in os.listdir("/".join(["data/train", "images"])):
        img_path = "/".join(["data/train", "images", img])
        msk_path = img_path.replace("/images/IMG_", "/masks/MSK_")
        train_img_paths.append((img_path, msk_path))

val_img_paths = []
for dep in os.listdir("data/val"):
    for img in os.listdir("/".join(["data/val", "images"])):
        img_path = "/".join(["data/val", "images", img])
        msk_path = img_path.replace("/images/IMG_", "/masks/MSK_")
        val_img_paths.append((img_path, msk_path))

class Datagen(tf.keras.utils.Sequence):
    def __init__(self, path_list, batch_size, random_state, val_rate, train, return_x_only = False, transforms=None):
        self.batch_size = batch_size
        self.random_state = random_state
        self.ids = np.array(path_list)
        self.train = train
        self.transforms = transforms
        self.rng = np.random.RandomState(random_state)
        self.rng.shuffle(self.ids)
        if train:
            self.ids = self.ids[:round((1 - val_rate) * len(self.ids))]
        else :
            self.ids = self.ids[round((1 - val_rate) * len(self.ids)):]
        self.current_index = 0
        self.num_batch = 0
        self.return_x_only = return_x_only
 
    def __augment(self, x, y):
        return x, y

    def __len__(self):
        ''' return total number of batches '''
        return math.floor(len(self.ids)/self.batch_size)

    def on_epoch_end(self):
        self.current_index = 0
        self.num_batch = 0
        if self.train : self.rng.shuffle(self.ids)
        ''' shuffle data after every epoch '''
        # fix on epoch end it's not working, adding shuffle in len for alternative
        pass

    def __getitem__(self, idx):
        
        if self.num_batch == self.__len__() - 1 or self.current_index > len(self.ids) - self.batch_size:
            self.current_index = 0
            self.num_batch = 0
            if self.train : self.rng.shuffle(self.ids)

        # list of current batch indexes
        batch_ids = self.ids[self.current_index:(self.current_index + self.batch_size)]
        x = []
        y = []
        for img_path, msk_path in batch_ids:
            x.append(read_image(img_path, mask=False))
            y.append(read_image(msk_path, mask=True))
 
        if len(x) == 0:
            print(self.current_index)
            for img_path, msk_path in self.ids[0:(0 + self.batch_size)]:
                x.append(read_image(img_path, mask=False))
                y.append(read_image(msk_path, mask=True))
        x = np.concatenate([np.expand_dims(img, axis=0) for img in x], axis=0)
        y = np.concatenate([np.expand_dims(msk, axis=0) for msk in y], axis=0)

        self.current_index += self.batch_size
        self.num_batch += 1
        
        # augmentation
        # https://github.com/albumentations-team/albumentations/issues/816
        if self.transforms is not None:
            for i in range (0,self.batch_size):
                # print (i, x[i].shape, y[i].shape)
                sample = {"image" : x[i].swapaxes(0, 2).swapaxes(0, 1), "mask": y[i]}
                transformed_sample = self.transforms(**sample)

                x[i] = transformed_sample["image"].swapaxes(0, 2).swapaxes(1, 2)
                y[i] = transformed_sample["mask"]
                # print (i, x[i].shape, y[i].shape)
        if self.return_x_only:
            return tf.convert_to_tensor(x)
        else:
            return tf.convert_to_tensor(x), tf.convert_to_tensor(y)


"""# Fine-tune a SegFormer model"""

model2 = TFSegformerForSemanticSegmentation.from_pretrained(
    "nvidia/mit-b0", # seletecd size of pretrained segformer model b0,b1,b2,b3,b4,b5
    num_labels=13
)

new_config = model2.config
# print(new_config)

new_config.num_channels = 5

model = TFSegformerForSemanticSegmentation(new_config)
model.build(input_shape=(1,5,512,512))

wts = model.get_weights()
wts2 = model2.get_weights()

for i in range(len(wts)):
    if wts[i].shape != wts2[i].shape:
        print(i, wts[i].shape,wts2[i].shape)

wts2[0] = np.concatenate([wts2[0], wts2[0][:,:,:2,:]], axis = 2)
model.set_weights(wts2)
del model2

tf.__version__


"""## Set up the Trainer"""

IMAGE_SIZE = 512 #@param {type:"number"}
BATCH_SIZE = 8 #@param {type:"number"}
NUM_CLASSES = 13 #@param {type:"number"}
DATA_DIR = "data/train"
metadata = "data/flair-one_metadata.json" 
LR = 0.00006 #@param {type:"number"}
random_state = 42 #@param {type:"number"}
checkpoint = "models/segformer_b0_5c/segformer" #@param {type:"string"}

#Define parameters for our model.
optim = keras.optimizers.Adam(learning_rate=LR)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor = 0.5, patience = 1, min_lr=0.00001, cooldown = 4, verbose = True)
stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose = True)
model_checkpoint_callback  = tf.keras.callbacks.ModelCheckpoint(checkpoint, monitor='val_loss',
                                                                verbose=1,
                                                                save_best_only=True,
                                                                save_weights_only=True,
                                                                mode='min',
                                                                save_freq='epoch')

# Loss
def dice_coef(y_true, y_pred, smooth):   
    y_true_f = K.flatten(tf.one_hot(y_true, depth = 13)[:,:,:,:-1])
    y_pred_f = K.flatten(K.softmax(y_pred, axis = -1)[:,:,:,:-1])
    intersection = K.sum(y_true_f * y_pred_f)
    dice = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return dice

def dice_coef_loss(y_true, y_pred, smooth = 100):
    return 1 - dice_coef(y_true, y_pred, smooth)

def mean_dice_coef(y_true, y_pred, smooth):
    y_true_f = tf.one_hot(y_true, depth = 13)[:,:,:,:-1]
    y_pred_f = K.softmax(y_pred, axis = -1)[:,:,:,:-1]
    intersection = K.sum(y_true_f * y_pred_f, axis = (0,1,2))
    dice = (2. * intersection + smooth) / (K.sum(y_true_f, axis = (0,1,2)) + K.sum(y_pred_f, axis = (0,1,2)) + smooth)
    return K.mean(dice[K.sum(y_true_f, axis = (0,1,2)) > 0])

def mean_dice_coef_loss(y_true, y_pred, smooth = 100):
    return 1 - mean_dice_coef(y_true, y_pred, smooth)

# Metric
class MyMeanIOU(tf.keras.metrics.MeanIoU):
    def update_state(self, y_true, y_pred, sample_weight=np.array([1,1,1,1,1,1,1,1,1,1,1,1,0])):
        y_pred = tf.image.resize(tf.transpose(y_pred, perm = (0,2,3,1)), size=(512,512), method="bilinear")
        #return super().update_state(tf.argmax(y_true, axis=-1), tf.argmax(y_pred, axis=-1), sample_weight)
        #y_true = tf.argmax(y_true, axis = -1)
        return super().update_state(y_true, tf.argmax(y_pred, axis=-1), tf.gather(np.array([1,1,1,1,1,1,1,1,1,1,1,1,0]), tf.cast(y_true, tf.int32)))

metrics = [MyMeanIOU(num_classes = 13)]


def my_loss(weights):
    def loss(labels, logits):
        logits = tf.image.resize(tf.transpose(logits, perm = (0,2,3,1)), size=(512,512), method="bilinear")
        
        labels = tf.cast(labels, tf.int32)
        return tf.compat.v1.losses.sparse_softmax_cross_entropy(labels, logits, tf.gather(weights, labels)) #tf.gather(weights, labels)
    return loss

def weighted_categorical_crossentropy(weights):
    # weights = [0.9,0.05,0.04,0.01]
    def wcce(y_true, y_pred):
        Kweights = K.constant(weights)
        #y_pred = K.constant(y_pred)
        y_true = K.cast(y_true, y_pred.dtype)
        return K.categorical_crossentropy(y_true, y_pred) * K.sum(y_true * Kweights, axis=-1)
    return wcce

# Compile model
model.compile(optimizer = optim, loss = my_loss(np.array([1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0]).astype(np.float32)), metrics=[metrics])

model.summary()

val_path = [(img, msk) for img, msk in val_img_paths]
train_path = [(img, msk) for img, msk in train_img_paths]

train_gen = Datagen(train_path, batch_size = BATCH_SIZE, random_state = random_state, val_rate=0, train=True, transforms=train_transform) 
val_gen = Datagen(val_path, batch_size = BATCH_SIZE, random_state = random_state, val_rate=1, train=False,transforms=test_transform)

# if pretrained models...
# model.load_weights(checkpoint)

"""## Training"""

history = model.fit(
    train_gen,
    epochs = 5,
    verbose=1,
    validation_data=val_gen,
    callbacks = [reduce_lr, stopping, model_checkpoint_callback], 
    use_multiprocessing=False
)

model.load_weights(checkpoint)