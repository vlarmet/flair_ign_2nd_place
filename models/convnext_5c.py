import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import backend as K
from sklearn.model_selection import train_test_split
from osgeo import gdal
import importlib

# Import functions defined in utils/train.py
file_path = os.path.realpath(__file__)
file_root = "/".join(file_path.replace("\\", "/").split("/")[:-2])
sys.path.append(file_root + "/utils")

import train as tr

# Import convnext architecture (not available for tensorflow v2.6)

from convnext import *

# Import tfdet library (https://github.com/Burf/TFDetection)

MODULE_PATH = file_root + "/utils/TFDetection-main/tfdet/__init__.py"
MODULE_NAME = "tfdet"
spec = importlib.util.spec_from_file_location(MODULE_NAME, MODULE_PATH)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module 
spec.loader.exec_module(module)

import tfdet

# Params
BATCH_SIZE = 4
CHECKPOINT_NAME = "model_512_unetpp_convnext_ft.h5"
LR = 0.0001

# Image and mask paths
img_paths = []
for dep in os.listdir(tr.DATA_DIR):
    for zone in os.listdir("/".join([tr.DATA_DIR, dep])):
        for img in os.listdir("/".join([tr.DATA_DIR, dep, zone, "img"])):
            if img.__contains__("xml"):
                continue
            img_path = "/".join([tr.DATA_DIR, dep, zone, "img", img])
            msk_path = img_path.replace("/img/IMG_", "/msk/MSK_")
            img_paths.append((img_path, msk_path))


# Model (duplicate pretrained weights for first convolution filters)

x = tf.keras.layers.Input(shape=(512,512,5))

model = ConvNeXtTiny(
    include_top=False,
    include_preprocessing=True,
    weights="imagenet",
    input_tensor=None,
    input_shape=(512,512,3),
    pooling=None,
    classifier_activation=None)

model2 = ConvNeXtTiny(
    include_top=False,
    include_preprocessing=True,
    weights=None,
    input_tensor=x,
    input_shape=(512,512,5),
    pooling=None,
    classifier_activation=None)

weights = model.get_weights()
weights2 = model2.get_weights()
conv_wt = weights[0]

conv_wt = np.concatenate([conv_wt, conv_wt[:,:,:2,:]], axis = 2)
weights[0] = conv_wt

model2.set_weights(weights)

model = model2
del model2


feature = tfdet.model.backbone.convnext_tiny(x, model = model)   
out = tfdet.model.detector.unet_2plus(feature, n_class = 13, logits_activation=None)
out = tf.keras.layers.UpSampling2D((4, 4))(out)
model = tf.keras.Model(x, out)

# Hyperparameters
## Optimizer
optim = keras.optimizers.Adam(LR)

## Reduce LR policy
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor = 0.2,
                              patience = 3, min_lr=0.0000001, verbose = True)

## Abort training policy
stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=8,
    verbose = True)

## Checkpoint
model_checkpoint_callback  = tf.keras.callbacks.ModelCheckpoint(
tr.CHECKPOINT_DIR + CHECKPOINT_NAME, monitor='val_loss', verbose=0, save_best_only=True,
save_weights_only=True, mode='min', save_freq='epoch')

## Metric 
metrics = [tr.MyMeanIOU(num_classes = 13)]

# Train/validation split
train_path, val_path = train_test_split(img_paths, test_size = 1000, random_state=42)

# Data generators
train_gen = tr.Datagen(train_path, batch_size = BATCH_SIZE, random_state = 10, val_rate=0, 
                    train=True, 
                    augment= None,
                    normalize = True,
                    standardize = False, 
                    channel_order = [2,1,0,3,4],
                    pytorch_style = False) 

val_gen = tr.Datagen(val_path, batch_size = BATCH_SIZE, random_state = 10, val_rate=1, 
                  train=False, 
                  normalize = True,
                  standardize = False, 
                  channel_order = [2,1,0,3,4],
                  pytorch_style = False)

# Compile model
model.compile(optimizer = optim, loss = tr.my_loss(np.array([1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,0.0]).astype(np.float32))
, metrics=[metrics])

# Train
history = model.fit(
    train_gen,
    epochs = 200,
    verbose=1,
    validation_data=val_gen,
    callbacks = [reduce_lr, stopping, model_checkpoint_callback],
    use_multiprocessing=False
)