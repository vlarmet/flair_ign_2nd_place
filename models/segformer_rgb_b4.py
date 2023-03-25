import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import backend as K
from sklearn.model_selection import train_test_split
from osgeo import gdal
from transformers import TFSegformerForSemanticSegmentation, SegformerConfig

# import function defined in utils/train.py 
file_path = os.path.realpath(__file__)
file_root = "/".join(file_path.replace("\\", "/").split("/")[:-2])
sys.path.append(file_root + "/utils")

from train import *

# Global parameters
BATCH_SIZE = 2
CHECKPOINT_NAME = "model_512_segnetb4_rgb_val1000"
LR = 0.0001

# Image and mask paths
img_paths = []
for dep in os.listdir(DATA_DIR):
    for zone in os.listdir("/".join([DATA_DIR, dep])):
        for img in os.listdir("/".join([DATA_DIR, dep, zone, "img"])):
            if img.__contains__("xml"):
                continue
            img_path = "/".join([DATA_DIR, dep, zone, "img", img])
            msk_path = img_path.replace("/img/IMG_", "/msk/MSK_")
            img_paths.append((img_path, msk_path))


# Model
model = TFSegformerForSemanticSegmentation.from_pretrained(
    SEGFORMER_IMAGENET_PATH + "mit_b4",
    num_labels=13
)

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
CHECKPOINT_DIR + CHECKPOINT_NAME, monitor='val_loss', verbose=0, save_best_only=True,
save_weights_only=True, mode='min', save_freq='epoch')

# Metric
metrics = [MyMeanIOU_segformer(num_classes = 13)]

# Train/validation split
train_path, val_path = train_test_split(img_paths, test_size = 1000, random_state=42)

# Data generators
train_gen = Datagen(train_path, batch_size = BATCH_SIZE, random_state = 10, val_rate=0, 
                    train=True, 
                    augment=None,
                    normalize = True,
                    standardize = True, 
                    channel_order = [0,1,2],
                    pytorch_style = True) 

val_gen = Datagen(val_path, batch_size = BATCH_SIZE, random_state = 10, val_rate=1, 
                  train=False, 
                  normalize = True,
                  standardize = True, 
                  channel_order = [0,1,2],
                  pytorch_style = True)

# Compile model
model.compile(optimizer = optim, loss = my_loss_segformer(np.array([1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,0.0]).astype(np.float32))
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