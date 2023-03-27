import sys
import os
import importlib
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import backend as K
from transformers import TFSegformerForSemanticSegmentation, SegformerConfig
# Import convnext and efficientnetv2s architectures
file_path = os.path.realpath(__file__)
file_root = "/".join(file_path.replace("\\", "/").split("/")[:-2])
sys.path.append(file_root + "/utils")

# Import effficientnet_v2 architecture (not available for tensorflow v2.6)

from effficientnet_v2 import *
from convnext import *

# Import tfdet library (https://github.com/Burf/TFDetection)

MODULE_PATH = file_root + "/utils/TFDetection-main/tfdet/__init__.py"
MODULE_NAME = "tfdet"
spec = importlib.util.spec_from_file_location(MODULE_NAME, MODULE_PATH)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module 
spec.loader.exec_module(module)

import tfdet

CHECKPOINT_DIR = "C:/Users/vincent/Documents/flair/"

class Vmodel:
    def __init__(self, model, name=None, tta=None, pytorch_style=False, 
                 normalize=True, standardize=True, channel_order=[0,1,2,3,4], softmax=True) -> None:
        self.model = model
        self.name = name
        self.tta = tta
        self.pytorch_style = pytorch_style
        self.normalize = normalize
        self.standardize = standardize
        self.channel_order = channel_order
        self.softmax = softmax

    def predict(self, image) -> np.ndarray:
        #print(self.name, image[0,:2,:2,0])
        image = image.astype(np.float32)

        if self.normalize:
            image = image/255.0

        if self.standardize :
            for channel,avg,std in zip(
                self.channel_order,
                [0.44050665, 0.45704361, 0.42254708, 0.40987858, 0.06875153], 
                [0.20264351, 0.1782405 , 0.17575739, 0.15510736, 0.11867123]):

                image[:,:,:,channel] = ((image[:,:,:,channel]) - avg)/std
        #print(self.name, image[0,:2,:2,0])
        # keep used channel in correct order
        image = image[:,:,:,self.channel_order]

        if self.pytorch_style:
            image = image.transpose((0,3,1,2))
            if self.tta is not None:
                pred = [np.expand_dims(self.tta(image[img_index,:,:,:], model = self.model), axis=0) for img_index in range(image.shape[0])]
                pred = np.concatenate(pred, axis = 0)

            else:
                pred = list(self.model.predict(image).values())[0]

            pred = pred.transpose((0,2,3,1))
            

        else:
            if self.tta is not None:
                pred = [np.expand_dims(self.tta(image[img_index,:,:,:], model = self.model), axis=0) for img_index in range(image.shape[0])]
                pred = np.concatenate(pred, axis = 0)

            else:
                pred = self.model.predict(image)

        if self.softmax:
            pred = K.softmax(pred, axis = -1)


        return pred
    
def tta(x, model):
    xs = np.concatenate([np.expand_dims(np.rot90(x, k=i, axes = (0,1)), axis = 0) for i in range(0,4)], axis = 0)
    pred = K.softmax(model.predict(xs))
    preds = np.mean(np.array([np.rot90(pred[i,:,:,:], k = -i, axes = (0,1)) for i in range(0,4)]), axis = 0)
    return preds

def tta_segformer(x, model):
    xs = np.concatenate([np.expand_dims(np.rot90(x, k=i, axes = (1,2)), axis = 0) for i in range(0,4)], axis = 0)
    pred = K.softmax(list(model.predict(xs).values())[0], axis = 1)
    preds = np.mean(np.array([np.rot90(pred[i,:,:,:], k = -i, axes = (1,2)) for i in range(0,4)]), axis = 0)
    return preds

def load_models(unet_efficientnetv2s = True,
                unetpp_convnext = True,
                segformerb0_5c = True,
                segformerb1_5c=True,
                segformerb0_rgb = True,
                segformerb1_rgb = True,
                segformerb2_rgb = True,
                segformerb3_rgb = True,
                segformerb4_rgb = True) -> list:
    '''
    Return list containing trained models and associated hyperparameters
    Trained models are used by instantiating "Vmodel" objects.
    Initialization parameters:

    model : TF model
    name : model name (str)
    tta : Test-Time Augmentation function (default None)
    pytorch_style : channel first (default False) 
    normalize : divide by 255 (default True)
    standardize : imagenet stats standardization after normalization (default True)
    channel_order : channel order by index (default [0,1,2,3,4])
    softmax : use softmax activation (default True)
    '''
    models = []
    if unet_efficientnetv2s:
        # Unet efficientnet2 small 5 channels
        x = tf.keras.layers.Input(shape=(512,512,5))
        model = EfficientNetV2S(
            include_top=False,
            weights=None,
            input_tensor=x,
            input_shape=(512,512,5),
            pooling=None,
            classes=1000,
            classifier_activation=None,
            include_preprocessing=True,
        )

        feature = tfdet.model.backbone.effnet_v2_s(x, model = model)   
        out = tfdet.model.detector.unet(feature, n_class = 13, logits_activation=None)
        out = tf.keras.layers.UpSampling2D((4, 4))(out)
        model = tf.keras.Model(x, out)
        model.load_weights(CHECKPOINT_DIR + "model_512_unet_effnetv2_s.h5")

        mod = Vmodel(model = model, name = "effnetv2S", 
                    tta = tta, 
                    normalize = False, standardize = False, 
                    channel_order = [2,1,0,3,4],
                    softmax=False)

        models.append(mod)
        
    if unetpp_convnext:
        # Unet++ convnext tiny 5 channels
        x = tf.keras.layers.Input(shape=(512,512,5))
        model = ConvNeXtTiny(
            include_top=False,
            include_preprocessing=True,
            weights=None,
            input_tensor=x,
            input_shape=(512,512,5),
            pooling=None,
            classifier_activation=None)
        feature = tfdet.model.backbone.convnext_tiny(x, model = model)   
        out = tfdet.model.detector.unet_2plus(feature, n_class = 13, logits_activation=None)
        out = tf.keras.layers.UpSampling2D((4, 4))(out)
        model = tf.keras.Model(x, out)
        model.load_weights(CHECKPOINT_DIR + "model_512_unetpp_convnext_ft.h5")

        mod = Vmodel(model = model, name = "convnext", 
                    tta = tta, 
                    standardize=False,
                    channel_order = [2,1,0,3,4],
                    softmax=False)
        models.append(mod)
        
    if segformerb0_5c:
        # Segformer b0 5 channels
        config = SegformerConfig(
            num_channels=5, 
            num_labels=13
            )
            
        model = TFSegformerForSemanticSegmentation(config)
        model.build(input_shape=(1,5,512,512))
        model.load_weights(CHECKPOINT_DIR + "model_512_segnetb0_val1000")

        mod = Vmodel(model = model, name = "segb0_5c", 
                            pytorch_style = True)
        models.append(mod)
        
    if segformerb1_5c:
        # Segformer b1 5 channels
        config = SegformerConfig(
            num_channels=5, 
            num_labels=13,
            depths=[2, 2, 2, 2],
            hidden_sizes=[64, 128, 320, 512],
            decoder_hidden_size=256
            )
            
        model = TFSegformerForSemanticSegmentation(config)
        model.build(input_shape=(1,5,512,512))
        model.load_weights(CHECKPOINT_DIR + "model_512_segnetb1_val1000")

        mod = Vmodel(model = model, name = "segb1_5c", 
                    pytorch_style = True)
        models.append(mod)
        
    if segformerb0_rgb:
        config = SegformerConfig(
            num_channels=3, 
            num_labels=13
            )
            
        model = TFSegformerForSemanticSegmentation(config)
        model.build(input_shape=(1,3,512,512))
        model.load_weights(CHECKPOINT_DIR + "model_512_segnetb0_rgb_aug_val1000")

        mod = Vmodel(model = model, name = "segb0_rgb", 
            pytorch_style = True, 
            channel_order=[0,1,2])
        models.append(mod)
        
    if segformerb1_rgb:
        config = SegformerConfig(
            num_channels=3, 
            num_labels=13,
            depths=[2, 2, 2, 2],
            hidden_sizes=[64, 128, 320, 512],
            decoder_hidden_size=256
            )
            
        model = TFSegformerForSemanticSegmentation(config)
        model.build(input_shape=(1,3,512,512))
        model.load_weights(CHECKPOINT_DIR + "model_512_segnetb1_rgb_aug_val1000")

        mod = Vmodel(model = model, name = "segb1_rgb", 
            pytorch_style = True, 
            channel_order=[0,1,2])
        models.append(mod)
        
    if segformerb2_rgb:
        config = SegformerConfig(
            num_channels=3, 
            num_labels=13,
            depths=[3, 4, 6, 3],
            hidden_sizes=[64, 128, 320, 512],
            decoder_hidden_size=768
            )
            
        model = TFSegformerForSemanticSegmentation(config)
        model.build(input_shape=(1,3,512,512))
        model.load_weights(CHECKPOINT_DIR + "model_512_segnetb2_rgb_aug_val1000")

        mod = Vmodel(model = model, name = "segb2_rgb", 
            pytorch_style = True, 
            channel_order=[0,1,2])
        models.append(mod)
        
    if segformerb3_rgb:
        config = SegformerConfig(
            num_channels=3, 
            num_labels=13,
            depths=[3, 4, 18, 3],
            hidden_sizes=[64, 128, 320, 512]	,
            decoder_hidden_size=768
            )
            
        model = TFSegformerForSemanticSegmentation(config)
        model.build(input_shape=(1,3,512,512))
        model.load_weights(CHECKPOINT_DIR + "model_512_segnetb3_rgb_aug_val1000")

        mod = Vmodel(model = model, name = "segb3_rgb", 
            pytorch_style = True, 
            channel_order=[0,1,2])
        models.append(mod)
        
    if segformerb4_rgb:
        config = SegformerConfig(
            num_channels=3, 
            num_labels=13,
            depths=[3, 8, 27, 3],
            hidden_sizes=[64, 128, 320, 512]	,
            decoder_hidden_size=768
            )
            
        model = TFSegformerForSemanticSegmentation(config)
        model.build(input_shape=(1,3,512,512))
        model.load_weights(CHECKPOINT_DIR + "model_512_segnetb4_rgb_val1000")

        mod = Vmodel(model = model, name = "segb4_rgb", 
            pytorch_style = True, 
            channel_order=[0,1,2])
        models.append(mod)
        
    return models


def predict_ensemble(model_list, img, size_list = [128, 512], target_size = 512):
    img2 = copy.deepcopy(img)
    preds = [model.predict(img2) for model in model_list]
    # We average each predictions for each output size. We resize after
    res = []
    N = []
    for size in size_list:
        tmp = [pred for pred in preds if pred.shape[2] == size]
        N.append(len(tmp))
        tmp = np.mean(np.array(tmp), axis = 0)
        if size != target_size:
            tmp = tf.image.resize(tmp, size = [512,512], method = "bilinear")
        res.append(tmp)

    preds = np.sum([n * pred for n, pred in zip(N, res)], axis = 0)/np.sum(N)
    return preds

##################### Mosaique 
