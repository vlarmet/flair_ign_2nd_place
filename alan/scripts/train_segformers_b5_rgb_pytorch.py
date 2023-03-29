# -*- coding: utf-8 -*-
from torch.utils.data import Dataset, DataLoader
from transformers import AdamW
import torch
from torch import nn
from sklearn.metrics import accuracy_score
from tqdm.notebook import tqdm
import os
from PIL import Image
from transformers import SegformerForSemanticSegmentation, SegformerFeatureExtractor
import pandas as pd
import cv2
import numpy as np
import albumentations as aug
import random
import rasterio
from pathlib import Path
import splitfolders
import shutil


"""## Define a class for the image segmentation dataset"""

def get_data_paths (path, filter):
    for path in Path(path).rglob(filter):
        yield path.resolve().as_posix()

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

class ImageSegmentationDataset(Dataset):
    """Image segmentation dataset."""

    def __init__(self, root_dir, feature_extractor, transforms=None):
        self.root_dir = root_dir
        self.feature_extractor = feature_extractor
        self.transforms = transforms
        self.images = sorted(list(get_data_paths(Path(self.root_dir), 'IMG*.tif')), key=lambda x: int(x.split('_')[-1][:-4]))
        self.masks = sorted(list(get_data_paths(Path(self.root_dir), 'MSK*.tif')), key=lambda x: int(x.split('_')[-1][:-4]))
        assert len(self.images) == len(self.masks), "There must be as many images as there are segmentation maps"

    def read_img(self, raster_file: str) -> np.ndarray:
        with rasterio.open(raster_file) as src_img:
            rgb = src_img.read([1,2,3]).swapaxes(0, 2).swapaxes(0, 1)
            return rgb

    def read_msk(self, raster_file: str) -> np.ndarray:
        with rasterio.open(raster_file) as src_msk:
            array = src_msk.read()[0]
            array = np.squeeze(array)
            return array

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_file = self.images[idx]
        image = self.read_img(raster_file=image_file)
        mask_file = self.masks[idx]
        segmentation_map = self.read_msk(raster_file=mask_file)
        segmentation_map[segmentation_map > 12] = 0
        if self.transforms is not None:
            augmented = self.transforms(image=image, mask=segmentation_map)
            encoded_inputs = self.feature_extractor(augmented['image'], augmented['mask'], return_tensors="pt")
        else:
            encoded_inputs = self.feature_extractor(image, segmentation_map, return_tensors="pt")
        for k,v in encoded_inputs.items():
            encoded_inputs[k].squeeze_() # remove batch dimension
        return encoded_inputs


"""## Data augmentation with albumentation"""

MEAN = np.array([0.44050665, 0.45704361, 0.42254708])
STD = np.array([0.20264351, 0.1782405 , 0.17575739])

train_transform = aug.Compose([
    aug.VerticalFlip(p=0.5),
    aug.HorizontalFlip(p=0.5),
    aug.RandomRotate90(p=0.5),
    aug.Normalize(mean=MEAN, std=STD),
    aug.augmentations.transforms.ColorJitter(p=0.5),
])

test_transform = aug.Compose([
    aug.Normalize(mean=MEAN, std=STD),
])

feature_extractor = SegformerFeatureExtractor(ignore_index=0, reduce_labels=False, do_resize=False, do_rescale=False, do_normalize=False)

root_train = 'data/train'
root_val = 'data/val'

train_dataset = ImageSegmentationDataset(root_dir=root_train, feature_extractor=feature_extractor, transforms=train_transform)
valid_dataset = ImageSegmentationDataset(root_dir=root_val, feature_extractor=feature_extractor, transforms=test_transform)

print("Number of training examples:", len(train_dataset))
print("Number of validation examples:", len(valid_dataset))

"""## Classes metadata"""

classes = ['None','building','pervious surface','impervious surface','bare soil','water','coniferous','deciduous','brushwood','vineyard','herbaceous vegetation','agricultural land','plowed land']
id2label = pd.DataFrame(classes)[0].to_dict()
label2id = {v: k for k, v in id2label.items()}

num_labels = len(id2label)
num_labels

"""# Fine-tune a SegFormer model"""

from transformers import SegformerForSemanticSegmentation
pretrained_model_name =  "nvidia/mit-b5" #@param {type:"string"} # seletecd size of pretrained segformer model b0,b1,b2,b3,b4,b5
model = SegformerForSemanticSegmentation.from_pretrained(
    pretrained_model_name,
    id2label=id2label,
    label2id=label2id,
    reshape_last_stage=True,
    ignore_mismatched_sizes=True
)

"""## Set up the Trainer"""

from transformers import TrainingArguments
epochs = 30 #@param {type:"number"}
lr = 6e-5 #@param {type:"number"}
batch_size = 8 #@param {type:"number"}
outputdir = "models/segformer_b0_rgb" #@param {type:"string"}

training_args = TrainingArguments(
    outputdir,
    learning_rate=lr,
    num_train_epochs=epochs,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    evaluation_strategy="steps",
    save_strategy="steps",
    save_steps=2000,
    eval_steps=500,
    # warmup_steps=500,
    # weight_decay=0.05,
    remove_unused_columns=False,
    report_to='tensorboard',
    load_best_model_at_end=True,
)

"""## Metrics for eval"""

import torch
from torch import nn
import evaluate
import multiprocessing

metric = evaluate.load("mean_iou")

def compute_metrics(eval_pred):
  with torch.no_grad():
    logits, labels = eval_pred
    logits_tensor = torch.from_numpy(logits)
    # scale the logits to the size of the label
    logits_tensor = nn.functional.interpolate(
        logits_tensor,
        size=labels.shape[-2:],
        mode="bilinear",
        align_corners=False,
    ).argmax(dim=1)

    pred_labels = logits_tensor.detach().cpu().numpy()
    metrics = metric._compute(
            predictions=pred_labels,
            references=labels,
            num_labels=len(id2label),
            ignore_index=0,
            # reduce_labels=feature_extractor.reduce_labels,
        )
    
    #add per category metrics as individual key-value pairs
    per_category_accuracy = metrics.pop("per_category_accuracy").tolist()
    per_category_iou = metrics.pop("per_category_iou").tolist()

    metrics.update({f"accuracy_{id2label[i]}": v for i, v in enumerate(per_category_accuracy)})
    metrics.update({f"iou_{id2label[i]}": v for i, v in enumerate(per_category_iou)})

    return metrics

"""## Training"""

from transformers import Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()