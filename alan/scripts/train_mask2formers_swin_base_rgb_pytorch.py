# -*- coding: utf-8 -*-

from torch.utils.data import Dataset, DataLoader
from transformers import AdamW
import torch
from torch import nn
from sklearn.metrics import accuracy_score
from tqdm.notebook import tqdm
import os
from PIL import Image
from transformers import MaskFormerImageProcessor, Mask2FormerForUniversalSegmentation
import pandas as pd
import cv2
import numpy as np
import albumentations as aug
import rasterio
from pathlib import Path
import splitfolders
import shutil


"""## Define a class for the image segmentation dataset"""

def get_data_paths (path, filter):
    for path in Path(path).rglob(filter):
        yield path.resolve().as_posix()

class ImageSegmentationDataset(Dataset):
    """Image segmentation dataset."""

    def __init__(self, root_dir, transforms=None, train=True):
        """
        Args:
            root_dir (string): Root directory of the dataset containing the images + annotations.
            feature_extractor (SegFormerFeatureExtractor): feature extractor to prepare images + segmentation maps.
            train (bool): Whether to load "training" or "validation" images + annotations.
        """
        self.root_dir = root_dir
        self.train = train
        self.transforms = transforms
        self.images = sorted(list(get_data_paths(Path(self.root_dir), 'IMG*.tif')), key=lambda x: int(x.split('_')[-1][:-4]))
        self.masks = sorted(list(get_data_paths(Path(self.root_dir), 'MSK*.tif')), key=lambda x: int(x.split('_')[-1][:-4]))
        assert len(self.images) == len(self.masks), "There must be as many images as there are segmentation maps"
      
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        
        image_file = self.images[idx]
        with rasterio.open(image_file) as src_img:
            original_image = src_img.read([1,2,3]).swapaxes(0, 2).swapaxes(0, 1)

        mask_file = self.masks[idx]
        with rasterio.open(mask_file) as src_msk:
            original_segmentation_map = src_msk.read()[0]
        original_segmentation_map = np.squeeze(original_segmentation_map)
        original_segmentation_map[original_segmentation_map > 12] = 0
        transformed = self.transforms(image=original_image, mask=original_segmentation_map)
        image, segmentation_map = transformed['image'], transformed['mask']

        # convert to C, H, W
        image = image.transpose(2,0,1)
        return image, segmentation_map, original_image, original_segmentation_map


"""## Data augmentation with albumentation"""

MEAN = np.array([0.44050665, 0.45704361, 0.42254708]) 
STD = np.array([0.20264351, 0.1782405 , 0.17575739]) 

train_transform = aug.Compose([
    aug.VerticalFlip(p=0.5),
    aug.HorizontalFlip(p=0.5),
    aug.RandomRotate90(p=0.5),
    aug.ColorJitter(),
    aug.RandomBrightnessContrast(),
    aug.Normalize(mean=MEAN, std=STD),
])

test_transform = aug.Compose([
    aug.Normalize(mean=MEAN, std=STD),
])

root_train = 'data/train'
root_val = 'data/val'

train_dataset = ImageSegmentationDataset(root_dir=root_train,  transforms=train_transform)
valid_dataset = ImageSegmentationDataset(root_dir=root_val, transforms=test_transform, train=False)

print("Number of training examples:", len(train_dataset))
print("Number of validation examples:", len(valid_dataset))

from transformers import MaskFormerImageProcessor

# Create a preprocessor
preprocessor = MaskFormerImageProcessor(ignore_index=0, reduce_labels=False, do_resize=False, do_rescale=False, do_normalize=False)

from torch.utils.data import DataLoader
def collate_fn(batch):
    inputs = list(zip(*batch))
    images = inputs[0]
    segmentation_maps = inputs[1]
    # this function pads the inputs to the same size,
    # and creates a pixel mask
    # actually padding isn't required here since we are cropping
    batch = preprocessor(
        images,
        segmentation_maps=segmentation_maps,
        return_tensors="pt",
    )
    batch["original_images"] = inputs[2]
    batch["original_segmentation_maps"] = inputs[3]
    return batch

train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
test_dataloader = DataLoader(valid_dataset, batch_size=2, shuffle=False, collate_fn=collate_fn)

batch = next(iter(train_dataloader))
for k,v in batch.items():
  if isinstance(v, torch.Tensor):
    print(k,v.shape)
  else:
    print(k,v[0].shape)


"""## Classes metadata"""

classes = ['None','building','pervious surface','impervious surface','bare soil','water','coniferous','deciduous','brushwood','vineyard','herbaceous vegetation','agricultural land','plowed land']
id2label = pd.DataFrame(classes)[0].to_dict()
label2id = {v: k for k, v in id2label.items()}
pixel_values = batch["pixel_values"][0].numpy()
pixel_values.shape


"""# Fine-tune a Mask2former model"""

from transformers import Mask2FormerForUniversalSegmentation
# Replace the head of the pre-trained model
model = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-base-IN21k-ade-semantic",
                                                          id2label=id2label,
                                                          ignore_mismatched_sizes=True)

import evaluate
metric = evaluate.load("mean_iou")

# if you have a pretrained checkpoint...
# model.load_state_dict(torch.load("models/mask2former-swin-base-ade-semantic/mask2former-swin-base-ade-semantic"))

import torch
from tqdm.auto import tqdm
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=6e-5)
step=0
running_loss = 0.0
num_samples = 0
for epoch in range(2):
  print("Epoch:", epoch)
  model.train()
  
  for idx, batch in enumerate(tqdm(train_dataloader)):
      # Reset the parameter gradients
      optimizer.zero_grad()

      # Forward pass
      outputs = model(
          pixel_values=batch["pixel_values"].to(device),
          mask_labels=[labels.to(device) for labels in batch["mask_labels"]],
          class_labels=[labels.to(device) for labels in batch["class_labels"]],
      )

      # Backward propagation
      loss = outputs.loss
      loss.backward()

      batch_size = batch["pixel_values"].size(0)
      running_loss += loss.item()
      num_samples += batch_size

      if idx % 100 == 0:
        print("Loss:", running_loss/num_samples)

      # Optimization
      optimizer.step()
      step += 1
      if step % 5000 == 0:
          model.eval()
          for idx, batch in enumerate(tqdm(test_dataloader)):
            if idx > 310:
              break

            pixel_values = batch["pixel_values"]
            
            # Forward pass
            with torch.no_grad():
                outputs = model(pixel_values=pixel_values.to(device))

            # get original images
            original_images = batch["original_images"]
            target_sizes = [(image.shape[0], image.shape[1]) for image in original_images]
            # predict segmentation maps
            predicted_segmentation_maps = preprocessor.post_process_semantic_segmentation(outputs,
                                                                                          target_sizes=target_sizes)

            # get ground truth segmentation maps
            ground_truth_segmentation_maps = batch["original_segmentation_maps"]

            metric.add_batch(references=ground_truth_segmentation_maps, predictions=predicted_segmentation_maps)
          
          # NOTE this metric outputs a dict that also includes the mIoU per category as keys
          # so if you're interested, feel free to print them as well
          print("Mean IoU:", metric.compute(num_labels = len(id2label), ignore_index = 0)['mean_iou'])
          torch.save(model.state_dict(), 'models/mask2former-swin-base-ade-semantic/mask2former-swin-base-ade-semantic')