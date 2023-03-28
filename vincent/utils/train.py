import os
import numpy as np
import math
import tensorflow as tf
from osgeo import gdal

DATA_DIR = "C:/Users/vincent/Documents/flair/train"
SEGFORMER_IMAGENET_PATH = "C:/Users/vincent/Downloads/"
CHECKPOINT_DIR = "C:/Users/vincent/Documents/flair/"

#create checkpoint folder if doesnt exist
if not os.path.exists(CHECKPOINT_DIR):
    os.makedirs(CHECKPOINT_DIR)


def read_image(image_path, mask=False):
    
    im = gdal.Open(image_path)
    if mask:
        image = im.ReadAsArray().transpose()
        image = np.where(np.isin(image, [19,13,14,15,16,17,18]), 13, image) - 1

    else:
        image = im.ReadAsArray().transpose()

    im = None

    return image

# Weighted cross entropy loss
def my_loss(weights):
    def loss(labels, logits):
        labels = tf.cast(labels, tf.int32)
        return tf.compat.v1.losses.sparse_softmax_cross_entropy(labels, logits, tf.gather(weights, labels)) #tf.gather(weights, labels)
    return loss

# Weighted cross entropy loss for segformer (include resizing logits)
def my_loss_segformer(weights):
    def loss(labels, logits):
        logits = tf.image.resize(tf.transpose(logits, perm = (0,2,3,1)), size=(512,512), method="bilinear")
        
        labels = tf.cast(labels, tf.int32)
        return tf.compat.v1.losses.sparse_softmax_cross_entropy(labels, logits, tf.gather(weights, labels)) 
    return loss

# Weighted mIOU
class MyMeanIOU(tf.keras.metrics.MeanIoU):
    def update_state(self, y_true, y_pred, sample_weight=np.array([1,1,1,1,1,1,1,1,1,1,1,1,0])):

        return super().update_state(y_true, tf.argmax(y_pred, axis=-1), tf.gather(np.array([1,1,1,1,1,1,1,1,1,1,1,1,0]), tf.cast(y_true, tf.int32)))
        
# Weighted mIOU for segformer
class MyMeanIOU_segformer(tf.keras.metrics.MeanIoU):
    def update_state(self, y_true, y_pred, sample_weight=np.array([1,1,1,1,1,1,1,1,1,1,1,1,0])):
        y_pred = tf.image.resize(tf.transpose(y_pred, perm = (0,2,3,1)), size=(512,512), method="bilinear")

        return super().update_state(y_true, tf.argmax(y_pred, axis=-1), tf.gather(np.array([1,1,1,1,1,1,1,1,1,1,1,1,0]), tf.cast(y_true, tf.int32)))
        
# Data generator used for training
class Datagen(tf.keras.utils.Sequence):
    def __init__(self, path_list, batch_size, random_state, val_rate, 
                 train, return_x_only = False, augment = None, 
                 normalize = True,
                 standardize = True, channel_order = [0,1,2,3,4],
                 pytorch_style = False):

        self.batch_size = batch_size
        self.random_state = random_state

        self.ids = np.array(path_list)
        self.train= train

        self.rng = np.random.RandomState(random_state)
        self.rng.shuffle(self.ids)
        if train:
            self.ids = self.ids[:round((1 - val_rate) * len(self.ids))]
        else :
            self.ids = self.ids[round((1 - val_rate) * len(self.ids)):]
        self.current_index = 0
        self.num_batch = 0
        self.return_x_only = return_x_only
        self.augment = augment
        self.normalize = normalize
        self.standardize = standardize
        self.channel_order = channel_order
        self.pytorch_style = pytorch_style

    def __augment(self, x, y):
        aug = self.augment(image = x, mask = y)

        return aug["image"], aug["mask"]

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
        x = x[:,:,:,self.channel_order]
        y = np.concatenate([np.expand_dims(msk, axis=0) for msk in y], axis=0)
        
        if self.augment is not None:
            for i in range(x.shape[0]):
                new_x, new_y = self.__augment(x[i,:,:,:].astype(np.uint8), y[i,:,:])
                x[i,:,:,:] = new_x
                y[i,:,:] = new_y

        x = x.astype(np.float32)

        if self.normalize:
            x = x/255.0

        if self.standardize:
            for channel,avg,std in zip(
                range(len(self.channel_order)),
                [0.44050665, 0.45704361, 0.42254708, 0.40987858, 0.06875153], 
                [0.20264351, 0.1782405 , 0.17575739, 0.15510736, 0.11867123]):

                x[:,:,:,channel] = ((x[:,:,:,channel]) - avg)/std

        if self.pytorch_style:
            x = x.transpose((0,3,1,2))

        self.current_index += self.batch_size
        self.num_batch += 1


        if self.return_x_only:
            return tf.convert_to_tensor(x)
        else:
            return tf.convert_to_tensor(x), tf.convert_to_tensor(y)