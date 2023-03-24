import numpy as np
import tensorflow as tf
from osgeo import gdal

def read_image(image_path, mask=False, normalize = True, standardize = True, channel_order = [0,1,2,3,4]):
    
    im = gdal.Open(image_path)
    if mask:
        image = im.ReadAsArray().transpose()
        image = np.where(np.isin(image, [19,13,14,15,16,17,18]), 13, image) - 1

    else:
        image = im.ReadAsArray().transpose().astype(np.float32)[:,:,channel_order]      # rgb
        if normalize:
            image /= 255.0
        if standardize:
            for channel, avg, std in zip(
                range(image.shape[2]),
                [0.44050665, 0.45704361, 0.42254708, 0.40987858, 0.06875153],
                [0.20264351, 0.1782405 , 0.17575739, 0.15510736, 0.11867123]
            ):
                image[:,:,channel] = (image[:,:,channel] - avg)/std

    im = None

    return image



class Datagen(tf.keras.utils.Sequence):
    def __init__(self, path_list, batch_size, random_state, val_rate, train, return_x_only = False, resize_label = True, resize_x = True, augment = None):

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
        self.resize_label = resize_label
        self.resize_x = resize_x
        self.augment = augment

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
            x.append(read_image(img_path, mask=False, resize=self.resize_x))
            y.append(read_image(msk_path, mask=True, resize = self.resize_label))


        if len(x) == 0:
            print(self.current_index)

            for img_path, msk_path in self.ids[0:(0 + self.batch_size)]:
                x.append(read_image(img_path, mask=False, resize=self.resize_x))
                y.append(read_image(msk_path, mask=True, resize = self.resize_label))


        x = np.concatenate([np.expand_dims(img, axis=0) for img in x], axis=0)#[:,:3,:,:] # rgb
        y = np.concatenate([np.expand_dims(msk, axis=0) for msk in y], axis=0)
        
        if self.augment is not None:
            for i in range(x.shape[0]):
                new_x, new_y = self.__augment(x[i,:,:,:].astype(np.uint8), y[i,:,:])
                x[i,:,:,:] = new_x
                y[i,:,:] = new_y

       
        x = x.astype(np.float32)       

        self.current_index += self.batch_size
        self.num_batch += 1
        
        # augmentation
        #x, y = self.__data_augmentation(x, y)


        if self.return_x_only:
            return tf.convert_to_tensor(x)
        else:
            return tf.convert_to_tensor(x), tf.convert_to_tensor(y)