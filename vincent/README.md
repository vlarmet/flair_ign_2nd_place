# Training  

Each `python` file in `models` folder can be executed to train and save corresponding model.  
All models use imagenet pre trained weights as initialization.  

Models implemented are :  

- Unet++ with convnext Tiny encoder - 5 channels (`models/convnext_5c.py`)
- Unet with efficientnetv2 small encoder - 5 channels (`models/efficientnetv2s_5c.py`)
- Segformer b0 - 5 channels (`models/segformer_5c_b0.py`)
- Segformer b1 - 5 channels (`models/segformer_5c_b1.py`)
- Segformer b0 - RGB (`models/segformer_rgb_b0.py`)
- Segformer b1 - RGB (`models/segformer_rgb_b1.py`)
- Segformer b2 - RGB (`models/segformer_rgb_b2.py`)
- Segformer b3 - RGB (`models/segformer_rgb_b3.py`)
- Segformer b5 - RGB (`models/segformer_rgb_b4.py`)

To train, for example segformer b0 RGB model, open terminal and execute :  
`python models/segformer_rgb_b0.py`

Utility classes, functions and global parameters like folder paths are located in `utils/train.py`.  
Model-specific hyperparameters, like batch size or callbacks must be modified in each model files.  

## Training strategy

Only 1000 (randomly selected) images are used for validation.  

- Loss : weighted cross-entropy loss with weight set to 0 for class 13
- Starting Learning Rate = 0.0001, divide by 5 if validation loss does't decrease for 3 epochs.
- Stop training if validation loss does't decrease for 8 epochs.  
- Batch size varying from 2 to 8, depending model size and memory constraints.
- For 5-channels models, imagenet pre-trained weights of first convolution block are duplicated for 4th and 5th channels
- Augmentation used for segformers RGB, except segformer b4. (geometric, saturation, brightness, contrast)


# Inference
## Inference strategy

Test-time augmentation is used for Unets : each image is rotated by 0, 90, 180, 270° and probabilities are averaged.  

## Ensembling

Simple ensemble by averaging probability tensors of 9 models -> 64.1 mIOU.  

Fill data path, choose used models and execute `python  inference/ensemble.py`.   
*Note that only one model can be used so it will work as a one-model inference.*  

## Ensembling using reconstructed zones

By using georeferencing and projection information of each image, we can easily build VRTs (virtual rasters) of the test set. It is composed of 193 large zones. We predict each zone multiple times by shifting 128 pixels in x and y 4 times each. In that way, we obtain `4*4=16` predictions for each pixel. The final prediction of each pixel is voted by majority. Finally, we split predicted zones into original 512*512 patches. -> 64.84 mIOU.  

Fill data path, choose used models, pixel shift and execute `python  inference/ensemble_mosaic.py`.  

In the same way, `utils/inference.py` contains utility classes for inference and checkpoints path.  

# Weights

Imagenet pre-trained weights for segformers must be downloaded from here : 
- https://huggingface.co/nvidia/mit-b0
- https://huggingface.co/nvidia/mit-b1
- https://huggingface.co/nvidia/mit-b2
- https://huggingface.co/nvidia/mit-b3
- https://huggingface.co/nvidia/mit-b4

Our weights are available here :  
- [Unet++ with convnext Tiny encoder - 5 channels](https://drive.google.com/file/d/1ktEmo_s0O7_dqyV4jXgfElOp3ggJL_vf/view?usp=share_link)
- [Unet with efficientnetv2 small encoder - 5 channels](https://drive.google.com/drive/folders/1pB7lTfLvh1XPGGNmpnDPMefMiNgi_Gs5?usp=share_link)
- [Segformer b0 - 5 channels](https://drive.google.com/drive/folders/19mnr7NgOp3Skq86hcHbhwsfKVjtlvswR?usp=share_link)
- [Segformer b1 - 5 channels](https://drive.google.com/drive/folders/1faPQjGJRH70gi-D5RQK_ybPrc6vJFKSG?usp=share_link)
- [Segformer b0 - RGB](https://drive.google.com/drive/folders/1v9j82WJ8PoGGwqsvrD_EkQKP93b6Yh3T?usp=share_link)
- [Segformer b1 - RGB](https://drive.google.com/drive/folders/1nmoqqvlCy-cw_V8xCW6HQM1nMJW_HGs-?usp=share_link)
- [Segformer b2 - RGB](https://drive.google.com/drive/folders/1fTjtAPw0Fdunmlv5tjqYAHURuaRycRNP?usp=share_link)
- [Segformer b3 - RGB](https://drive.google.com/drive/folders/1ukTp69umhqKPEL-wod0DCdU4TYq9j0UZ?usp=share_link)
- [Segformer b4 - RGB](https://drive.google.com/drive/folders/1OlWGCgllfJWac0Kee_3c0YgUQUP498k8?usp=share_link)

# Used libraries
- numpy
- tensorflow
- gdal
- albumentation
- transformers
- tfdet (https://github.com/Burf/TFDetection)

# Specs

Training and inference have been done on a machine with the following specs :  
- OS : Windows 10
- Memory : 32Gb
- CPU : i5 11400
- GPU : NVIDIA RTX 3060 (12Gb)
