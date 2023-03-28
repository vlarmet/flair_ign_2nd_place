# Train models  

Each `python` file in `models` folder can be executed to train and save corresponding model.  
All models use imagenet pre trained weights as initialization.  

Models implemented are :  

- Unet++ with convnext Tiny encoder - 5 channels (`models/convnext_5c.py`)
- Unet with efficientnetv2 small encoder - 5 channels ()
- Segformer b0 - 5 channels ()
- Segformer b1 - 5 channels ()
- Segformer b0 - RGB ()
- Segformer b1 - RGB ()
- Segformer b2 - RGB ()
- Segformer b3 - RGB ()
- Segformer b5 - RGB ()

To train segformer b0 RGB model, open terminal and execute :  
`python models/segformer_rgb_b0.py`

Utility classes, functions and global parameters like folder paths are located in `utils/train.py`.  
Model-specific hyperparameters, like batch size or callbacks are in each model files.  

# Inference

Simple ensemble by averaging probability tensors of 9 models -> 61.4 mIOU.  
Fill data path, choose used models and execute `python  inference/ensemble.py`.  

# Inference using reconstructed landscape patches

By using georeferencing and projection information of each image, we can easily build VRTs (virtual rasters) of the test set. It is composed of 193 large zones. We predict each zone multiple times by shifting 128 pixels in x and y 4 times each. In that way, we obtain 4*4=16 predictions for each pixel. The final prediction of each pixel is voted by majority. Finally, we split predicted zones into original 512*512 patches. -> 64.84 mIOU.  
Fill data path, choose used models, stride and execute `python  inference/ensemble_mosaic.py`.  

In the same way, `utils/inference.py` contains utility classes for inference and checkpoints path.  

# Additional data

Imagenet pre-trained weights for segformers must be downloaded from here : 
- https://huggingface.co/nvidia/mit-b0
- https://huggingface.co/nvidia/mit-b1
- https://huggingface.co/nvidia/mit-b2
- https://huggingface.co/nvidia/mit-b3
- https://huggingface.co/nvidia/mit-b4

Our weights are available here : 