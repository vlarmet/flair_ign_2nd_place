# FLAIR - Semantic segmentation and domain adaptation challenge proposed by the French National Institute of Geographical and Forest Information (IGN)

## Context

We present here a large dataset ( >20 billion pixels) of aerial imagery, topographic information and land cover (buildings, water, forest, agriculture...) annotations with the aim to further advance research on semantic segmentation , domain adaptation and transfer learning. Countrywide remote sensing aerial imagery is by necessity acquired at different times and dates and under different conditions. Likewise, at large scales, the characteristics of semantic classes can vary depending on location and become heterogenous. This opens up challenges for the spatial and temporal generalization of deep learning models!


# Data and baseline: 

The FLAIR-one dataset consists of 77,412 high resolution (0.2 m spatial resolution) patches with 13 semantic classes (19 original classes remapped to 13, see the associated paper in the starting kit for explanation). The dataset covers a total of approximatly 800 km², with patches that have been sampled accross the entire metropolitan French territory to be illustrating the different climate and landscapes (spatial domains). The aerial images included in the dataset were acquired during different months and years (temporal domains).

A U-Net architecture with a pre-trained ResNet34 encoder from the pytorch segmentation models library is used for the baselines. The used architecture allows integration of patch-wise metadata information and employs commonly used image data augmentation techniques. Results are presented in the technical description of the dataset.

[Go to challenge link](https://codalab.lisn.upsaclay.fr/competitions/8769). 

# Usage

In order to compute our final submission, you must produce two predictions by following instructions in README files of `vincent` and `alan` folders of this repository.  
After that, execute the following in terminal :  
`python assemble.py`   

You must fill correct paths in `assemble.py` before executing it.  

# Abstract of our solution

Please read this [document](https://github.com/vlarmet/flair_ign_2nd_place/blob/master/abstract.pdf) detailing our approach.