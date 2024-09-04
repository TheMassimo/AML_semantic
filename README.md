# Real-time Domain Adaptation in Semantic Segmentation

Semantic segmentation in real-time is a very important task for image analysis. Over time, various techniques have been discovered and experimented with, leading to the need to apply segmentation from a source domain to a target domain that differ from each other—typically, the former being synthetic and the latter real-world data. Here, we will explore the possibility of performing semantic segmentation with domain adaptation, starting from a dataset of images from GTA5 and transitioning to a real-world image dataset like Cityscapes. Additionally, tools such as FDA will be introduced to enhance performance.
This repository houses the code for our project in the "Advanced Machine Learning" course at Politecnico di Torino. 

#### AUTHORS
- s318098 - Massimo Porcheddu
- s317715 - Paolo Muccilli
- s318109 - Miriam Ivaldi


## Project structure
- `datasets/`: classes to handle the datasets used in the project.
  - [CityscapesDataset](datasets/CityScapesDataset.py)
  - [GTA5Dataset](datasets/GTA5Dataset.py)
  - [GTA5 mapping classes](datasets/gta5_mapping.py)

- `model/`: implementation of the models used in the project and their components.
  - [BiSeNet](model/model_stages.py)
  - [BiSeNet Discriminator](model/model_stages.py)
  - [STDCNet813 Backbone](model/stdcnet.py)

- `pretrained/`: folder to pretrained model
  - [Pretrained model](pretrained/STDCNet813M_73.91)

- `utils.py`: General utility functions
- `train.py`: Code for implementing the various training strategies used in the project..
- `Project Overview.pdf`: the compiled version of the original project requirements file.

## Used dataset
The required dataset are not included, but they can be download at the following link
- Cityscapes and gta5 [datasets_drive_link](https://drive.google.com/drive/u/0/folders/1iE8wJT7tuDOVjEBZ7A3tOPZmNdroqG1m).
Also the pre-trained model is not included, but it can be download at the following link
- STDCNet813 [stdcnet_drive_link](https://drive.google.com/drive/folders/1wROFwRt8qWHD4jSo8Zu1gp1d6oYJ3ns1).

## Results of various steps
### TESTING REAL-TIME SEMANTIC SEGMENTATION**
####  A) Defining the upper bound for the domain adaptation phase
  
    