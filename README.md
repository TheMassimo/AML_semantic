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
  - [GTA5Dataset](datasets/Gta5Dataset.py)
  - [GTA5 mapping classes](datasets/gta5_mapping.json)

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
#### The required dataset are not included, but they can be download at the following link:
- Cityscapes and gta5 [datasets_drive_link](https://drive.google.com/drive/u/0/folders/1iE8wJT7tuDOVjEBZ7A3tOPZmNdroqG1m).
#### Also the pre-trained model is not included, but it can be download at the following link
- STDCNet813 [stdcnet_drive_link](https://drive.google.com/drive/folders/1wROFwRt8qWHD4jSo8Zu1gp1d6oYJ3ns1).

## Results of various steps
### TESTING REAL-TIME SEMANTIC SEGMENTATION**
####  A) Defining the upper bound for the domain adaptation phase
  
##### Semantic segmentation Cityscapes (real dataset).

    ```bash
    train.py --task semantic_segmentation --dataset cityscapes --num_classes 19  --root_dir Cityscapes_ds/Cityspaces --batch_size 40 --num_workers 4 --learning_rate 0.01 --num_epochs 50 --pretrain_path pretrained/STDCNet813M_73.91 --save_model_path final_model\cityscapes --optimizer sgd
    ```

    | Accuracy _(%)_ | mIoU _(%)_ | Train Time (avg per-epochs) |
    |----------------|------------|-----------------------------|
    |      80.0      |     51.1   |          0m 47s             |

##### Semantic segmentation GTA5 (synthetic dataset).
  
    ```bash
    train.py --task semantic_segmentation --dataset gta5 --num_classes 19  --root_dir GTA5_ds --augmentation none --batch_size 40 --num_workers 4 --learning_rate 0.01 --num_epochs 50 --pretrain_path pretrained/STDCNet813M_73.91 --save_model_path final_model\gta5 --optimizer sgd
    ```

    | Accuracy _(%)_ | mIoU _(%)_ | Train Time (avg per-epochs) |
    |----------------|------------|-----------------------------|
    |      81.4      |     62.5   |          1m 0s              |

##### Domain shift GTA5 -> Cityscapes (GTA5 model of previous point).

    ```bash
    train.py --task domain_shift --num_classes 19 --pretrain_path final_model\gta5\best.pth --root_dir Cityscapes_ds/Cityspaces  --batch_size 40 --num_workers 4 --learning_rate 0.01 --num_epochs 50  --save_model_path final_model\domain_shift --optimizer sgd
    ```

    | Accuracy _(%)_ | mIoU _(%)_ |
    |----------------|------------|
    |      60.7      |    20.7    |

##### Semantic segmentation GTA5 with augmentation (synthetic dataset).
    - To change augmentation change "--augmentation" with: brightness, contrast, saturation, all
  
    ```bash
    train.py --task semantic_segmentation --dataset gta5 --num_classes 19  --root_dir GTA5_ds --augmentation all --batch_size 40 --num_workers 4 --learning_rate 0.01 --num_epochs 50 --pretrain_path pretrained/STDCNet813M_73.91 --save_model_path final_model\gta5 --optimizer sgd
    ```

    | Augemtnation | Accuracy _(%)_ | mIoU _(%)_ | Train Time (avg per-epochs) |
    |--------------|----------------|------------|-----------------------------|
    | brightness   |      81.4      |     62.5   |          0m 59s             |
    | contrast     |      0         |     0      |          0m 0s              |
    | saturation   |      0         |     0      |          0m 0s              |
    | all          |      81.2      |     60.5   |          1m 0s              |

##### Domain shift GTA5 -> Cityscapes with augmentation
