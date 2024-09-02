#!/usr/bin/python
# -*- encoding: utf-8 -*-
from model.model_stages import BiSeNet
from datasets.CityScapesDataset import CityScapesDataset
from datasets.Gta5Dataset import Gta5Dataset
import torch
from torch.utils.data import DataLoader
import logging
import argparse
import numpy as np
from tensorboardX import SummaryWriter
import torch.cuda.amp as amp
from utils import poly_lr_scheduler
from utils import reverse_one_hot, compute_global_accuracy, fast_hist, per_class_iu
from tqdm import tqdm

import os
from utils import show_image_and_label
from MyArgs import MyArgs
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from PIL import Image

logger = logging.getLogger()



# Supponiamo che il codice della classe Gta5Dataset sia stato già eseguito e la classe sia disponibile

# Specifica il percorso alla directory dei dati e l'immagine di riferimento
root_dir = os.path.join(os.path.dirname(__file__), 'GTA5_ds')
reference_image_folder_path = os.path.join(os.path.dirname(__file__), 'Cityscapes_ds', 'Cityspaces', 'images', 'train')
reference_image_file_path   = os.path.join(os.path.dirname(__file__), 'Cityscapes_ds', 'Cityspaces', 'images', 'train', 'krefeld', 'krefeld_000000_027954_leftImg8bit.png')

# Crea un'istanza del dataset con l'augmentazione di tipo 'reinhard'
dataset = Gta5Dataset(root=root_dir, augmentation='reinhard', reference_image_file_path=reference_image_file_path, reference_image_folder_path=reference_image_folder_path)

# Carica un'immagine dal dataset
img_idx = 0  # Cambia questo indice per caricare altre immagini
image, label = dataset[img_idx]

# Converte il tensor dell'immagine normalizzata di nuovo a una immagine PIL per la visualizzazione
unnormalize = transforms.Compose([
    transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1/0.229, 1/0.224, 1/0.225]),
    transforms.ToPILImage()
])

# Mostra l'immagine originale e quella normalizzata
image_normalized = unnormalize(image)
image_normalized.show(title='Immagine Normalizzata')

# Carica e mostra l'immagine originale
original_image_path = dataset.data["image_path"].iloc[img_idx]
original_image = Image.open(original_image_path).resize(dataset.resize, Image.BILINEAR)
original_image.show(title='Immagine Originale')

# Carica e mostra l'immagine di riferimento
reference_image = Image.open(reference_image_file_path).resize(dataset.resize, Image.BILINEAR)
reference_image.show(title='Immagine di Riferimento')

# Alternativamente, puoi mostrare le immagini in un unico plot usando matplotlib
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.title('Immagine Originale')
plt.imshow(original_image)

plt.subplot(1, 3, 2)
plt.title('Immagine di Riferimento')
plt.imshow(reference_image)

plt.subplot(1, 3, 3)
plt.title('Immagine Normalizzata (Reinhard)')
plt.imshow(image_normalized)

plt.show()



