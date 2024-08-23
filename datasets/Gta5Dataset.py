import os
import json
import pandas as pd
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset

class Gta5Dataset(Dataset):
    def __init__(self, root, dimension=(1024, 512)):
        super(Gta5Dataset, self).__init__()

        self.root = os.path.normpath(root)
        self.resize = dimension

        mapping_path = os.path.join(os.path.dirname(__file__), 'gta5_mapping.json')
        self.lb_map = self._load_label_map(mapping_path)

        # Define the transform pipeline for images and labels
        normalizer = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            normalizer  # Aggiungi la normalizzazione qui
        ])
        self.to_tensor_label = transforms.PILToTensor()

        # List all image and label files
        image_files = sorted(os.listdir(os.path.join(self.root, 'images')))
        label_files = sorted(os.listdir(os.path.join(self.root, 'labels')))

        # Ensure there is a matching number of images and labels
        assert len(image_files) == len(label_files), "Mismatch between number of images and labels."

        self.data = pd.DataFrame({
            "image_path": [os.path.join(self.root, 'images', img) for img in image_files],
            "label_path": [os.path.join(self.root, 'labels', lbl) for lbl in label_files]
        })

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image_path = self.data["image_path"].iloc[idx]
        label_path = self.data["label_path"].iloc[idx]

        # Load and resize image and label
        image = Image.open(image_path).resize(self.resize, Image.BILINEAR)
        label = Image.open(label_path).resize(self.resize, Image.NEAREST)

        # Convert to tensor and normalize the image
        image = self.to_tensor(image)
        label = self.to_tensor_label(label)
        label = self._convert_labels(label)

        return image, label

    def _load_label_map(self, json_path):
        with open(json_path, 'r') as fr:
            labels_info = json.load(fr)
        return {el['id']: el['trainId'] for el in labels_info if 'id' in el}

    def _convert_labels(self, label):
        # Convert the label tensor to a numpy array, apply mapping, then convert back to tensor
        label_np = label.numpy()
        label_np = np.vectorize(self.lb_map.get)(label_np)
        return torch.tensor(label_np, dtype=torch.long)
