import os
import json
import pandas as pd
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
from torchvision.transforms import RandomApply
import random

class Gta5Dataset(Dataset):
    def __init__(self, root, augmentation='none', reference_image_file_path=None, reference_image_folder_path=None, dimension=(1024, 512)):
        super(Gta5Dataset, self).__init__()

        self.root = os.path.normpath(root)
        self.resize = dimension

        mapping_path = os.path.join(os.path.dirname(__file__), 'gta5_mapping.json')
        self.lb_map = self._load_label_map(mapping_path)

        # Determine reference image for Reinhard normalization
        if reference_image_folder_path is not None:
            reference_image_file_path = self._select_random_image(reference_image_folder_path)
            print("Percorso", reference_image_file_path)

        # Define the transform pipeline for images based on the augmentation parameter
        if augmentation == 'color_jitter':
            color_jitter = transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)
            transform_list = [
                RandomApply([color_jitter], p=0.5),  # Apply randomly
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))  # Normalization
            ]
        elif augmentation == 'reinhard' and reference_image_file_path is not None:
            self.ref_means, self.ref_stds = self._calculate_reference_stats(reference_image_file_path)
            transform_list = [
                transforms.ToTensor(),
                transforms.Lambda(lambda img: self._reinhard_normalization(img)),  # Reinhard normalization
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))  # Normalization
            ]
        else:
            transform_list = [
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))  # Normalization
            ]
        
        self.to_tensor = transforms.Compose(transform_list)
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

        # Convert to tensor and apply transformations
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

    def _calculate_reference_stats(self, reference_image_path):
        """ Calculate the mean and std of the reference image. """
        ref_image = Image.open(reference_image_path)
        ref_image = ref_image.resize(self.resize, Image.BILINEAR)
        ref_tensor = transforms.ToTensor()(ref_image)

        # Calculate mean and std per channel
        means = ref_tensor.mean(dim=(1, 2))
        stds = ref_tensor.std(dim=(1, 2))

        return means, stds

    def _reinhard_normalization(self, img):
        """ Apply Reinhard normalization using precomputed reference stats. """
        # Calculate the mean and std dev per channel of the input image
        img_means = img.mean(dim=(1, 2))
        img_stds = img.std(dim=(1, 2))
        
        # Normalize image by shifting its mean and adjusting its contrast
        img = (img - img_means.view(3, 1, 1)) / (img_stds.view(3, 1, 1) + 1e-5)
        img = img * self.ref_stds.view(3, 1, 1) + self.ref_means.view(3, 1, 1)
        
        return img

    def _select_random_image(self, folder_path):
        """ Select a random image from a folder and its subfolders. """
        all_images = []
        for root, _, files in os.walk(folder_path):
            all_images += [os.path.join(root, file) for file in files if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        
        if not all_images:
            raise ValueError("No images found in the specified reference folder path.")
        
        return random.choice(all_images)
