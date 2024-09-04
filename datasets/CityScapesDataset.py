import os
import pandas as pd
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset

def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')
	

class CityScapesDataset(Dataset):
    def __init__(self, root_dir, mode='train', dimension= (2048, 1024), transform=None):
        """
        Inizializza il dataset CityScapes.

        Args:
            root_dir (str): Directory principale del dataset.
            mode (str): Modalità del dataset ('train', 'val', 'test').
            dimension (int, int): Altezza e Larghezza a cui ridimensionare le immagini.
        """
        super(CityScapesDataset, self).__init__()

        self.root_dir = root_dir
        self.mode = mode  # mode can be 'train', 'val', or 'test'
        self.transform = transform
        self.resize = dimension

        # Cityscapes directory structure
        self.images_dir = os.path.join(self.root_dir, 'images', self.mode)
        self.labels_dir = os.path.join(self.root_dir, 'gtFine', self.mode)

        # Get list of all image and label files
        self.images_paths = []
        self.labels_paths = []

        # Trasformazioni da applicare alle immagini e alle etichette
        normalizer = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        self.to_tensor = transforms.Compose([
            transforms.Resize(self.resize),
            transforms.ToTensor(),
            normalizer  # Aggiungi la normalizzazione qui
        ])
        self.to_tensor_label = transforms.Compose([
            transforms.Resize(self.resize, interpolation=Image.NEAREST),
            transforms.PILToTensor()
        ])

        #caricamento percorsi di immagini ed etichette
        self.images_paths = self.__get_file_paths__(self.images_dir, ['.png', '.jpg', '.jpeg'])
        #escludiamo color per prendere solo le maschere
        self.labels_paths = self.__get_file_paths__(self.labels_dir, ['.png'], include_keywords=['train'])

        # Creazione del DataFrame con percorsi di immagini e etichette
        self.data = pd.DataFrame({
            "image_path": sorted(self.images_paths),
            "label_path": sorted(self.labels_paths)
        })

    def __get_file_paths__(self, dir_path, valid_extensions, include_keywords=None):
        """
        Restituisce una lista di percorsi di file validi in una directory.

        Args:
            dir_path (str): Percorso della directory da esaminare.
            valid_extensions (list): Estensioni valide per i file.
            include_keywords (list): Parole chiave da includere nei nomi dei file (opzionale).

        Returns:
            list: Lista di percorsi dei file validi.
        """
        file_paths = []
        for root, _, files in os.walk(dir_path):
            for file in files:
                if any(file.lower().endswith(ext) for ext in valid_extensions):
                    # Se include_keywords è None, includi tutte le immagini
                    if include_keywords is None or any(keyword in file.lower() for keyword in include_keywords):
                        file_paths.append(os.path.join(root, file))
        return file_paths

    def __getitem__(self, idx):
        """
        Restituisce una coppia immagine-etichetta trasformata.

        Args:
            idx (int): Indice della coppia immagine-etichetta.

        Returns:
            tuple: Coppia (immagine, etichetta) entrambe come tensori.
        """
        image_path = self.data.iloc[idx]["image_path"]
        label_path = self.data.iloc[idx]["label_path"]

        image = pil_loader(image_path)
        label = Image.open(label_path)

        image = self.to_tensor(image)
        label = self.to_tensor_label(label)

        return image, label

    def __len__(self):
        #restituisce il numero di coppie immagine-etichetta
        return len(self.data)
