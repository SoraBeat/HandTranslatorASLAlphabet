import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import Dataset
from PIL import Image
import os

# Definición de la arquitectura ResNet
class ResNet(nn.Module):
    def __init__(self, num_classes):
        super(ResNet, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.resnet(x)

# Dataset personalizado para cargar las imágenes
class ASLDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.file_list = []
        self.class_to_idx = {}
        self.idx_to_class = {}

        # Obtener la lista de categorías (subdirectorios)
        categories = os.listdir(data_dir)
        categories.sort()  # Ordenar alfabéticamente para asegurar consistencia

        for i, category in enumerate(categories):
            category_path = os.path.join(data_dir, category)
            if os.path.isdir(category_path):
                for file_name in os.listdir(category_path):
                    file_path = os.path.join(category_path, file_name)
                    self.file_list.append((file_path, i))
                self.class_to_idx[category] = i
                self.idx_to_class[i] = category

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_name, label = self.file_list[idx]
        image = Image.open(img_name)
        if self.transform:
            image = self.transform(image)
        return image, label
