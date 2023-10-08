import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
import os
from PIL import Image
from tqdm import tqdm

# Definición de la arquitectura ResNet
class ResNet(nn.Module):
    def __init__(self, num_classes):
        super(ResNet, self).__init__()
        self.resnet = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
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

# Transformaciones para preprocesar las imágenes
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# Directorio que contiene tus imágenes categorizadas
data_dir = "C:/Users/elian/OneDrive/Escritorio/asl_alphabet_train/asl_alphabet_train"

# Crear el dataset personalizado
dataset = ASLDataset(data_dir, transform=transform)

# Dividir el dataset en conjuntos de entrenamiento, validación y prueba
train_size = int(0.7 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

# Cargar los datos en lotes (batch)
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# Número total de clases (incluyendo las categorías adicionales)
num_classes = len(dataset.class_to_idx)

# Crear la instancia del modelo y el optimizador
model = ResNet(num_classes)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Entrenar el modelo
num_epochs = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model.to(device)
best_val_loss = float('inf')

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0  # Agrega una variable para el seguimiento de la pérdida de entrenamiento
    total_batches = len(train_loader)

    # Utiliza tqdm para crear una barra de progreso
    with tqdm(total=total_batches, desc=f'Epoch {epoch + 1}/{num_epochs}', unit='batch') as pbar:
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()  # Agrega la pérdida actual al seguimiento
            pbar.update(1)  # Actualiza la barra de progreso
            pbar.set_postfix(loss=f'{train_loss / (pbar.n * batch_size):.4f}')  # Actualiza la pérdida en la barra de progreso

    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            val_loss += criterion(outputs, labels)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_loss /= len(val_loader)
    accuracy = 100 * correct / total

    print(f"Epoch [{epoch + 1}/{num_epochs}] - "
          f"Loss: {val_loss:.4f}, "
          f"Validation Accuracy: {accuracy:.2f}%")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'best_model.pth')

print("Entrenamiento finalizado!")

# Evaluar el modelo en el conjunto de prueba
model.load_state_dict(torch.load('best_model.pth'))
model.eval()
test_correct = 0
test_total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        test_total += labels.size(0)
        test_correct += (predicted == labels).sum().item()

test_accuracy = 100 * test_correct / test_total
print(f"Test Accuracy: {test_accuracy:.2f}%")
