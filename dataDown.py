import os
import json
import torch
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader

os.makedirs('dataset', exist_ok=True)
os.makedirs('mnist/training', exist_ok=True)
os.makedirs('mnist/testing', exist_ok=True)

transform = transforms.Compose([
    transforms.ToTensor(),
])

train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_data = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_data, batch_size=1, shuffle=True)
test_loader = DataLoader(test_data, batch_size=1, shuffle=False)

train_images = []
for i, (image, label) in enumerate(train_loader):
    if i >= 60000:  
        break
    image_filename = f"mnist/training/{i+1:05d}.png"
    save_image(image, image_filename)
    train_images.append({'filename': image_filename, 'label': int(label.item())})

test_images = []
for i, (image, label) in enumerate(test_loader):
    if i >= 10000:  
        break
    image_filename = f"mnist/testing/{i+1:05d}.png"
    save_image(image, image_filename)
    test_images.append({'filename': image_filename, 'label': int(label.item())})


dataset_images = train_images + test_images
for i, (image, label) in enumerate(train_loader):
    if i >= 40000:  
        break
    image_filename = f"dataset/{i+1:05d}.png"
    save_image(image, image_filename)
    dataset_images.append({'filename': image_filename, 'label': int(label.item())})

json_data = {
    'images': dataset_images
}
with open('dataset/annotations.json', 'w') as f:
    json.dump(json_data, f)

print("Данные успешно сохранены!")
