import torch
import torch.optim as optim
import torch.nn as nn
from model import SimpleCNN
from data_loader import get_data_loaders
from utils import evaluate

def train_model(model, train_loader, num_epochs=5, lr=0.001):
    criterion = nn.CrossEntropyLoss()
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(num_epochs):
        model.train()  
        running_loss = 0.0
        correct = 0
        total = 0
        for images, labels in train_loader:
            optimizer.zero_grad()  

            outputs = model(images)

            loss = criterion(outputs, labels)
            
            loss.backward()
            
            optimizer.step()

            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = 100 * correct / total
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%")

    return model

def save_model(model, path='model.pth'):
    torch.save(model.state_dict(), path)
    print(f"Модель сохранена в {path}")

