import torch
import model
import dataset
from dataset import train_loader
from model import device, criterion, model

val_loss = 0.0
correct = 0
total = 0
model.eval()

def model_eval(val_loss, correct, total):
    with torch.no_grad():
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            break

    print(f"Validation Loss: {val_loss/len(train_loader):.4f}, Accuracy: {100 * correct / total:.2f}%")
    
model_eval(val_loss, correct, total)
    