import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader

def img_transforms():
    transform = transforms.Compose([
    transforms.Resize((32, 32)),  # Resize all images to 32x32
    transforms.ToTensor(),         # Convert images to tensors
    transforms.Normalize((0,), (1,))  # Normalize (for better training stability)
])
    return transform

# steup a training data
train_data = torchvision.datasets.GTSRB(
    root = "data", #where to download the data
    split = "train", # do we want a training dataset --> True
    download = True, # do we want to download the dataset --> yes
    transform = img_transforms(), # how do we want to trasform the data
    target_transform = None  # how do we want to transform the labels
)

# setting up the test data

test_data = torchvision.datasets.GTSRB(
    root = "data", #where to download the data
    split = "test", # do we want a training dataset --> False
    download = True, # do we want to download the dataset --> yes
    transform = img_transforms(), # how do we want to trasform the data
    target_transform = None # how do we want to transform the labels
)


train_loader = DataLoader(train_data,
                        batch_size=64,
                        shuffle=True)
test_loader = DataLoader(test_data,
                        batch_size=64,
                        shuffle=False)
 
