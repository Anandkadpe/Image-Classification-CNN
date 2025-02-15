import torch
from torch import nn
import torch.nn.functional as F

class TrafficSigCNN(nn.Module):
  def __init__(self, num_classes = 43):
    super(TrafficSigCNN, self).__init__()
    self.network = nn.Sequential(
    nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding= 'same'),
    nn.ReLU(),
    nn.BatchNorm2d(32),
    nn.Dropout(0.3),
    nn.MaxPool2d(kernel_size=3, stride=2),

    nn.Conv2d(32, 64, kernel_size=3, stride=1, padding='same'),
    nn.ReLU(),
    nn.BatchNorm2d(64),
    nn.Dropout(0.3),
    nn.MaxPool2d(kernel_size=3, stride=2),

    nn.Conv2d(64, 128, kernel_size=3, stride=1, padding='same'),
    nn.ReLU(),
    nn.BatchNorm2d(128),
    nn.Dropout(0.3),
    nn.MaxPool2d(kernel_size=3, stride= 2),
    )

    self.flattened_size = self._get_flattened_size()

    self.classifier = nn.Sequential(
    nn.Flatten(),
    nn.Linear( self.flattened_size, 256),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear( 256 , num_classes),
    nn.Dropout(0.5))

  def _get_flattened_size(self):
    with torch.no_grad():
        sample_input = torch.randn(1, 3, 32, 32)
        output = self.network(sample_input)
        return output.view(1, -1).size(1)

  def forward(self, x):
    x = self.network(x)
    x = self.classifier(x)
    return x

def train_model():
    model = TrafficSigCNN(num_classes= 43)
    return model
    
criterion = nn.CrossEntropyLoss()
model = train_model()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
