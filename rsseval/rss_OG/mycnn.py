import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 28, 128)
        self.fc2 = nn.Linear(128, 8)  

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  
        x = self.pool(F.relu(self.conv2(x))) 
        x = x.view(x.size(0), -1)            
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = x.view(x.shape[0], 4 , 2)
        return F.softmax(x, dim=-1)         

# Esempio di utilizzo:
if __name__ == "__main__":
    model = CNN()
    # Creiamo un input dummy con batch_size=8
    input_tensor = torch.randn(8, 1, 28, 112)
    output = model(input_tensor)
    print("Output shape:", output.shape)  # Dovrebbe essere [8, 20]