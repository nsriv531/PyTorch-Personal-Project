import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
batch_size = 64
learning_rate = 0.001
epochs = 10

# Data Preprocessing and Loading the MNIST Dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # Normalizing with mean and std deviation
])

train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Building the CNN model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # Convolutional layer 1: 1 input channel (grayscale), 16 output channels, kernel size 3x3
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        # Convolutional layer 2: 16 input channels, 32 output channels, kernel size 3x3
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        # Fully connected layer 1
        self.fc1 = nn.Linear(32*7*7, 128)
        # Fully connected layer 2 (Output layer)
        self.fc2 = nn.Linear(128, 10)

        # Max Pooling layer
        self.pool = nn.MaxPool2d(2, 2)
    
    def forward(self, x):
        # Convolutional + ReLU + Max Pooling
        x = self.pool(F.relu(self.conv1(x)))  # Output size: (16, 14, 14)
        x = self.pool(F.relu(self.conv2(x)))  # Output size: (32, 7, 7)
        # Flatten the tensor for fully connected layers
        x = x.view(-1, 32*7*7)
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)  # No activation for the output (cross-entropy loss will handle that)
        return x

# Model initialization
model = CNN().to(device)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()  # Cross-entropy loss for classification
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training the model
def train(model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    total_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        # Forward pass
        output = model(data)
        loss = criterion(output, target)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()

        if batch_idx % 100 == 0:
            print(f'Epoch {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}], Loss: {loss.item():.4f}')
    avg_loss = total_loss / len(train_loader)
    print(f'Epoch {epoch}, Average Training Loss: {avg_loss:.4f}')

# Evaluating the model
def evaluate(model, device, test_loader):
    model.eval()
    correct = 0
    test_loss = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()  # Accumulate loss
            pred = output.argmax(dim=1, keepdim=True)  # Get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    test_loss /= len(test_loader)
    accuracy = 100. * correct / len(test_loader.dataset)
    
    print(f'Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)')

# Main training and evaluation loop
for epoch in range(1, epochs + 1):
    train(model, device, train_loader, optimizer, criterion, epoch)
    evaluate(model, device, test_loader)

# Save the trained model
torch.save(model.state_dict(), "mnist_cnn.pth")