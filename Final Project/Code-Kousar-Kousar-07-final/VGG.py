import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# Your provided VGG code
class SimpleVGG(nn.Module):
    def __init__(self, num_classes=2):  # Updated to 2 classes (binary classification)
        super(SimpleVGG, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.5),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(64 * 7 * 7, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# Your provided dataset information
data_dir = './Dataset1'
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Create datasets
train_data = datasets.ImageFolder(data_dir + '/train', transform=train_transforms)
test_data = datasets.ImageFolder(data_dir + '/train', transform=test_transforms)  # Fixed to use the test dataset

# Create DataLoaders
trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
testloader = torch.utils.data.DataLoader(test_data, batch_size=64)
dataloaders_dict = {'train': trainloader, 'val': testloader}

# Instantiate the simpler model
model = SimpleVGG(num_classes=2)  # Two classes: Cat and Dog

# CrossEntropyLoss is appropriate for binary classification with a single output and softmax activation
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.1)

model.to('cuda:3')

# Training loop 
num_epochs = 30
train_loss_history = []
val_loss_history = []
best_val_loss = float('inf')


for epoch in range(num_epochs):
    print('Epoch', epoch)
    model.train()
    train_loss = 0.0
    for inputs, labels in dataloaders_dict['train']:
        inputs, labels = inputs.to('cuda:3'), labels.to('cuda:3')
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * inputs.size(0)

    avg_train_loss = train_loss / len(dataloaders_dict['train'].dataset)
    train_loss_history.append(avg_train_loss)

    # Evaluation on the validation set
    model.eval()
    val_loss = 0.0
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for inputs, labels in dataloaders_dict['val']:
            inputs, labels = inputs.to('cuda:3'), labels.to('cuda:3')
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            predictions = torch.argmax(outputs, dim=1)
            val_loss += loss.item() * inputs.size(0)
            total_correct += (predictions == labels).sum().item()
            total_samples += labels.size(0)

    avg_val_loss = val_loss / len(dataloaders_dict['val'].dataset)
    val_loss_history.append(avg_val_loss)
    accuracy = total_correct / total_samples
    print(f"Accuracy on the test set: {accuracy * 100:.2f}%")

    print(f'Epoch [{epoch+1}/{num_epochs}] => '
          f'Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}')


# Plotting the loss graph
plt.plot(train_loss_history, label='train')
plt.plot(val_loss_history, label='val')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.xlim(0, num_epochs)
plt.savefig('VGG.png')
plt.show()
