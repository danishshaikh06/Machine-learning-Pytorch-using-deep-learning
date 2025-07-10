import torch
import torch.nn as nn
import torchvision
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import os

# Select device (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define the Neural Network
class AnimalClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(AnimalClassifier, self).__init__()
        self.linear = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.output = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.relu(self.linear(x))
        x = self.output(x)
        return x

# Data Transformations
transform = transforms.Compose([
    transforms.Resize((64, 64)),  
    transforms.RandomHorizontalFlip(),  # Flip images randomly
    transforms.RandomRotation(10),  # Slight rotation
    transforms.ToTensor(),  
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # Normalizes the image using mean=0.5, std=0.5, making the values range from [-1,1] (helps training).
])
# Load Dataset
dataset_path = 'C:\\Users\\shaik\\.cache\\kagglehub\\datasets\\asaniczka\\mammals-image-classification-dataset-45-animals\\versions\\1\\mammals'
train_dataset = datasets.ImageFolder(root=dataset_path, transform=transform) # train_dataset[i] contain(imagedata_tensor,labels) ImageFolder:It gives the tensor of images
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

#mammals_path = 'C:\\Users\\shaik\\.cache\\kagglehub\\datasets\\asaniczka\\mammals-image-classification-dataset-45-animals\\versions\\1\\mammals'
#print(os.listdir(mammals_path))

'''print(f"Number of classes: {len(train_dataset.classes)}")# to print number of classes in dataset
print(f"Class names: {train_dataset.classes}") # to check the class name in data set
for i, (images, labels) in enumerate(train_loader):
    print(f"Labels in first batch: {labels}")
    break'''

# Split into train and validation
torch.manual_seed(42)  # For reproducibility
train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_data, val_data = torch.utils.data.random_split(train_dataset, [train_size, val_size])
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False)

# Model Parameters
input_size = 64 * 64 * 3  # Flattened image size
hidden_size = 64
num_classes = len(train_dataset.classes)  # Number of unique animal classes
# Initialize Model, Loss Function, and Optimizer
model = AnimalClassifier(input_size, hidden_size, num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# Training Loop
epochs = 30
for epoch in range(epochs):
    for images, labels in train_loader:  # Load data in batches
        train_loss=0
        images = images.view(images.shape[0], -1).to(device)  # Flatten and move to device
        labels = labels.to(dtype=torch.long).to(device)  # Move labels to device and ensure correct type

        # Forward pass
        y_pred = model(images)

        # Compute loss
        loss = criterion(y_pred, labels)
        train_loss += loss.item()
        # Backward pass
        loss.backward()

        #weight updation
        optimizer.step()

        #make gradient zero for next iteration
        optimizer.zero_grad()
        
        # average loss of  batches
        train_loss /= len(train_loader) 

        # Validation phase
    model.eval()  # Set model to evaluation mode
    val_loss = 0
    with torch.no_grad():  # No gradients needed for validation
        for images, labels in val_loader:
            images = images.view(images.shape[0], -1).to(device)
            labels = labels.to(device)
            y_pred = model(images)
            val_loss += criterion(y_pred, labels).item()
    val_loss /= len(val_loader)  # Average over batches

    # Print every 5 epochs (adjust as needed)
    if epoch % 2 == 0:
        print(f'Epoch {epoch}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        
#to save the model
torch.save(model.state_dict(), "my_model.pth")

### Bottom Line
#- **Linearity** (e.g., `nn.Linear`): Straight-line math, great for basic transformations.
#- **Non-Linearity** (e.g., `ReLU`): Adds bends and twists, crucial for modeling real-world complexity.
'''
model = AnimalClassifier(input_size, hidden_size, num_classes)
model.load_state_dict(torch.load("animalclassifier.pth"))
model.eval()  # Set to evaluation mode'''

#model.load_state_dict(torch.load("animalclassifier.pth")) loads the weights.
#model.eval() disables dropout/batch norm (if present) for inference.


'''import torch

# Without seed
print(torch.rand(3))  # Different every run, e.g., [0.23, 0.67, 0.91]
print(torch.rand(3))  # Different again, e.g., [0.45, 0.12, 0.88]

# With seed
torch.manual_seed(42)
print(torch.rand(3))  # [0.19, 0.98, 0.72] (exact numbers depend on PyTorch version)
torch.manual_seed(42)  # Reset seed
print(torch.rand(3))  # [0.19, 0.98, 0.72] again—same as above'''