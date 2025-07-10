import torch
import torch.nn as nn
from torchvision import transforms, datasets
from PIL import Image

# Define the model architecture
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

# Model parameters
input_size = 64 * 64 * 3
hidden_size = 128
num_classes = 45  # Manually set the number of classes

# Load model
model = AnimalClassifier(input_size, hidden_size, num_classes)
model.load_state_dict(torch.load("my_model.pth"))
model.eval()  #disables dropout/batch norm (if present) for inference.

# Define transformations
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),# Normalizes the image using mean=0.5, std=0.5, making the values range from [-1,1] (helps training).
])
dataset_path = 'C:\\Users\\shaik\\.cache\\kagglehub\\datasets\\asaniczka\\mammals-image-classification-dataset-45-animals\\versions\\1\\mammals'
train_dataset = datasets.ImageFolder(root=dataset_path, transform=transform)
# Load an image
image_path = "cat2.jpg"  # Replace with actual image path
image = Image.open(image_path).convert('RGB')

# Preprocess the image
image = transform(image)
image = image.view(1, -1)  # Flatten and add batch dimension

# Predict
with torch.no_grad():
    output = model(image)
    predicted_class = torch.argmax(output, dim=1).item() #for example:tensor([[-1.2, 2.5, 0.8]])  # These are raw scores (logits)

# Manually define class names or reload train_dataset if necessary
class_names = train_dataset.classes  # Replace with actual class names
predicted_label = class_names[predicted_class]

print(f"Predicted Class: {predicted_label}")


