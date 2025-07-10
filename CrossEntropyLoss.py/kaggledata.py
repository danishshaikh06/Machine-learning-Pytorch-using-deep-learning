import kagglehub

# Download latest version
path = kagglehub.dataset_download("asaniczka/mammals-image-classification-dataset-45-animals")

print("Path to dataset files:", path)