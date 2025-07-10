import torch
print("CUDA Available:", torch.cuda.is_available())
print("Device Name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU found")

'''import torch
print(torch.version.cuda)  # Check CUDA version'''

