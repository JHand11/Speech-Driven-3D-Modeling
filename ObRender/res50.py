import os
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm

class ImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        print("initialize dataset")
        self.root_dir = root_dir
        self.transform = transform
        self.folders = [os.path.join(root_dir, f) for f in os.listdir(root_dir)]
        print("Done initializing")

    def __len__(self):
        return len(self.folders)

    def __getitem__(self, idx):
        folder = self.folders[idx]
        images = os.listdir(folder)
        if not images:
            return None, None  # No images in folder
        image_path = os.path.join(folder, images[0])  # Load the first image
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, folder

# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Initialize dataset and dataloader
dataset = ImageDataset(root_dir='views', transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)

# Load a pre-trained model
model = models.resnet50(pretrained=True)
model = model.to('cuda')
model.eval()

# Function to extract features
def extract_features(model, dataloader):
    features = []
    for inputs, paths in tqdm(dataloader, desc="Extracting Features"):
        if inputs is None:
            continue
        inputs = inputs.to('cuda')
        with torch.no_grad():
            outputs = model(inputs)
        features.extend(zip(paths, outputs.cpu().numpy()))
    return features

# Extract features
features = extract_features(model, dataloader)

# Convert to DataFrame
df = pd.DataFrame(features, columns=['path', 'features'])

# Save to CSV
df.to_csv('extracted_features.csv', index=False)

