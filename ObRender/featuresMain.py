import os
import torch
import clip
from PIL import Image
import pandas as pd
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset

# Configuration
views_directory = "views"
batch_size = 4096  # Adjust based on how many folders (objects) you want to process in a batch
device = "cuda" if torch.cuda.is_available() else "cpu"
print(torch.cuda.is_available())

# Load the CLIP model
print("Loading CLIP model...")
model, preprocess = clip.load("ViT-B/32", device=device)
model.eval()  # Set the model to evaluation mode
print("Model loaded and set to evaluation mode.")

# Image preprocessing
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
])
print("Image preprocessing configured.")

# Dataset to handle folders
class FolderDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.folder_paths = [os.path.join(root_dir, entry) for entry in os.listdir(root_dir)]
        print(f"Total folders loaded: {len(self.folder_paths)}")


    def __len__(self):
        return len(self.folder_paths)

    def __getitem__(self, idx):
        folder_path = self.folder_paths[idx]
        image_paths = [os.path.join(folder_path, img) for img in os.listdir(folder_path)]
        images = [preprocess(Image.open(path).convert("RGB")) for path in image_paths]
        return torch.stack(images), folder_path

# DataLoader for batch processing
print("Preparing dataset and dataloader...")
dataset = FolderDataset(views_directory)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
print("Dataset and DataLoader are ready.")

# Function to process a batch of folders
def process_batch(batch):
    batch_features = []
    for images, folder_path in zip(*batch):
        images = images.to(device)
        with torch.no_grad():
            features = model.encode_image(images.view(-1, *images.shape[-3:]))  # Flatten batch for processing
        averaged_features = features.view(4, -1).mean(dim=0)  # Average features for each object
        object_id = os.path.basename(folder_path)
        batch_features.append((object_id, averaged_features.cpu().numpy()))
    return batch_features

# Process folders in batches and extract features
print("Starting feature extraction...")
object_features = []
for i, batch in enumerate(data_loader):
    features = process_batch(batch)
    object_features.extend(features)
    print(f"Processed {len(object_features)} objects out of {len(dataset)}")

# Convert to DataFrame and save features
feature_columns = [f'feature_{i}' for i in range(object_features[0][1].shape[0])]
df_features = pd.DataFrame([features[1] for features in object_features], columns=feature_columns)
df_features.insert(0, 'object_id', [features[0] for features in object_features])

# Save to CSV
csv_path = "object_features.csv"
df_features.to_csv(csv_path, index=False)
print("Feature extraction completed and saved to 'object_features.csv'.")

