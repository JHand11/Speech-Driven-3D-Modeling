import os
import torch
import clip
from PIL import Image
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader

print("starting up")
# Configuration
views_directory = "views"
batch_size = 64  # Adjust based on GPU memory
num_workers = 8  # Adjust based on your CPU cores
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the CLIP model
print("preprocessing model")
model, preprocess = clip.load("ViT-B/32", device=device)
model.eval()  # Set the model to evaluation mode
print("model evaluated")

class ImageDataset(Dataset):
    def __init__(self, views_directory):
        print("initializing image dataset")
        self.objects = self._load_objects(views_directory)

    def __len__(self):
        return len(self.objects)

    def __getitem__(self, idx):
        image_paths = self.objects[idx]
        images = [preprocess(Image.open(path).convert("RGB")) for path in image_paths]
        images = torch.stack(images)
        return images

    def _load_objects(self, directory):
        print("load objects")
        objects = []

        processed_folders = 0

        for folder_name in os.listdir(directory):
            folder_path = os.path.join(directory, folder_name)
            if os.path.isdir(folder_path):
                image_paths = [os.path.join(folder_path, img) for img in os.listdir(folder_path)]
                if len(image_paths) == 4:  # Ensure there are exactly 4 images per object
                    objects.append(image_paths)

            processed_folders += 1
            print(processed_folders)
            if processed_folders % 10 == 0:  # Print progress every 10 folders or at the end
                print(f"Processed {processed_folders} folders")

        return objects


def extract_clip_features(batch):
    batch = batch.to(device)
    with torch.no_grad():
        features = model.encode_image(batch.view(-1, *batch.shape[-3:]))  # Flatten batch for processing
    features = features.view(batch_size, 4, -1)  # Reshape back to separate objects
    averaged_features = features.mean(dim=1)  # Average features for each object
    return averaged_features.cpu().numpy()

# Initialize dataset and dataloader
print("defining dataset")
dataset = ImageDataset(views_directory)
print("defining dataloader")
data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True)

# Process images and extract features
object_features = []
print("starting batches")
for batch in data_loader:
    features = extract_clip_features(batch)
    object_features.extend(features)
    print(f"Processed {len(object_features)} objects out of {len(dataset)}")

# Convert to DataFrame and save features
feature_columns = [f'feature_{i}' for i in range(object_features[0].shape[0])]
df_features = pd.DataFrame(object_features, columns=feature_columns)

csv_path = "features.csv"
df_features.to_csv(csv_path, index=False)
print("Feature extraction completed and saved to 'features.csv'.")

