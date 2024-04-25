import os
import torch
import clip
import pandas as pd
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torch.multiprocessing import Pool, Process, set_start_method

# Try to set the start method to spawn to avoid issues
try:
    set_start_method('spawn')
except RuntimeError:
    pass

# Configuration
original_views_directory = "views"
preprocessed_views_directory = "preprocessed_views"
os.makedirs(preprocessed_views_directory, exist_ok=True)

# Load the CLIP model (make sure it's compatible with multiprocessing)
model, preprocess = clip.load("ViT-B/32")
model.eval()

# Custom dataset to load image paths
class ImagePathDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.image_paths = [os.path.join(root, f) for root, _, files in os.walk(root_dir) for f in files]
        print(f"Total images found: {len(self.image_paths)}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        return self.image_paths[idx]

def process_and_save_images(batch, gpu_id, batch_index, total_batches):
    device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    processed_images = []

    for img_path in batch:
        image = preprocess(Image.open(img_path).convert("RGB")).unsqueeze(0).to(device)
        with torch.no_grad():
            processed_image = model.encode_image(image)
        processed_images.append(processed_image.cpu())

    batch_name = os.path.basename(os.path.dirname(img_path))
    torch.save(torch.stack(processed_images), os.path.join(preprocessed_views_directory, f'{batch_name}.pt'))
    print(f"Processed batch {batch_index + 1}/{total_batches} on GPU {gpu_id}")

# Function to distribute batches across GPUs
def process_in_parallel(batch_size=1000, num_workers=10):
    dataset = ImagePathDataset(original_views_directory)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    batches = list(data_loader)
    num_batches = len(batches)
    print("Total batches to process:", num_batches)

    pool = Pool(num_workers)
    for i, batch in enumerate(batches):
        gpu_id = i % torch.cuda.device_count()
        pool.apply_async(process_and_save_images, args=(batch, gpu_id, i, num_batches))

    pool.close()
    pool.join()

# Run the parallel processing
print("Starting image preprocessing and saving...")
process_in_parallel()

device = 'cuda'
preprocessed_views_directory = "preprocessed_views"

# Dataset to load preprocessed image tensors
class TensorDataset(Dataset):
    def __init__(self, directory):
        self.directory = directory
        self.files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.pt')]
        print(f"Total tensor files found: {len(self.files)}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = torch.load(self.files[idx])
        return data

# Function to extract and average features
def extract_and_average_features(data_loader):
    all_features = []
    total_objects = len(data_loader.dataset)
    print("Starting feature extraction...")

    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            batch = batch.to(device)
            features = model.encode_image(batch)
            averaged_features = features.mean(dim=0)  # Average across images
            all_features.append(averaged_features.cpu().numpy())
            print(f"Processed object {i + 1}/{total_objects}")

    return all_features

# Load dataset and create DataLoader
tensor_dataset = TensorDataset(preprocessed_views_directory)
data_loader = DataLoader(tensor_dataset, batch_size=1000, shuffle=False)

# Extract and average features
object_features = extract_and_average_features(data_loader)

# Convert to DataFrame
feature_columns = [f'feature_{i}' for i in range(object_features[0].shape[0])]
df_features = pd.DataFrame(object_features, columns=feature_columns)

# Save to CSV or further processing
csv_path = "object_features.csv"
df_features.to_csv(csv_path, index=False)
print("Feature extraction completed and saved to 'object_features.csv'.")

