import os
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import pandas as pd
import numpy as np
import time

# Ensure that PyTorch is using your GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def extract_cnn_features(image_path, model, transform):
    img = Image.open(image_path).convert('RGB')
    img_t = transform(img)
    batch_t = torch.unsqueeze(img_t, 0).to(device)

    with torch.no_grad():
        model.eval()
        features = model(batch_t)
    return features.cpu().numpy().flatten()

def process_folder(folder_path, model, transform):
    features_list = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):  # Add other image formats if needed
            file_path = os.path.join(folder_path, filename)
            start_time = time.time()
            features = extract_cnn_features(file_path, model, transform)
            features_list.append(features)
            print(f"Processed {filename} in {time.time() - start_time:.2f} seconds.")
    return np.mean(features_list, axis=0)

def main(directory_path):
    # Use a pre-trained model, e.g., ResNet
    model = models.resnet50(pretrained=True).to(device)

    # Remove the last layer (classification layer) to get feature extractor
    model = torch.nn.Sequential(*(list(model.children())[:-1]))

    # Define the image transformations
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    all_features = []
    total_folders = len(os.listdir(directory_path))
    processed_folders = 0

    for folder_name in os.listdir(directory_path):
        folder_path = os.path.join(directory_path, folder_name)
        if os.path.isdir(folder_path):
            print(f"Processing folder {folder_name} ({processed_folders+1}/{total_folders})...")
            start_time = time.time()
            avg_features = process_folder(folder_path, model, transform)
            all_features.append([folder_name] + avg_features.tolist())
            print(f"Completed folder {folder_name} in {time.time() - start_time:.2f} seconds.")
            processed_folders += 1

    columns = ['ObjectID'] + [f'Feature{i}' for i in range(len(all_features[0]) - 1)]
    df = pd.DataFrame(all_features, columns=columns)
    df.to_csv('object_features.csv', index=False)
    print("Feature extraction complete. CSV file saved.")

main('views')

