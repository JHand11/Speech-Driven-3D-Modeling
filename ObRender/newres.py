import os
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import pandas as pd
import numpy as np
from multiprocessing import Pool, cpu_count
import torch.multiprocessing as mp

def extract_cnn_features(image_path, model, transform, device):
    img = Image.open(image_path).convert('RGB')
    img_t = transform(img)
    batch_t = torch.unsqueeze(img_t, 0).to(device)

    with torch.no_grad():
        model.eval()
        features = model(batch_t)
    return features.cpu().numpy().flatten()

def process_folder(folder_info):
    folder_name, directory_path, device_id = folder_info
    device = torch.device(f"cuda:{device_id}")

    # Define model and transformations here
    model = models.resnet50(pretrained=True).to(device)
    model = torch.nn.Sequential(*(list(model.children())[:-1]))
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    folder_path = os.path.join(directory_path, folder_name)
    features_list = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            file_path = os.path.join(folder_path, filename)
            features = extract_cnn_features(file_path, model, transform, device)
            features_list.append(features)
    avg_features = np.mean(features_list, axis=0)
    return folder_name, avg_features

def main(directory_path):
    folder_names = [f for f in os.listdir(directory_path) if os.path.isdir(os.path.join(directory_path, f))]
    num_gpus = torch.cuda.device_count()
    print(f"Number of GPUs available: {num_gpus}")

    folder_infos = [(folder_name, directory_path, i % num_gpus) for i, folder_name in enumerate(folder_names)]

    mp.set_start_method('spawn', force=True)
    with Pool(processes=cpu_count()) as pool:
        results = pool.map(process_folder, folder_infos)

    all_features = [list(result[1]) for result in results]
    columns = ['ObjectID'] + [f'Feature{i}' for i in range(len(all_features[0]))]
    df = pd.DataFrame(all_features, columns=columns, index=[result[0] for result in results])
    df.to_csv('object_features.csv', index_label='ObjectID')
    print("Feature extraction complete. CSV file saved.")

if __name__ == "__main__":
    main('views')

