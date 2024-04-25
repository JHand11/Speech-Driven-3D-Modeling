import os
import pandas as pd
import torch
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from clip import load
from multiprocessing import Pool, set_start_method
import tqdm

def extract_features(data):
    folder_path, device = data
    model, preprocess = load('ViT-B/32', device=device)

    try:
        image_name = sorted(os.listdir(folder_path))[0]
        image_path = os.path.join(folder_path, image_name)
        image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
        with torch.no_grad():
            features = model.encode_image(image)
        return os.path.basename(folder_path), features.cpu().numpy()
    except Exception as e:
        print(f"Error processing folder {folder_path}: {e}")
        return os.path.basename(folder_path), None

def main():
    set_start_method('spawn', force=True)
    views_folder = 'views'
    folder_paths = [os.path.join(views_folder, f) for f in os.listdir(views_folder)]
    print("loaded folders")

    num_gpus = 2
    num_workers = int(os.cpu_count() / 2) or 1

    data = [(path, f'cuda:{i % num_gpus}') for i, path in enumerate(folder_paths)]
    print("loaded data")

    results = []
    with Pool(num_workers) as p:
        for result in tqdm.tqdm(p.imap_unordered(extract_features, data), total=len(data)):
            results.append(result)

    df = pd.DataFrame([r for r in results if r[1] is not None], columns=['object_id', 'features'])
    df.to_csv('object_features.csv', index=False)
    print("Script completed successfully.")

if __name__ == '__main__':
    main()

