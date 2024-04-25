import os
import random
import shutil

def select_random_images(source_dir, dest_dir, num_images=2000):
    print("Listing all folders...")
    all_folders = [os.path.join(source_dir, f) for f in os.listdir(source_dir)]
    selected_images = {}

    print("Selecting images...")
    while len(selected_images) < num_images:
        folder = random.choice(all_folders)
        images = [os.path.join(folder, img) for img in os.listdir(folder) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if images:
            selected_image = random.choice(images)
            folder_name = os.path.basename(folder)
            image_count = 1
            while f"{folder_name}_{image_count}.png" in selected_images:
                image_count += 1
            selected_images[f"{folder_name}_{image_count}.png"] = selected_image
        if len(selected_images) % 100 == 0:
            print(f"Selected {len(selected_images)} images so far...")
        if len(selected_images) >= num_images:
            break

    print("Copying images...")
    os.makedirs(dest_dir, exist_ok=True)
    for count, (new_name, img_path) in enumerate(selected_images.items(), 1):
        dest_path = os.path.join(dest_dir, new_name)
        shutil.copy2(img_path, dest_path)
        if count % 10 == 0:
            print(f"Copied {count} images.")

    print(f"{len(selected_images)} images have been copied to {dest_dir}")

source_directory = 'views'  # Replace with your source directory path
destination_directory = 'destination'  # Replace with the destination directory path
select_random_images(source_directory, destination_directory)

