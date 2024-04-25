import os
import matplotlib.pyplot as plt

def label_images(base_folder):
    labels = {}
    object_folders = [os.path.join(base_folder, o) for o in os.listdir(base_folder) 
                      if os.path.isdir(os.path.join(base_folder,o))]

    for object_folder in object_folders:
        object_id = os.path.basename(object_folder)
        images = [os.path.join(object_folder, img) for img in os.listdir(object_folder) 
                  if img.endswith('.png') or img.endswith('.jpg')]  # adjust the extension as needed

        if images:
            # Display the first image for each object
            first_image_path = images[0]
            img = plt.imread(first_image_path)
            plt.imshow(img)
            plt.show()

            # Label for the entire object
            label = input(f"Enter label for object {object_id} (1 for good, 0 for bad): ")
            labels[object_id] = label
        else:
            print(f"No images found in {object_folder}")

    return labels

views_directory = "views"
labeled_data = label_images(views_directory)

# Save labels to a file
with open('GoodBadlabels.csv', 'w') as f:
    for object_id, label in labeled_data.items():
        f.write(f"{object_id},{label}\n")

