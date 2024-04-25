import os
import json

# Directory containing your .png files
directory_path = r'/media/vislab-002/SP2 4TB/objaverse-rendering-main/grids'

# Initialize an empty dictionary to hold your data
data = {}
count = 0
# Loop through each file in the directory
for filename in os.listdir(directory_path):
    print(count)
    count = count+1
    if filename.endswith('.png'):
        # Use the filename (without extension) as the caption
        caption = os.path.splitext(filename)[0]
        
        # Add the data to the dictionary
        data[caption] = {
            "caption": caption,
            "train_resolution": [1024, 1024]
        }

# Specify the filename of your JSON file
json_filename = r'/media/vislab-002/SP2 4TB/objaverse-rendering-main/meta_lat2.json'

# Write the dictionary to a file in JSON format
with open(json_filename, 'w') as json_file:
    json.dump(data, json_file, indent=2)

print(f"JSON file '{json_filename}' has been created with {len(data)} entries.")
