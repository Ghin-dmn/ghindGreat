import os
import json

# Specify the input folder containing JSON files and the output folder for TXT files
input_folder = r'C:\Users\johna\Downloads\Sliced Fruits and Vegetables.v8i.yolov8 (1)\train\labels'
output_folder = r'C:\Users\johna\Downloads\Sliced Fruits and Vegetables.v8i.yolov8 (1)\resulta'

# Ensure the output folder exists
os.makedirs(output_folder, exist_ok=True)

# Loop through all files in the input folder
for file_name in os.listdir(input_folder):
    if file_name.endswith('.json'):  # Process only JSON files
        input_file_path = os.path.join(input_folder, file_name)
        output_file_path = os.path.join(output_folder, file_name.replace('.json', '.txt'))

        print(f"Processing file: {input_file_path}")  # Debug print

        # Read JSON content
        try:
            with open(input_file_path, 'r', encoding='utf-8') as json_file:
                data = json.load(json_file)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON in {file_name}: {e}")
            continue

        # Write the JSON content to a TXT file in a readable format
        with open(output_file_path, 'w', encoding='utf-8') as txt_file:
            txt_file.write(json.dumps(data, indent=4))

        print(f"Converted {file_name} to {output_file_path}")  # Debug print

print(f"Conversion complete! TXT files are saved in: {output_folder}")