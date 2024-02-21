import os
import yaml

from autodistill_yolov8 import YOLOv8


# ----------------------------------------------------------------------------------------------------------------------
# Functions
# ----------------------------------------------------------------------------------------------------------------------

def create_training_yaml(yaml_files, output_dir):
    """

    :param yaml_files:
    :param output_dir:
    :return:
    """
    # Initialize variables to store combined data
    combined_data = {'names': [], 'nc': 0, 'train': [], 'val': []}

    try:
        # Iterate through each YAML file
        for yaml_file in yaml_files:
            with open(yaml_file, 'r') as file:
                data = yaml.safe_load(file)

                # If the class isn't already in the combined list
                if data['names'] not in combined_data['names']:
                    # Combine 'names' field
                    combined_data['names'].extend(data['names'])

                    # Combine 'nc' field
                    combined_data['nc'] += data['nc']

                # Combine 'train' and 'val' paths
                combined_data['train'].append(data['train'])
                combined_data['val'].append(data['val'])

        # Create a new YAML file with the combined data
        output_file_path = f"{output_dir}/training_data.yaml"

        with open(output_file_path, 'w') as output_file:
            yaml.dump(combined_data, output_file)

        # Check that it was written
        if os.path.exists(output_file_path):
            return output_file_path

    except Exception as e:
        raise Exception(f"ERROR: Could not output YAML file!\n{e}")


# ----------------------------------------------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------------------------------------------

# Get the root data directory (Data); OCD
root = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) + "\\Data"
root = root.replace("\\", "/")
assert os.path.exists(root)

# The root folder containing *all* post-processed dataset for training
training_data_dir = f"{root}/Training_Data"
assert os.path.exists(training_data_dir)

# CV Tasks
DETECTION = False
SEGMENTATION = True

# There can only be one
assert DETECTION != SEGMENTATION

# Number of training epochs
num_epochs = 25

# ----------------------
# Dataset Creation
# ----------------------

# Here we loop though all the datasets in the training_data_dir,
# get their image / label folders, and the data.yml file.
yaml_files = []

dataset_folders = os.listdir(training_data_dir)

for dataset_folder in dataset_folders:
    # Get the folder for the dataset
    dataset_folder = f"{training_data_dir}/{dataset_folder}"
    # Get the YAML file for the dataset
    yaml_file = f"{dataset_folder}/data.yaml"
    assert os.path.exists(yaml_file)
    # Add to the list
    yaml_files.append(yaml_file)

# Create a new temporary YAML file for the merged datasets
training_yaml = create_training_yaml(yaml_files, training_data_dir)

# Get weights based on task
if DETECTION:
    weights = "yolov8n.pt"
else:
    weights = "yolov8n-seg.pt"

# Train model
target_model = YOLOv8(weights)
target_model.train(training_yaml, epochs=num_epochs)

print("Done.")

