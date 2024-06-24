import os
import yaml
import datetime

from ultralytics import YOLO

from Auto_Distill import remove_bad_data


# ----------------------------------------------------------------------------------------------------------------------
# Functions
# ----------------------------------------------------------------------------------------------------------------------
def get_now():
    """

    :return:
    """
    # Get the current datetime
    now = datetime.datetime.now()
    now = now.strftime("%Y-%m-%d_%H-%M-%S")

    return now


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
                if not any([c in combined_data['names'] for c in data['names']]):
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
if __name__ == '__main__':

    # Get the root data directory (Data); OCD
    root = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) + "\\Data"
    root = root.replace("\\", "/")
    assert os.path.exists(root)

    # The root folder containing *all* post-processed dataset for training
    training_data_dir = f"{root}/Training_Data"
    assert os.path.exists(training_data_dir)

    # Where to place the output model run
    run_dir = f"{root}/Runs"
    os.makedirs(run_dir, exist_ok=True)

    # CV Tasks
    DETECTION = True
    SEGMENTATION = False

    # There can only be one
    assert DETECTION != SEGMENTATION

    if DETECTION:
        task = "detect"
    else:
        task = "segment"

    # Number of training epochs
    num_epochs = 35

    # ----------------------
    # Dataset Creation
    # ----------------------

    # Here we loop though all the datasets in the training_data_dir,
    # get their image / label folders, and the data.yml file.
    yaml_files = []

    dataset_folders = os.listdir(training_data_dir)

    for dataset_folder in dataset_folders:

        # Get the full path of the folder
        full_path = os.path.join(training_data_dir, dataset_folder)

        if not os.path.isdir(full_path):
            continue

        # Check to see if you want to include this dataset
        user_input = input(f"Include? (Y/n) - {os.path.basename(dataset_folder)}: ")

        if user_input.lower() == 'n':
            continue

        # Remove images and labels from train/valid if they were deleted from rendered
        # remove_bad_data(full_path)

        # Get the YAML file for the dataset
        yaml_file = os.path.join(full_path, 'data.yaml')

        if not os.path.exists(yaml_file):
            print(f"Warning: {yaml_file} does not exist.")
            continue

        # Add to the list
        yaml_files.append(yaml_file)

    # Create a new temporary YAML file for the merged datasets
    training_yaml = create_training_yaml(yaml_files, training_data_dir)

    # Get weights based on task
    if DETECTION:
        weights = "yolov10m.pt"
    else:
        weights = "yolov10m-seg.pt"

    # Name of the run
    run_name = f"{get_now()}_{task}_{weights.split('.')[0]}"

    # Access pre-trained model
    target_model = YOLO(weights)

    # Train model w/ parameters
    results = target_model.train(data=training_yaml,
                                 cache=False,
                                 device=0,
                                 half=True,
                                 epochs=num_epochs,
                                 patience=10,
                                 batch=8,
                                 project=run_dir,
                                 name=run_name,
                                 save_period=5,
                                 plots=True,
                                 single_cls=True)
    print("Done.")
