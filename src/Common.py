import os
import yaml
import datetime
from typing import List
from tqdm.auto import tqdm

import numpy as np

import supervision as sv


# ----------------------------------------------------------------------------------------------------------------------
# Functions
# ----------------------------------------------------------------------------------------------------------------------

def get_now():
    """
    Returns a datetime string formatted according to the current time.

    :return:
    """
    # Get the current datetime
    now = datetime.datetime.now()
    now = now.strftime("%Y-%m-%d_%H-%M-%S")

    return now


def render_dataset(dataset, output_dir, include_boxes=True, include_masks=True):
    """
    Render the dataset with annotations and save the images.

    :param dataset: The dataset to render
    :param output_dir: Directory to save rendered images
    :param include_boxes: Whether to include bounding boxes in the rendering
    :param include_masks: Whether to include masks in the rendering
    :return: None
    """
    # Images
    image_names = list(dataset.images.keys())

    # Create the annotation objects
    mask_annotator = sv.MaskAnnotator()
    box_annotator = sv.BoundingBoxAnnotator()
    labeler = sv.LabelAnnotator()

    with sv.ImageSink(target_dir_path=output_dir, overwrite=False) as sink:
        for image_name in tqdm(image_names, desc="Rendering Frames"):
            # Get the images and annotation
            image = dataset.images[image_name].astype(np.uint8)
            annotations = dataset.annotations[image_name]
            labels = annotations.tracker_id

            if include_boxes:
                # Get the boxes for the annotations
                image = box_annotator.annotate(scene=image, detections=annotations)

            if include_masks:
                # Get the masks for the annotations
                image = mask_annotator.annotate(scene=image, detections=annotations)

            labeler.annotate(scene=image, detections=annotations, labels=labels)

            output_file = f"{os.path.basename(image_name)}"
            sink.save_image(image=image, image_name=output_file)


def create_training_yaml(yaml_files: List[str], output_dir: str):
    """
    Combines multiple YAML files into one at the time of training.

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