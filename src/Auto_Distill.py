import os
import glob
import yaml
import shutil
from tqdm import tqdm

import cv2
import numpy as np

import supervision as sv
from autodistill import helpers
from autodistill_yolov8 import YOLOv8
from autodistill.detection import CaptionOntology
from autodistill_grounded_sam import GroundedSAM
from autodistill_grounding_dino import GroundingDINO


# ----------------------------------------------------------------------------------------------------------------------
# Functions
# ----------------------------------------------------------------------------------------------------------------------


def extract_frames(video_paths, image_dir, start_at=1500, end_at=4500, frame_stride=30):
    """

    :param video_paths:
    :param image_dir:
    :param start_at:
    :param end_at:
    :param frame_stride:
    :return:
    """
    # Loop through each video
    for video_path in video_paths:
        # Create a name pattern
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        image_name_pattern = video_name + "-{:05d}.png"

        # Get the video feed
        cap = cv2.VideoCapture(video_path)

        # Create output directory
        os.makedirs(image_dir, exist_ok=True)

        # Video is opened, frame extracted
        success, image = cap.read()
        frame_number = 0

        if not success:
            raise Exception("Could not open video!")

        while success:
            # Every Nth frame, that is between N and N
            if frame_number % frame_stride == 0 and start_at <= frame_number <= end_at:
                # Write the image to disk if it doesn't already exist
                frame_filename = os.path.join(image_dir, image_name_pattern.format(frame_number))
                if not os.path.exists(frame_filename):
                    cv2.imwrite(frame_filename, image)

            # Continue with the video feed
            success, image = cap.read()
            frame_number += 1

        # Release me!
        cap.release()


def batch_and_copy_images(root, source_folder, batch_size=100):
    """

    :param source_folder:
    :param batch_size:
    :return:
    """
    # Ensure the source folder exists
    if not os.path.exists(source_folder):
        print(f"Source folder '{source_folder}' not found.")
        return

    # Create a list of image files in the source folder
    image_files = glob.glob(os.path.join(source_folder, '*.png')) + glob.glob(os.path.join(source_folder, '*.jpg'))

    # Check if there are any image files
    if not image_files:
        print(f"No image files found in '{source_folder}'.")
        return

    # Create output folders
    output_folder_base = f"{root}/images"
    output_folder_index = 1

    # Iterate over image files and copy them into batches
    for index, image_file in enumerate(image_files):
        # Check if a new batch needs to be created
        if index % batch_size == 0:
            current_output_folder = f"{output_folder_base}_{output_folder_index}"
            os.makedirs(current_output_folder, exist_ok=True)
            output_folder_index += 1

        # Create the path for the copy in the current output folder
        copy_path = os.path.join(current_output_folder, os.path.basename(image_file))

        # Copy the image file to the current output folder
        shutil.copy2(image_file, copy_path)

    print(f"Copies of images successfully created in batches of {batch_size}.")


def filter_detections(image, annotations, area_threshold=0.45):
    """

    :param image:
    :param annotations:
    :param area_threshold:
    :return indices:
    """
    # Calculate the area of the image
    image_area = image.shape[0] * image.shape[1]

    # Calculate the area of each bounding box
    bounding_box_areas = annotations.box_area

    # Calculate the ratio of bounding box area to image area
    area_ratio = bounding_box_areas / image_area

    # Filter bounding boxes where the area ratio is less than threshold
    filtered_indices = np.where(area_ratio < area_threshold)[0]

    if not len(filtered_indices):
        filtered_indices = np.arange(0, len(annotations))

    return filtered_indices


def render_dataset(dataset, output_dir, include_boxes=True, include_masks=False):
    """

    :param dataset:
    :param output_dir:
    :param include_boxes:
    :param include_masks:
    :return:
    """
    # Images
    image_names = list(dataset.images.keys())

    # Create the annotation object
    mask_annotator = sv.MaskAnnotator()
    box_annotator = sv.BoundingBoxAnnotator()

    with sv.ImageSink(target_dir_path=output_dir, overwrite=False) as sink:
        for i_idx, image_name in enumerate(image_names):

            # Get the images and annotation
            image = dataset.images[image_name]
            annotations = dataset.annotations[image_name]

            if include_boxes:
                # Get the boxes for the annotations
                image = box_annotator.annotate(scene=image, detections=annotations)

            if include_masks:
                # Get the masks for the annotations
                image = mask_annotator.annotate(scene=image, detections=annotations)

            output_file = f"{os.path.basename(image_name)}"
            sink.save_image(image=image, image_name=output_file)


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

if __name__ == "__main__":

    # Get the root data directory (Data); OCD
    root = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) + "\\Data"
    root = root.replace("\\", "/")

    # Converted videos from TATOR get placed here
    converted_video_dir = f"{root}/Converted_Videos"
    os.makedirs(converted_video_dir, exist_ok=True)

    # Extracted frames from Converted videos go here
    extracted_frames_dir = f"{root}/Extracted_Frames"
    os.makedirs(extracted_frames_dir, exist_ok=True)

    # Frames are batched (RAM) and temporarily placed here
    temporary_frames_dir = f"{root}/Temporary_Frames"

    # If it exists from last time (exited early) delete
    if os.path.exists(temporary_frames_dir):
        shutil.rmtree(temporary_frames_dir)

    # Create the directory
    os.makedirs(temporary_frames_dir, exist_ok=True)

    # Auto labeled data; this is also temporary until being filtered
    auto_labeled_dir = f"{root}/Auto_Labeled"
    os.makedirs(auto_labeled_dir, exist_ok=True)

    # The root folder containing *all* post-processed dataset for training
    training_data_dir = f"{root}/Training_Data"
    os.makedirs(training_data_dir, exist_ok=True)

    # ------------------------------------------------------
    # UPDATE THIS AND ONLY THIS

    # Currently we're creating single-class datasets, and
    # merging them together right before training the model
    dataset_name = "Rock"

    # The directory for the current dataset being created
    current_data_dir = f"{training_data_dir}/{dataset_name}"
    os.makedirs(current_data_dir, exist_ok=True)

    # Directory contains visuals of images and labels (for QA/QC)
    rendered_data_dir = f"{current_data_dir}/rendered"
    # ------------------------------------------------------
    # Modify each of these as needed!

    # Define the workflow
    EXTRACT_FRAMES = False
    CREATE_LABELS = True
    TRAIN_MODEL = True

    # Debug
    SAVE_LABELS = True

    # CV Tasks
    DETECTION = True
    SEGMENTATION = False

    # There can only be one
    assert DETECTION != SEGMENTATION

    # Set up the labeling ontology
    ontology = CaptionOntology({
        "rock": "rock",
        "tiny rock": "rock",
        "small rock": "rock",
        "big rock": "rock",
        "fuzzy rock": "rock",
        "smooth rock": "rock",
    })

    # Polygon's size as a ratio of the image
    # Large polygons shouldn't be included...
    area_threshold = 0.4

    # ---------------------------------------
    # Workflow
    # ---------------------------------------

    if EXTRACT_FRAMES:
        # Get converted video paths
        video_paths = glob.glob(f"{converted_video_dir}/*.mp4")
        print("Converted Videos Found: ", len(video_paths))
        # Extract frames from training video (if needed)
        extract_frames(video_paths, extracted_frames_dir, frame_stride=200)

        # -----------------------------------------
        # Manually delete any images if needed!
        # (use a debugger breakpoint to pause)
        # -----------------------------------------

        # Get a count of the images extracted from all videos
        image_paths = sv.list_files_with_extensions(directory=extracted_frames_dir, extensions=["png", "jpg", "jpg"])
        print("Extracted Images Found: ", len(image_paths))

    if CREATE_LABELS:
        # Make copies of the extracted frames, make in batches of N
        # This has to be done because the auto labeler is RAM heavy
        batch_and_copy_images(temporary_frames_dir, extracted_frames_dir)
        temporary_image_folders = glob.glob(f"{temporary_frames_dir}/images_*")
        print("Batch Folders Found: ", len(temporary_image_folders))

        if DETECTION:
            # Initialize the foundational base model, set the thresholds
            base_model = GroundingDINO(ontology=ontology,
                                       box_threshold=0.05,
                                       text_threshold=0.05)
            # For rendering
            include_boxes = True
            include_masks = False

        else:
            # Initialize the foundational base model, set the thresholds
            base_model = GroundedSAM(ontology=ontology,
                                     box_threshold=0.05,
                                     text_threshold=0.05)
            # For rendering
            include_boxes = False
            include_masks = True

        # Loop through the temp folders of images
        for temporary_image_folder in temporary_image_folders:

            # Create labels for the images in temp folder
            dataset = base_model.label(input_folder=temporary_image_folder,
                                       extension=".png",
                                       output_folder=auto_labeled_dir,
                                       record_confidence=True)

            # Delete the temporary copies
            shutil.rmtree(temporary_image_folder)

            # Filter the dataset
            image_names = list(dataset.images.keys())

            for image_name in tqdm(image_names):
                # numpy arrays for this image
                image = dataset.images[image_name]
                annotations = dataset.annotations[image_name]
                class_id = dataset.annotations[image_name].class_id
                # Filter based on area and confidence
                indices = filter_detections(image, annotations)
                annotations = annotations[indices]

                if DETECTION:
                    # Filter based on NMS; This is slow for SAM / Masks
                    annotations = annotations.with_nms(threshold=0.075)

                # Update the annotations and class IDs in dataset
                dataset.annotations[image_name] = annotations
                dataset.annotations[image_name].class_id = np.zeros_like(class_id)

            # Change the dataset classes
            dataset.classes = [f'{dataset_name}']

            if SAVE_LABELS:
                # Save the filtered dataset (this is used for training)
                dataset.as_yolo(current_data_dir + "/images",
                                current_data_dir + "/annotations",
                                min_image_area_percentage=0.01,
                                data_yaml_path=current_data_dir + "/data.yaml")

            # Display the filtered dataset
            render_dataset(dataset,
                           rendered_data_dir,
                           include_boxes=include_boxes,
                           include_masks=include_masks)

        if SAVE_LABELS:
            # Split the filtered dataset into training / valid
            helpers.split_data(current_data_dir, record_confidence=True)

    # TODO move training to a separate script
    if TRAIN_MODEL:

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

        if DETECTION:
            weights = "yolov8n.pt"
        else:
            weights = "yolov8n-seg.pt"

        target_model = YOLOv8(weights)
        target_model.train(training_yaml, epochs=10)
