import os
import glob
import shutil
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

import supervision as sv
from autodistill import helpers
from autodistill.detection import CaptionOntology
from autodistill_grounded_sam import GroundedSAM
from autodistill_grounding_dino import GroundingDINO
from autodistill_yolov8 import YOLOv8Base


# ----------------------------------------------------------------------------------------------------------------------
# Functions
# ----------------------------------------------------------------------------------------------------------------------


def extract_frames_from_video(video_path, image_dir, start_ratio=.15, end_ratio=.85, frame_stride=15):
    """

    :param video_path:
    :param image_dir:
    :param start_ratio:
    :param end_ratio:
    :param frame_stride:
    :return:
    """
    # Create a name pattern
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    image_name_pattern = video_name + "-{:05d}.png"

    # Get the video feed
    cap = cv2.VideoCapture(video_path)

    # Figure out the start and end points
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start_at = int(start_ratio * total_frames)
    end_at = int(end_ratio * total_frames)

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


def extract_frames(video_paths, image_dir, start_ratio=.15, end_ratio=.85, frame_stride=15):
    """

    :param video_paths:
    :param image_dir:
    :param start_ratio:
    :param end_ratio:
    :param frame_stride:
    :return:
    """
    with ThreadPoolExecutor() as executor:
        # Use executor to process each video in parallel
        executor.map(
            lambda video_path: extract_frames_from_video(video_path, image_dir, start_ratio, end_ratio, frame_stride),
            video_paths
        )


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


def filter_detections(image, annotations, area_thresh=1.0, conf_thresh=0.0):
    """

    :param image:
    :param annotations:
    :param area_thresh:
    :param conf_thresh:
    :return annotations:
    """

    height, width, channels = image.shape
    image_area = height * width

    # Filter by relative area first
    annotations = annotations[(annotations.box_area / image_area) <= area_thresh]

    # Get the box areas
    boxes = annotations.xyxy
    num_boxes = len(boxes)

    # Filter out larger boxes that contain smaller boxes
    is_large_box = np.zeros(num_boxes, dtype=bool)

    for i, box1 in enumerate(boxes):
        x1, y1, x2, y2 = box1

        # Count the number of smaller boxes contained within box1
        contained_count = np.sum(
            (boxes[:, 0] >= x1) &
            (boxes[:, 1] >= y1) &
            (boxes[:, 2] <= x2) &
            (boxes[:, 3] <= y2)
        )

        # Check if box1 is a large box containing at least N smaller boxes
        is_large_box[i] = contained_count >= 5

    annotations = annotations[~is_large_box]

    # Filter by confidence
    annotations = annotations[annotations.confidence > conf_thresh]

    return annotations


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


def remove_bad_data(data_dir):
    """

    :param data_dir:
    :return:
    """
    # Set the paths
    train_dir = f"{data_dir}/train"
    valid_dir = f"{data_dir}/valid"
    render_dir = f"{data_dir}/rendered"

    # Make sure they exist
    assert os.path.exists(train_dir)
    assert os.path.exists(valid_dir)
    assert os.path.exists(render_dir)

    combined_dict = {}

    # Get all the training images and labels paths
    train_images = glob.glob(f"{train_dir}/images/*.jpg")
    train_labels = glob.glob(f"{train_dir}/labels/*.txt")

    for image, label in zip(train_images, train_labels):
        basename = os.path.basename(image).split(".")[0]
        combined_dict[basename] = {
            "image": image,
            "label": label
        }

    # Get all the validation images and labels paths
    valid_images = glob.glob(f"{valid_dir}/images/*.jpg")
    valid_labels = glob.glob(f"{valid_dir}/labels/*.txt")

    for image, label in zip(valid_images, valid_labels):
        basename = os.path.basename(image).split(".")[0]
        combined_dict[basename] = {
            "image": image,
            "label": label
        }

    # Get the rendered images
    render_images = glob.glob(f"{render_dir}/*.png")

    # Loop through the rendered images and removes those that
    # exist from the combined dictionary.
    for render_image in render_images:
        basename = os.path.basename(render_image).split(".")[0]
        if basename in combined_dict:
            combined_dict.pop(basename)

    # Finally, loop though the remaining image / labels,
    # representing the bad data, and delete them.
    for value in combined_dict.values():
        print(f"NOTE: Deleting {value['image']}")
        os.remove(value['image'])
        print(f"NOTE: Deleting {value['label']}")
        os.remove(value['label'])


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

    # If it exists from last time (exited early) delete
    if os.path.exists(auto_labeled_dir):
        shutil.rmtree(auto_labeled_dir)

    # Create the directory
    os.makedirs(auto_labeled_dir, exist_ok=True)

    # The root folder containing *all* post-processed dataset for training
    training_data_dir = f"{root}/Training_Data"
    os.makedirs(training_data_dir, exist_ok=True)

    # ------------------------------------------------------
    # UPDATE THIS AND ONLY THIS

    # Currently we're creating single-class datasets, and
    # merging them together right before training the model
    dataset_name = "Rock2"

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

    # Debug
    SAVE_LABELS = True

    # CV Tasks
    DETECTION = False
    SEGMENTATION = True

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
        "pebble": "rock",
        "cobble": "rock",
        "stone": "rock",
        "boulder": "rock",
    })

    # Polygon's size as a ratio of the image
    # Large polygons shouldn't be included.
    area_thresh = 0.9

    # Non-maximum suppression threshold
    nms_thresh = 0.5

    # Extract every N frames
    frame_stride = 15

    # ---------------------------------------
    # Workflow
    # ---------------------------------------

    if EXTRACT_FRAMES:
        # Get converted video paths
        video_paths = glob.glob(f"{converted_video_dir}/*.mp4")
        print("Converted Videos Found: ", len(video_paths))
        # Extract frames from training video (if needed)
        extract_frames(video_paths, extracted_frames_dir, frame_stride=frame_stride)

        # -----------------------------------------
        # Manually delete any images as needed!
        # -----------------------------------------
        response = input(f"Delete any bad frames from {os.path.basename(extracted_frames_dir)} now...")

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
            include_boxes = True
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

                # Filter based on area and confidence (removes large and unconfident)
                annotations = filter_detections(image, annotations, area_thresh)

                # Filter boxes with NMS (removes all the duplicates, faster than with_nms)
                predictions = np.column_stack((annotations.xyxy, annotations.confidence))
                indices = sv.detection.utils.box_non_max_suppression(predictions, nms_thresh)
                annotations = annotations[indices]

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

            # -----------------------------------------
            # Manually delete any images as needed!
            # -----------------------------------------
            response = input(f"Delete any bad labeled frames from {os.path.basename(current_data_dir)} now...")
            # Remove images and labels from train/valid if they were deleted from rendered
            remove_bad_data(current_data_dir)

    print("Done.")
