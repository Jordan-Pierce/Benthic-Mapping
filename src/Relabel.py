import os
import glob
import shutil
from tqdm import tqdm

import numpy as np

import supervision as sv
from ultralytics import YOLO
from autodistill import helpers

from Auto_Distill import extract_frames
from Auto_Distill import render_dataset
from Auto_Distill import remove_bad_data
from Auto_Distill import batch_and_copy_images


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

    # The root folder containing *all* post-processed dataset for training
    training_data_dir = f"{root}/Training_Data"
    os.makedirs(training_data_dir, exist_ok=True)

    # Provide the weights to the previously best trained YOLO model
    source_weights = "B:\\Benthic-Mapping\\Data\\Runs\\2024-05-31_18-39-21_detect_yolov8s\\weights\\best.pt"

    # ------------------------------------------------------
    # UPDATE THIS AND ONLY THIS

    # Currently we're creating single-class datasets, and
    # merging them together right before training the model
    dataset_name = "Sponge_Relabeled"

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

    if DETECTION:
        # For rendering
        include_boxes = True
        include_masks = False

    else:
        # For rendering
        include_boxes = True
        include_masks = True

    # Non-maximum suppression threshold
    conf = 0.30
    iou = 0.1

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
        batch_and_copy_images(temporary_frames_dir, extracted_frames_dir, batch_size=500)
        temporary_image_folders = glob.glob(f"{temporary_frames_dir}/images_*")
        print("Batch Folders Found: ", len(temporary_image_folders))

        # Initialize the trained model
        yolo_model = YOLO(source_weights)

        # Loop through the temp folders of images
        for temporary_image_folder in temporary_image_folders:

            # Create labels for the images in temp folder
            results = yolo_model.predict(f"{temporary_image_folder}/*.*",
                                         conf=conf,
                                         iou=iou,
                                         # imgsz=1280,
                                         show=False,
                                         show_labels=False,
                                         stream=True)

            # Convert Ultralytics results to Supervision dataset
            classes = {}
            images = {}
            annotations = {}

            for result in results:
                detection = sv.Detections.from_ultralytics(result)
                classes.update(result.names)
                images[result.path] = result.orig_img
                annotations[result.path] = detection

            classes = list(classes.values())
            dataset = sv.DetectionDataset(classes, images, annotations)

            # Filter the dataset
            image_names = list(dataset.images.keys())

            for image_name in tqdm(image_names):
                # numpy arrays for this image
                image = dataset.images[image_name]
                annotations = dataset.annotations[image_name]
                class_id = dataset.annotations[image_name].class_id

                # Filter boxes with NMS (removes all the duplicates, faster than with_nms)
                predictions = np.column_stack((annotations.xyxy, annotations.confidence))
                indices = sv.detection.utils.box_non_max_suppression(predictions, iou)
                annotations = annotations[indices]

                # Update the annotations and class IDs in dataset
                dataset.annotations[image_name] = annotations
                dataset.annotations[image_name].class_id = np.zeros_like(class_id)

            # Change the dataset classes
            dataset.classes = classes

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
        # remove_bad_data(current_data_dir)

    # Delete the temporary copies
    shutil.rmtree(temporary_frames_dir)

    print("Done.")
