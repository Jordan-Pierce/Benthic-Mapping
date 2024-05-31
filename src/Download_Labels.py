import os
import shutil
import argparse
import traceback
from tqdm import tqdm
import concurrent.futures

import cv2
import numpy as np
import pandas as pd
import supervision as sv
from autodistill import helpers

import tator

from Auto_Distill import render_dataset


# ----------------------------------------------------------------------------------------------------------------------
# Functions
# ----------------------------------------------------------------------------------------------------------------------
def download_frame(api, media, frame_id, output_dir):
    """

    :param api:
    :param media:
    :param frame_id:
    :param output_dir:
    :return:
    """

    try:
        frame_path = f"{output_dir}/{str(media.id)}_{str(frame_id)}.jpg"

        if not os.path.exists(frame_path):
            # Get the frame from TATOR
            temp = api.get_frame(id=media.id, tile=f"{media.width}x{media.height}", frames=[int(frame_id)])
            # Move to download directory
            shutil.move(temp, frame_path)

    except Exception as e:
        raise Exception(f"ERROR: Could not get frame {frame_id} from {media.id}.\n{e}")

    return frame_id, frame_path


def download_frames(api, media, frames, output_dir):
    """

    :param api:
    :param media:
    :param frames:
    :param output_dir:
    :return:
    """
    frame_paths = []

    # Setup multithreading for downloading / saving frames
    with concurrent.futures.ThreadPoolExecutor(max_workers=11) as executor:
        future_to_frame = {executor.submit(download_frame,
                                           api,
                                           media,
                                           frame,
                                           output_dir): frame for frame in frames}

        # Get the results, store  for later
        for future in concurrent.futures.as_completed(future_to_frame):

            frame_idx = future_to_frame[future]

            try:
                # Get the results, add to list
                frame_path = future.result()
                frame_paths.append(frame_path[1])
                print(f"NOTE: Frame {frame_path[0]} downloaded to {frame_path[1]}")

            except Exception as e:
                print(f"ERROR: Could not download frame {frame_idx}.\n{e}")

    return frame_paths


def download_labels(api_token, project_id, search_string):
    """

    :param api_token:
    :param project_id:
    :param search_string:
    :return:
    """
    try:
        # Setting the api given the token, authentication
        token = api_token
        api = tator.get_api(host='https://cloud.tator.io', token=token)
        print(f"NOTE: Authentication successful for {api.whoami().username}")

    except Exception as e:
        raise Exception(f"ERROR: Could not authenticate with provided API Token\n{e}")

    # Project id containing media
    if not project_id:
        raise Exception(f"ERROR: Project ID provided is invalid; please check input")

    # List of media
    if not search_string:
        raise Exception(f"ERROR: Search string provided is invalid; please check input")

    # ------------------------------------------------
    # Directory setup
    # ------------------------------------------------

    # Class name for all objects
    class_name = "Fish"

    # Get the root data directory (Data); OCD
    root = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    root = root.replace("\\", "/")

    # Get the root data directory
    data_dir = f"{root}/Data/"

    # Raw frame location (from TATOR)
    temp_dir = f"{data_dir}/Temporary_Data"
    os.makedirs(temp_dir, exist_ok=True)

    # Training dataset location
    train_dir = f"{data_dir}/Training_Data"
    os.makedirs(train_dir, exist_ok=True)

    # Dataset location
    current_data_dir = f"{train_dir}/{class_name}"
    os.makedirs(current_data_dir, exist_ok=True)

    # ------------------------------------------------
    # Download labels
    # ------------------------------------------------
    try:
        # Get the video handler
        query = api.get_localization_list(project=project_id, encoded_search=search_string)
        print(f"NOTE: Found {len(query)} Localizations")

    except Exception as e:
        raise Exception(f"ERROR: Could not complete query")

    # ------------------------------------------------
    # Download data
    # ------------------------------------------------
    try:
        data = []

        for q in query:

            # Extract relevant data
            media = q.media
            frame = q.frame
            x = q.x
            y = q.y
            width = q.width
            height = q.height

            # Add manually
            label = class_name

            # Create a symbolic frame path here
            frame_name = f"{str(media)}_{str(frame)}.jpg"

            if frame == 0:
                continue

            # Add to list
            data.append([media, frame_name, frame, x, y, width, height, label])

        # Convert pandas
        data = pd.DataFrame(data, columns=['media', 'name', 'frame', 'x', 'y', 'width', 'height', 'label'])

        # Updated with media dimensions
        new_data = []
        frame_paths = []

        # Loop through by media
        for media_id in data['media'].unique():
            # Get the current media
            subset = data[data['media'] == media_id].copy()
            frames = subset['frame'].unique()

            # Grab the actual media from tator
            media = api.get_media(id=int(media_id))
            subset['media_width'] = media.width
            subset['media_height'] = media.height

            # Download all frames associated with current media
            frames = download_frames(api, media, frames, temp_dir)
            frame_paths.extend(frames)

            new_data.append(subset)

    except Exception as e:
        raise Exception(f"ERROR: Could not complete media download.\n{e}")

    # Updated dataframe
    data = pd.concat(new_data)

    # Mapping of image basename to image
    images = {}
    # Mapping of image basename to detection (annotations)
    detections = {}

    # Loop though based on the images
    for frame_path in tqdm(frame_paths):

        try:
            # Get the annotations for the image
            name = os.path.basename(frame_path)
            subset = data[data['name'] == name]

            # Grab the coordinates, do transformation
            x = subset['x'].values
            y = subset['y'].values
            w = subset['width'].values
            h = subset['height'].values
            width = subset['media_width'].values[0]
            height = subset['media_height'].values[0]

            # Transform to xyxy
            xmin = x * width
            ymin = y * height
            xmax = w * width + xmin
            ymax = h * height + ymin

            # Concat the values
            xyxy = np.column_stack([xmin, ymin, xmax, ymax])
            label = subset['label'].values
            class_id = np.zeros_like(label)

            # Create the detection object, map to image basename
            detection = sv.Detections(xyxy=xyxy, class_id=class_id)
            detections[name] = detection

            # Open the image, map
            images[name] = cv2.imread(frame_path)

        except Exception as e:
            print(e)

    # Create the dataset
    dataset = sv.DetectionDataset(classes=[class_name], images=images, annotations=detections)

    # Save the dataset
    dataset.as_yolo(current_data_dir + "/images",
                    current_data_dir + "/annotations",
                    min_image_area_percentage=0.01,
                    data_yaml_path=current_data_dir + "/data.yaml")

    # Display the filtered dataset
    render_dataset(dataset,
                   f"{current_data_dir}/rendered",
                   include_boxes=True)

    # Split into training / validation sets
    helpers.split_data(current_data_dir, record_confidence=False)


# ----------------------------------------------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------------------------------------------

def main():
    """

    """

    parser = argparse.ArgumentParser(description="Download labels from TATOR")

    parser.add_argument("--api_token", type=str,
                        default=os.getenv('TATOR_TOKEN'),
                        help="Tator API Token")

    parser.add_argument("--project_id", type=int, default=70,
                        help="Project ID for desired media")

    parser.add_argument("--search_string", type=str,
                        help="Encoded search string")

    args = parser.parse_args()

    try:
        download_labels(api_token=args.api_token,
                        project_id=args.project_id,
                        search_string=args.search_string)

        print("Done.")

    except Exception as e:
        print(f"ERROR: Could not finish process.\n{e}")
        print(traceback.format_exc())


if __name__ == "__main__":
    main()
