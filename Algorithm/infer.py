import os
import shutil
import traceback
import concurrent.futures

import numpy as np

import tator
import supervision as sv
from ultralytics import YOLO


# ----------------------------------------------------------------------------------------------------------------------
# Functions
# ----------------------------------------------------------------------------------------------------------------------
def download_frame(api, media, frame_idx, downloaded_frames_dir):
    """

    :param frame_idx:
    :param media:
    :param downloaded_frames_dir:
    :return:
    """
    try:
        # Get the frame from TATOR
        temp = api.get_frame(id=media.id, tile=f"{media.width}x{media.height}", frames=[frame_idx])
        # Move to download directory
        frame_path = f"{downloaded_frames_dir}/{str(frame_idx)}.png"
        shutil.move(temp, frame_path)

        return frame_idx, frame_path

    except Exception as e:
        raise Exception(f"ERROR: Could not get frame {frame_idx} from {media.id}.\n{e}")


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

        # Check if box1 is a large box containing at least two smaller boxes
        is_large_box[i] = contained_count >= 3

    annotations = annotations[~is_large_box]

    # Filter by confidence
    annotations = annotations[annotations.confidence > conf_thresh]

    return annotations


def algorithm(token, project_id, media_id, start_at, end_at, conf, iou):
    """

    :param token:
    :param project_id:
    :param media_id:
    :param start_at:
    :param end_at:
    :param conf:
    :param iou:
    :return:
    """
    try:
        # Setting the api given the token, authentication
        api = tator.get_api(host='https://cloud.tator.io', token=token)
        print(f"NOTE: Authentication successful for {api.whoami().username}")

    except Exception as e:
        raise Exception(f"ERROR: Could not authenticate with provided API Token\n{e}")

    # Project id containing media
    if not project_id:
        raise Exception(f"ERROR: Project ID provided is invalid; please check input")

    # List of media
    if not len(media_id):
        raise Exception(f"ERROR: Medias provided is invalid; please check input")

    # Get the root directory
    root = os.path.dirname(os.path.realpath(__file__))
    root = root.replace("\\", "/")

    # Get the root data directory
    data_dir = f"{root}/Data/"

    # Raw frame location (from TATOR)
    downloaded_frames_dir = f"{data_dir}/Downloaded_Frames"
    os.makedirs(downloaded_frames_dir, exist_ok=True)

    # Model location
    model_path = f"{root}/best.pth"

    if not os.path.exists(model_path):
        raise Exception(f"ERROR: Model weights not found in {root}!")

    try:
        # Load it up
        model = YOLO(model_path)
    except Exception as e:
        raise Exception(f"ERROR: Could not load model weights {model_path}.\n{e}")

    # ------------------------------------------------
    # Download media
    # ------------------------------------------------

    try:
        # Get the media, attributes
        media = api.get_media(media_id)
        width = media.width
        height = media.height

        # Ensure both values are divisible by 32
        height = height // 32 * 32
        width = width // 32 * 32

        # Image size (passed in model)
        imgsz = [height, width]

        # Get a list of frames to make predictions
        frames = np.arange(start_at, end_at + 1).tolist()

        if not frames:
            raise Exception("ERROR: No frames were extracted!")

        print(f"NOTE: Downloading {len(frames)} frames {start_at} through {end_at} for {media.name}...")

        # Setup multithreading for downloading / saving frames
        with concurrent.futures.ThreadPoolExecutor(max_workers=11) as executor:
            future_to_frame = {executor.submit(download_frame,
                                               api,
                                               media,
                                               frame_idx,
                                               downloaded_frames_dir): frame_idx for frame_idx in frames}

            # Get the results, store  for later
            for future in concurrent.futures.as_completed(future_to_frame):
                frame_idx = future_to_frame[future]
                try:
                    frame_path = future.result()
                    print(f"Frame {frame_path[0]} downloaded to {frame_path[1]}")
                except Exception as e:
                    print(f"Error downloading frame {frame_idx}: {e}")

    except Exception as e:
        raise Exception(f"ERROR: Could not get media from {media_id}.\n{e}")

    # -------------------------------------------
    # Make Inferences
    # -------------------------------------------

    try:
        # Generator for model performing inference
        results = model(f"{downloaded_frames_dir}/*.png",
                        conf=conf,
                        iou=iou,
                        imgsz=imgsz,
                        half=True,
                        augment=True,
                        max_det=2000,
                        verbose=False,
                        retina_masks=True,
                        stream=True)  # generator of Results objects

        # Loop through the results
        for result in results:

            # Original frame
            original_frame = result.orig_img
            original_paths = result.path

            # Convert the results
            detections = sv.Detections.from_ultralytics(result)

            # Filter the detections
            detections = filter_detections(original_frame, detections, 1.1)

            # Normalize the detections
            print("Pass")

    except Exception as e:
        pass

    if os.path.exists(downloaded_frames_dir) and False:
        shutil.rmtree(downloaded_frames_dir)


# ----------------------------------------------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------------------------------------------

def main():
    """

    """
    # Fill in the variables from env
    token = os.getenv("TOKEN")
    project_id = os.getenv("PROJECT_ID")
    media_id = os.getenv("MEDIA_ID")
    start_at = os.getenv("START_AT")
    end_at = os.getenv("END_AT")
    conf = os.getenv("CONFIDENCE")
    iou = os.getenv("IOU")

    # Do checks: if empty, exit early
    # pass

    try:
        algorithm(token=os.getenv('TATOR_TOKEN'),
                  project_id=155,
                  media_id="16406663",
                  start_at=0,
                  end_at=300,
                  conf=0.1,
                  iou=0.5)

        print("Done.")

    except Exception as e:
        print(f"ERROR: Failed to complete inference!\n{e}")
        print(traceback.format_exc())


if __name__ == "__main__":
    main()