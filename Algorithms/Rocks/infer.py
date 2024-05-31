import os
import glob
import shutil
import traceback
import concurrent.futures

import cv2
import numpy as np

import tator

import torch
import supervision as sv

from ultralytics import SAM
from ultralytics import YOLO

from sahi import AutoDetectionModel
from sahi.utils.cv import read_image
from sahi.predict import get_sliced_prediction


# ----------------------------------------------------------------------------------------------------------------------
# Functions
# ----------------------------------------------------------------------------------------------------------------------
def download_frame(api, media, frame_idx, downloaded_frames_dir):
    """

    :param api:
    :param media:
    :param frame_idx:
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


def mask_to_polygons(masks):
    """

    :param masks:
    :return:
    """
    # Get the contours for each of the masks
    polygons = []

    for mask in masks:

        try:
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # Find the largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            epsilon_length = 0.0025 * cv2.arcLength(largest_contour, True)
            largest_contour = cv2.approxPolyDP(largest_contour, epsilon_length, True)
            # Convert the contour to a numpy array and append to the list
            polygons.append(largest_contour.squeeze())

        except Exception as e:
            pass

    return polygons


def polygons_to_points(polygons, image):
    """

    :param polygons:
    :param image:
    :return:
    """
    normalized_polygons = []

    for polygon in polygons:
        # Extract x and y coordinates of the polygon points
        polygon_x = polygon[:, 0]
        polygon_y = polygon[:, 1]

        # Normalize x and y coordinates
        normalized_polygon_x = polygon_x / image.shape[1]
        normalized_polygon_y = polygon_y / image.shape[0]

        # Create a new list with normalized coordinates for the current polygon
        normalized_points = np.column_stack((normalized_polygon_x, normalized_polygon_y)).tolist()
        normalized_polygons.append(normalized_points)

    return normalized_polygons


def notify_user(api, project_id, media_id, start_at, end_at):
    """

    :param api:
    :param project_id:
    :param media_id:
    :param start_at:
    :param end_at:
    :return:
    """
    try:
        # Get the url to the start frame
        url = f"https://cloud.tator.io/{project_id}/annotation/{media_id}?frame={start_at}&version={545}"

        # Access user's email through api
        user_email = api.whoami().email
        subject = "Your images have been annotated!"
        text = f"Finished Running YOLOv8 on frames {start_at} - {end_at} for Media {media_id}!\n\n\n{url}"

        # Make email spec
        email_spec = tator.models.EmailSpec(recipients=[user_email],
                                            subject=subject,
                                            text=text)

        # Send it
        api.send_email(project=project_id, email_spec=email_spec)
        print("NOTE: Notified user")

    except Exception:
        # Fail silently
        pass


def algorithm(token, project_id, media_id, start_at, end_at, conf=.5, iou=.7, smol=True, debug=False):
    """

    :param token:
    :param project_id:
    :param media_id:
    :param start_at:
    :param end_at:
    :param conf:
    :param iou:
    :param smol:
    :param debug:
    :return:
    """
    # ------------------------------------------------
    # TATOR setup
    # ------------------------------------------------

    try:
        # Setting the api given the token, authentication
        api = tator.get_api(host='https://cloud.tator.io', token=token)
        print(f"NOTE: Authentication successful for {api.whoami().username}")

    except Exception as e:
        raise Exception(f"ERROR: Could not authenticate with provided API Token\n{e}")

    try:
        # Find the localization type for this project.
        loc_types = api.get_localization_type_list(project_id)
        found_loc = False

        for loc_type in loc_types:
            if loc_type.id == 460 and loc_type.dtype == 'poly':
                found_loc = True
                break

        if not found_loc:
            raise Exception("ERROR: Localization '460' not found.")

    except Exception as e:
        raise Exception(f"ERROR: Could not get correct localization type.\n{e}")

    try:
        # Find the version type for this project.
        version_types = api.get_version_list(project_id)
        found_version = False

        for version_type in version_types:
            if version_type.id == 545:
                found_version = True
                break

        if not found_version:
            raise Exception("ERROR: Version '545' not found.")

    except Exception as e:
        raise Exception(f"ERROR: Could not get correct version type.\n{e}")

    # ------------------------------------------------
    # Directory setup
    # ------------------------------------------------

    # Get the root directory
    root = os.path.dirname(os.path.realpath(__file__))
    root = root.replace("\\", "/")

    # Get the root data directory
    data_dir = f"{root}/Data/"

    # Raw frame location (from TATOR)
    downloaded_frames_dir = f"{data_dir}/Downloaded_Frames"
    os.makedirs(downloaded_frames_dir, exist_ok=True)

    # Rendered frames with detections
    rendered_frames_dir = f"{data_dir}/Rendered_Frames"
    os.makedirs(rendered_frames_dir, exist_ok=True)

    # Check for CUDA
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # Model location
    model_path = f"best.pt"

    if not os.path.exists(model_path):
        raise Exception(f"ERROR: Model weights not found in {root}!")

    try:
        # If using SAHI, initialize model differently
        if smol:
            # Load the YOLO model as auto-detection
            yolo_model = AutoDetectionModel.from_pretrained(
                model_type='yolov8',
                model_path=model_path,
                confidence_threshold=conf,
                device=device
            )
            # Load the SAM model
            sam_model = SAM('sam_l.pt')

        else:
            # Normal YOLO model convention
            yolo_model = YOLO(model_path)

        print(f"NOTE: Successfully loaded weights {model_path}")

        # Create the annotators for segmentation
        mask_annotator = sv.MaskAnnotator()
        box_annotator = sv.BoundingBoxAnnotator()

    except Exception as e:
        raise Exception(f"ERROR: Could not load model weights {model_path}.\n{e}")

    # ------------------------------------------------
    # Download media
    # ------------------------------------------------

    try:
        # Get the media, attributes
        media = api.get_media(media_id)

        if not 0 < start_at < media.num_frames or start_at > end_at:
            raise Exception(f"ERROR: Start frame is invalid")

        if not 0 < end_at < media.num_frames or end_at < start_at:
            raise Exception(f"ERROR: End frame is invalid")

        # Get a list of frames to make predictions
        frames = np.arange(start_at, end_at + 1).tolist()

        if not frames:
            raise Exception("ERROR: No frames were extracted!")

        print(f"NOTE: Downloading {len(frames)} frames, "
              f"starting at {start_at} through {end_at} for {media.name}...")

        frame_paths = []

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
                    # Get the results, add to list
                    frame_path = future.result()
                    frame_paths.append(frame_path[1])
                    print(f"NOTE: Frame {frame_path[0]} downloaded to {frame_path[1]}")

                except Exception as e:
                    raise Exception(f"ERROR: Could not download frame {frame_idx}.\n{e}")

    except Exception as e:
        raise Exception(f"ERROR: Could not complete media download for {str(media_id)}.\n{e}")

    # -------------------------------------------
    # Make Inferences
    # -------------------------------------------

    try:
        # To store normalized predictions
        localizations = []
        print("NOTE: Performing inference")

        # Loop through the results
        with sv.ImageSink(target_dir_path=rendered_frames_dir, overwrite=True) as sink:
            for frame_path in frame_paths:

                # Frame metadata
                frame_name = os.path.basename(frame_path)
                frame_id = frame_name.split("_")[-1].split(".")[0]
                original_frame = read_image(frame_path)

                if smol:
                    # Run the frame through the SAHI slicer, then SAM to get prediction
                    sliced_predictions = get_sliced_prediction(original_frame,
                                                               yolo_model,
                                                               overlap_height_ratio=0.65,
                                                               overlap_width_ratio=0.65,
                                                               postprocess_class_agnostic=True,
                                                               postprocess_match_threshold=0.9)

                    # Extract the bounding boxes
                    bboxes = np.array([_.bbox.to_xyxy() for _ in sliced_predictions.object_prediction_list])
                    confidences = np.array([_.score.value for _ in sliced_predictions.object_prediction_list])

                    if len(bboxes):

                        # Update results (version issue)
                        detections = sv.Detections(xyxy=bboxes,
                                                   confidence=confidences,
                                                   class_id=np.full(len(bboxes, ), fill_value=0))

                        # Run the boxes through SAM as prompts
                        masks = sam_model(original_frame, bboxes=bboxes, show=False)[0]
                        masks = masks.masks.data.cpu().numpy()
                        detections.mask = masks

                    else:
                        # If there are no detections, make dummy
                        detections = sv.Detections.empty()

                else:
                    # Run the frame through the YOLO model to get predictions
                    result = yolo_model(original_frame,
                                        imgsz=1280,
                                        conf=conf,
                                        iou=iou,
                                        half=False,
                                        augment=False,
                                        max_det=2000,
                                        verbose=False,
                                        retina_masks=True,
                                        show=False)[0]

                    # Version issues
                    result.obb = None

                    # Convert the results
                    detections = sv.Detections.from_ultralytics(result)

                # Use NMS
                detections = detections.with_nms(iou, class_agnostic=True)
                # Get the masks of the detections
                masks = detections.mask.astype(np.uint8)
                # Convert to polygons
                polygons = mask_to_polygons(masks)
                # Convert to points
                polygon_points = polygons_to_points(polygons, original_frame)

                # Add each of the polygon points to the localization list
                for points in polygon_points:
                    # Specify spec
                    spec = {
                        'type': loc_type.id,
                        'media_id': media.id,
                        'version': version_type.id,
                        'points': points,
                        'frame': int(frame_id),
                        'Label': 'Rock',
                    }
                    localizations.append(spec)

                if debug:
                    # Create rendered results
                    annotated_frame = mask_annotator.annotate(scene=original_frame, detections=detections)
                    annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=detections)
                    # Save to render folder
                    sink.save_image(image=annotated_frame, image_name=frame_name)
                    print(f"NOTE: Rendered results for {frame_name}")

        # ------------------------------------------------
        # Upload to TATOR
        # ------------------------------------------------

        print(f"NOTE: Uploading {len(localizations)} predictions to TATOR")

        # Total localizations uploaded
        num_uploaded = 0

        # Loop through and upload to TATOR
        for response in tator.util.chunked_create(api.create_localization_list, project_id, body=localizations):
            num_uploaded += 1

        # Notify the user
        notify_user(api, project_id, media_id, start_at, end_at)

    except Exception as e:
        print(f"ERROR: {e}")
        print(traceback.print_exc())

    if os.path.exists(downloaded_frames_dir):
        shutil.rmtree(downloaded_frames_dir)


# ----------------------------------------------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------------------------------------------

def main():
    """

    :return:
    """
    try:

        # Fill in the variables from env
        token = os.getenv("TOKEN")
        project_id = os.getenv("PROJECT_ID")
        media_id = os.getenv("MEDIA_ID")
        start_at = os.getenv("START_AT")
        end_at = os.getenv("END_AT")
        conf = os.getenv("CONFIDENCE")
        iou = os.getenv("IOU")
        smol = os.getenv("SMOL")
        debug = os.getenv("DEBUG")

        # Do checks: if empty, exit early
        if None in [token, project_id, media_id, start_at, end_at]:
            raise Exception("ERROR: Missing parameter value(s)!")

        # If the user didn't provide, use default values
        conf = conf if conf else 0.5
        iou = iou if iou else 0.7
        smol = smol if smol else False
        debug = debug if type(debug) == bool else False

        algorithm(token=str(token),
                  project_id=str(project_id),
                  media_id=str(media_id),
                  start_at=int(start_at),
                  end_at=int(end_at),
                  conf=float(conf),
                  iou=float(iou),
                  smol=bool(smol),
                  debug=debug)

        print("Done.")

    except Exception as e:
        print(f"ERROR: Failed to complete inference!\n{e}")
        print(traceback.format_exc())


if __name__ == "__main__":
    main()
