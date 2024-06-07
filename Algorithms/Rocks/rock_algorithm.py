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
    :param polygons: List of points
    :param image: numpy array
    :return: List of normalized points for each polygon relative to the provided image dim
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

class RockAlgorithm():
    """ Rock detection algorithm

    Utilizes a configuration file containing the various model/algorithm parameters.

    """

    def __init__(self, config: dict):
        """ Constructor

        :param config: Dictionary containing algorithm configuration information

        """

        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.config = config

    def initialize(self):
        """ Initialize the model
        """

        model_path = self.config["model_path"]

        if not os.path.exists(model_path):
            raise Exception(f"ERROR: Model weights not found in ({model_path})!")

        # If using SAHI, initialize model differently
        if self.config["smol"]:
            # Load the YOLO model as auto-detection
            self.yolo_model = AutoDetectionModel.from_pretrained(
                model_type=self.config["model_type"],
                model_path=model_path,
                confidence_threshold=self.config["model_confidence_threshold"],
                device=self.device
            )
            # Load the SAM model
            self.sam_model = SAM(self.config["sam_model_path"])

        else:
            # Normal YOLO model convention
            self.yolo_model = YOLO(model_path)

        print(f"NOTE: Successfully loaded weights {model_path}")

    @torch.no_grad()
    def infer(self, original_frame) -> list:
        """
        :param original_frame: numpy form of the image to process (use sahi read_image())
        """

        conf = self.config["model_confidence_threshold"]
        iou = self.config["iou_threshold"]

        if self.config["smol"]:
            # Run the frame through the SAHI slicer, then SAM to get prediction

            sahi_parameters = {
              "overlap_height_ratio": self.config["overlap_height_ratio"],
              "overlap_width_ratio": self.config["overlap_width_ratio"],
              "postprocess_class_agnostic": self.config["postprocess_class_agnostic"],
              "postprocess_match_threshold": self.config["postprocess_match_threshold"]
            }

            sliced_predictions = get_sliced_prediction(original_frame,
                                                       self.yolo_model,
                                                       **sahi_parameters)

            # Extract the bounding boxes
            bboxes = np.array([_.bbox.to_xyxy() for _ in sliced_predictions.object_prediction_list])
            confidences = np.array([_.score.value for _ in sliced_predictions.object_prediction_list])

            if len(bboxes):

                # Update results (version issue)
                detections = sv.Detections(xyxy=bboxes,
                                           confidence=confidences,
                                           class_id=np.full(len(bboxes, ), fill_value=0))

                # Run the boxes through SAM as prompts
                masks = self.sam_model(original_frame, bboxes=bboxes, show=False)[0]
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

        return polygon_points

