import os

import cv2
import numpy as np

import torch
import supervision as sv
from ultralytics import SAM
from ultralytics import YOLO
from ultralytics import RTDETR


# ----------------------------------------------------------------------------------------------------------------------
# Functions
# ----------------------------------------------------------------------------------------------------------------------
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
    # To hold the normalized polygons
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


def calculate_slice_parameters(height, width, slices_x=2, slices_y=2, overlap_percentage=0.1):
    """
    Calculates the slice parameters when using SAHI.

    :param height:
    :param width:
    :param slices_x:
    :param slices_y:
    :param overlap_percentage:
    :return:
    """
    # Calculate base slice dimensions
    slice_width = width // slices_x
    slice_height = height // slices_y

    # Calculate overlap
    overlap_width = int(slice_width * overlap_percentage)
    overlap_height = int(slice_height * overlap_percentage)

    # Adjust slice dimensions to include overlap
    adjusted_slice_width = slice_width + overlap_width
    adjusted_slice_height = slice_height + overlap_height

    # Calculate overlap ratios
    overlap_ratio_w = overlap_width / adjusted_slice_width
    overlap_ratio_h = overlap_height / adjusted_slice_height

    return (adjusted_slice_width, adjusted_slice_height), (overlap_ratio_w, overlap_ratio_h)


class RockAlgorithm:
    """
    Rock detection algorithm

    Utilizes a configuration file containing the various model/algorithm parameters.
    """

    def __init__(self, config: dict):
        """

        :param config:
        """

        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.config = config

        self.sam_model = None
        self.yolo_model = None

        self.slicer = None

    def initialize(self):
        """
        Initializes the model

        :return:
        """
        # Get the rock detection model path as specified in the config file
        model_path = self.config["model_path"]

        if not os.path.exists(model_path):
            raise Exception(f"ERROR: Model weights not found in ({model_path})!")

        try:
            if "yolo" in self.config['model_type'].lower():
                # Load the model weights
                self.yolo_model = YOLO(model_path)
            elif "rtdetr" in self.config['model_type'].lower():
                self.yolo_model = RTDETR(model_path)
            else:
                raise Exception(f"ERROR: Model type {self.config['model_type']} not recognized!")

            # Load the SAM model (sam_b, sam_l, sam_h)
            # This will download the file if it doesn't exist
            self.sam_model = SAM(self.config["sam_model_path"])

        except Exception as e:
            raise Exception(f"ERROR: Could not load model!\n{e}")

        print(f"NOTE: Successfully loaded weights {model_path}")

    @torch.no_grad()
    def slicer_callback(self, image_slice):
        """
        Callback function to perform SAHI using supervision

        :param image_slice:
        :return:
        """
        # Get the results of the slice
        results = self.yolo_model(image_slice, verbose=False)[0]
        # Convert results to supervision standards
        results = sv.Detections.from_ultralytics(results)

        return results

    @torch.no_grad()
    def infer(self, original_frame) -> list:
        """
        Performs inference on a single frame; if using smol mode, will use the
        slicer callback function to perform SAHI using supervision (detections will
        be aggregated together).

        Detections will be used with SAM to create instance segmentation masks.

        :param original_frame:
        """
        # Parameters in the config file
        iou = self.config["iou_threshold"]
        conf = self.config["model_confidence_threshold"]

        # Perform detection normally
        detections = self.yolo_model(original_frame,
                                     iou=iou,
                                     conf=conf,
                                     device=self.device,
                                     verbose=False)[0]

        # Convert results to supervision standards
        detections = sv.Detections.from_ultralytics(detections)

        if self.config["smol"]:

            # For the first image, calculate the SAHI parameters
            if self.slicer is None:
                # Get the image dimensions
                frame_height, frame_width = original_frame.shape[:2]
                # Get the slicer parameter values (defaults 2 x 2 with 10 % overlap)
                slice_wh, overlap_ratio_wh = calculate_slice_parameters(frame_height, frame_width)
                # Create the slicer
                self.slicer = sv.InferenceSlicer(callback=self.slicer_callback,
                                                 slice_wh=slice_wh,
                                                 iou_threshold=0.10,
                                                 overlap_ratio_wh=overlap_ratio_wh,
                                                 overlap_filter_strategy=sv.OverlapFilter.NON_MAX_MERGE)

            # Run the frame through the slicer
            smol_detections = self.slicer(original_frame)
            # Merge the results
            detections = sv.Detections.merge([detections, smol_detections])

        # Perform NMS, and filter based on confidence scores
        detections = detections.with_nms(iou, class_agnostic=True)
        detections = detections[detections.confidence > conf]

        # Predict instance segmentation masks using SAM
        bboxes = detections.xyxy
        masks = self.sam_model(original_frame, bboxes=bboxes, verbose=False)[0]
        masks = masks.masks.data.cpu().numpy()
        detections.mask = masks.astype(np.uint8)

        # Convert to polygons
        polygons = mask_to_polygons(detections.mask)
        # Convert to points
        polygon_points = polygons_to_points(polygons, original_frame)

        return polygon_points
