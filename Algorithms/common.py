import cv2
import numpy as np


# ----------------------------------------------------------------------------------------------------------------------
# Functions
# ----------------------------------------------------------------------------------------------------------------------


def mask_image(image, masks):
    """

    :param image:
    :param masks:
    :return:
    """
    masked_image = image.copy()
    combined_masks = np.zeros((image.shape[0], image.shape[1]), np.uint8)
    for mask in masks:
        combined_masks = np.logical_or(combined_masks, mask).astype(np.uint8)
    masked_image[combined_masks == 1] = 0

    return masked_image


def calculate_slice_parameters(height, width, slices_x=2, slices_y=2, overlap_percentage=0.3):
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
    
    overlap_pixels_w = int(overlap_ratio_w * adjusted_slice_width)
    overlap_pixels_h = int(overlap_ratio_h * adjusted_slice_height)

    return (adjusted_slice_width, adjusted_slice_height), (overlap_pixels_w, overlap_pixels_h)


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
            print(f"ERROR: Could not convert mask to polygon!\n{e}")
            continue

    return polygons


def polygons_to_points(polygons, image):
    """
    Convert polygon points to normalized coordinates relative to the image dimensions.

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

        # Ensure normalized coordinates are within bounds [0, 1]
        normalized_polygon_x = np.clip(normalized_polygon_x, 0, 1)
        normalized_polygon_y = np.clip(normalized_polygon_y, 0, 1)

        # Create a new list with normalized coordinates for the current polygon
        normalized_points = np.column_stack((normalized_polygon_x, normalized_polygon_y)).tolist()
        normalized_polygons.append(normalized_points)

    return normalized_polygons
