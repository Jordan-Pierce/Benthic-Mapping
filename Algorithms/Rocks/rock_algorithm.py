import gc

from torch import cuda

import supervision as sv

from Algorithms.algorithm import Algorithm

from Algorithms.common import mask_to_polygons
from Algorithms.common import polygons_to_points

import time


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class RockAlgorithm(Algorithm):
    """
    Rock detection algorithm

    Utilizes a configuration file containing the various model/algorithm parameters.
    """

    def __init__(self, config: dict):
        super().__init__(config)

    def infer(self, original_frame, image_width=None, image_height=None) -> list:
        """
        Performs inference on a single frame; if using sahi mode, will use the
        slicer callback function to perform SAHI using supervision (detections will
        be aggregated together).

        Detections will be used with SAM to create instance segmentation masks.

        :param original_frame:
        """
        start_time = time.time()

        # Perform detections using the whole frame
        t0 = time.time()
        detections = self.ultralytics_model(original_frame,
                                            iou=self.iou,
                                            conf=self.conf,
                                            imgsz=self.imgsz,
                                            device=self.device,
                                            verbose=False)[0]
        
        print(f"YOLO inference took: {time.time() - t0:.3f} seconds")
        
        # Convert results to supervision standards
        t0 = time.time()
        detections = sv.Detections.from_ultralytics(detections)
        print(f"Converting to supervision format took: {time.time() - t0:.3f} seconds")

        # Perform segmentations with bboxes using SAM (if specified in config)
        t0 = time.time()
        detections = self.apply_sam(original_frame, detections)
        print(f"SAM segmentation took: {time.time() - t0:.3f} seconds")

        # Perform detections / segmentations using SAHI (if specified in config)
        t0 = time.time()
        detections = self.apply_sahi(original_frame, detections)
        print(f"SAHI processing took: {time.time() - t0:.3f} seconds")

        # Do NMM / NMS with all detections (bboxes)
        t0 = time.time()
        if self.task == 'detect':
            detections = detections.with_nms(self.iou, class_agnostic=True)
        else:
            detections = detections.with_nmm(self.iou, class_agnostic=True)
            
        print(f"NMM/NMS took: {time.time() - t0:.3f} seconds")

        # Convert to polygons and points
        t0 = time.time()
        polygons = mask_to_polygons(detections.mask)
        polygon_points = polygons_to_points(polygons, original_frame)
        print(f"Polygon conversion took: {time.time() - t0:.3f} seconds")

        # Clear memory
        cuda.empty_cache()
        gc.collect()
        
        print(f"Total processing time: {time.time() - start_time:.3f} seconds")
        return polygon_points
