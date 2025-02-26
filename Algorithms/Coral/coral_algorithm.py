import gc

from torch import cuda

import supervision as sv

from Algorithms.algorithm import Algorithm

from Algorithms.common import polygons_to_points

import time


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class CoralAlgorithm(Algorithm):
    """
    Coral detection algorithm

    Utilizes a configuration file containing the various model/algorithm parameters.
    """

    def __init__(self, config: dict):
        """
        :param config:
        """
        super().__init__(config)

    def infer(self, original_frame, image_width=None, image_height=None):
        """
        Performs inference on a single frame; if using sahi mode, will use the
        slicer callback function to perform SAHI using supervision (detections will
        be aggregated together).

        Detections will be used with SAM to create instance segmentation masks.

        :param original_frame:
        """
        start_time = time.time()

        # Perform detection normally
        t1 = time.time()
        detections = self.ultralytics_model(original_frame,
                                            iou=self.iou,
                                            conf=self.conf,
                                            imgsz=self.imgsz,
                                            device=self.device,
                                            verbose=False)[0]
        
        print(f"YOLO inference time: {time.time() - t1:.3f}s")

        # Convert results to supervision standards
        t2 = time.time()
        detections = sv.Detections.from_ultralytics(detections)
        print(f"Convert to supervision time: {time.time() - t2:.3f}s")

        # Do NMM / NMS with all detections (bboxes)
        t3 = time.time()
        detections = detections.with_nms(self.iou, class_agnostic=True)
        print(f"NMS time: {time.time() - t3:.3f}s")

        # Get the boxes
        t4 = time.time()
        bboxes = detections.xyxy
        conf = detections.confidence.tolist()
        print(f"Get boxes time: {time.time() - t4:.3f}s")

        # Convert to points
        t5 = time.time()
        polygon_points = polygons_to_points(bboxes, original_frame)
        print(f"Convert to points time: {time.time() - t5:.3f}s")
        
        # Clear memory
        cuda.empty_cache()
        gc.collect()

        print(f"Total inference time: {time.time() - start_time:.3f}s")
        return polygon_points, conf
