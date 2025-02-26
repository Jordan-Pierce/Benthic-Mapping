import os

import numpy as np

import torch
from torch.cuda import is_available

import supervision as sv
from ultralytics import YOLO
from ultralytics import RTDETR
from ultralytics import SAM

from Algorithms.common import mask_image
from Algorithms.common import calculate_slice_parameters


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class Algorithm:
    """
    Base class for detection algorithms
    """
    def __init__(self, config: dict):
        """
        Initialize base algorithm with common attributes
        
        :param config: Configuration dictionary
        """
        self.config = config
        
        # Parameters in the config file with defaults
        self.iou = self.config.get("iou_threshold", 0.5)
        self.conf = self.config.get("model_confidence_threshold", 0.7)

        self.task = 'detect'
        self.ultralytics_model = None
        self.sam_model = None
        
        self.imgsz = 640
        
        self.slicer = None
        
        self.device = 'cuda:0' if is_available() else 'cpu'
        print("NOTE: Using device", self.device)
        
    def initialize_ultralytics_model(self):
        """
        Initializes the ultralytics model
        """
        if "model_path" not in self.config:
            raise Exception("ERROR: Model path not found in configuration file!")
        
        if "model_type" not in self.config:
            raise Exception("ERROR: Model type not found in configuration file!")
        
        model_path = self.config["model_path"]

        if not os.path.exists(model_path):
            raise Exception(f"ERROR: Model weights not found in ({model_path})!")

        try:
            if "yolo" in self.config['model_type'].lower():
                self.ultralytics_model = YOLO(model_path)
            elif "rtdetr" in self.config['model_type'].lower():
                self.ultralytics_model = RTDETR(model_path)
            else:
                raise Exception(f"ERROR: Model type {self.config['model_type']} not recognized!")
            
            # Update the task 
            self.task = self.ultralytics_model.task
            
            try:
                # Get the image size from the model (better results)
                self.imgsz = self.ultralytics_model.__dict__['overrides']['imgsz']
            except Exception:
                self.imgsz = 640
            
            # Perform a dummy inference to load the model
            self.ultralytics_model(torch.zeros((1, 3, self.imgsz, self.imgsz), device=self.device), 
                                   verbose=False, 
                                   device=self.device)

            print(f"NOTE: Successfully loaded weights {model_path}")
            
        except Exception as e:
            raise Exception(f"ERROR: Could not load ultralytics model!\n{e}")

    def initialize_sam_model(self):
        """
        Initializes the SAM model if configured
        """
        try:
            if "sam_model_path" in self.config:
                # Load the SAM model (will download if not found)
                self.sam_model = SAM(self.config["sam_model_path"])            
                # Update the task 
                self.task = 'segment'
                print(f"NOTE: Successfully loaded SAM model {self.config['sam_model_path']}")
                
        except Exception as e:
            raise Exception(f"ERROR: Could not load SAM model!\n{e}")

    def initialize(self):
        """
        Initializes all models
        """
        self.initialize_ultralytics_model()
        self.initialize_sam_model()

    def setup_slicer(self, frame):
        """
        Creates the SAHI slicer to be used within slicer callback;
        uses the video dimensions to determine slicer parameters.
        """
        if self.slicer is None:
            # Calculate slice parameters based on the size of the image
            slice_wh, overlap_wh = calculate_slice_parameters(frame.shape[0], frame.shape[1])
            # Create the slicer object
            self.slicer = sv.InferenceSlicer(slice_wh=slice_wh,
                                             overlap_ratio_wh=None,  # Must set to None manually, deprecated
                                             overlap_wh=overlap_wh,
                                             iou_threshold=0.25,  # Strict NMS
                                             callback=self.slicer_callback,
                                             thread_workers=os.cpu_count() // 3)

    def slicer_callback(self, image_slice):
        """
        Callback function to perform SAHI using supervision
        """
        # Perform inference on the image slice
        results = self.ultralytics_model(image_slice, verbose=False, device=self.device)[0]
        # Convert results to supervision standards
        results = sv.Detections.from_ultralytics(results)
        return results

    def apply_sahi(self, frame, detections):
        """
        Performs SAHI on a masked frame, where masked regions are areas
        that have already been detected / segmented by initial inference.
        """
        if self.slicer is None and self.config["sahi"]:
            # Setup slicer if not already initialized
            self.setup_slicer(frame)

            if detections:
                # Mask the frame using the detections
                masked_frame = mask_image(frame, detections.mask)
                # Perform SAHI on the masked frame
                sahi_detections = self.slicer(masked_frame)
                # Apply SAM to the detections (if available)
                sahi_detections = self.apply_sam(masked_frame, sahi_detections)

                if sahi_detections:
                    # Merge the detections together
                    detections = sv.Detections.merge([detections, sahi_detections])

        return detections

    def apply_sam(self, frame, detections):
        """
        Override apply_sam to implement SAM model processing
        """
        if self.sam_model and detections:
            # Perform SAM on the detections, get the bounding boxes
            bboxes = detections.xyxy
            # Perform instance segmentation using SAM
            masks = self.sam_model(frame, 
                                   bboxes=bboxes, 
                                   imgsz=640,
                                   device=self.device)[0]
            # Convert masks to numpy
            masks = masks.masks.data.cpu().numpy()
            detections.mask = masks.astype(np.uint8)

        return detections
    
    def infer(self, original_frame, image_width=None, image_height=None):
        """
        Performs inference on a single frame; if using sahi mode, will use the
        slicer callback function to perform SAHI using supervision (detections will
        be aggregated together).
        
        :param original_frame: Original frame to perform inference on
        """
        raise NotImplementedError("Inference method must be implemented in child class!")
