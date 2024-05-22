import os.path
import argparse
from tqdm import tqdm

import numpy as np

import torch
import supervision as sv

from ultralytics import SAM
from ultralytics import YOLO
from ultralytics.engine.results import Results

from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction


# ----------------------------------------------------------------------------------------------------------------------
# Functions
# ----------------------------------------------------------------------------------------------------------------------


def inference(source_weights, source_video, output_dir, task, start_at, end_at, conf=.5, iou=.7, smol=True, show=False):
    """

    :param source_weights:
    :param source_video:
    :param output_dir:
    :param task:
    :param start_at:
    :param end_at:
    :param conf:
    :param iou:
    :param smol:
    :param show:
    :return:
    """
    # Create the target path
    target_video_path = f"{output_dir}/{os.path.basename(source_video)}"
    # Create the target directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Check for CUDA
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    try:

        # If using SAHI, initialize model differently
        if smol:
            # Load the YOLO model as auto-detection
            yolo_model = AutoDetectionModel.from_pretrained(
                model_type='yolov8',
                model_path=source_weights,
                confidence_threshold=conf,
                device=device
            )
            # Load the SAM model
            sam_model = SAM('sam_l.pt')

        else:
            # Normal YOLO model convention
            yolo_model = YOLO(source_weights)

    except Exception as e:
        raise Exception(f"ERROR: Could not load models;\n{e}")

    if task == 'detect':
        # Load the tracker
        tracker = sv.ByteTrack()
        # Create the annotator for detection
        box_annotator = sv.BoundingBoxAnnotator()
        # Adds label to annotation (tracking)
        labeler = sv.LabelAnnotator()
        # TTA
        augment = True
        # Not applicable
        retina_masks = False

    elif args.task == 'segment':
        # Create the annotators for segmentation
        mask_annotator = sv.MaskAnnotator()
        box_annotator = sv.BoundingBoxAnnotator()
        # TTA
        augment = False
        # Use high quality masks
        retina_masks = True

    else:
        raise Exception("ERROR: Specify --task [detect, segment]")

    # Create the video generators
    frame_generator = sv.get_video_frames_generator(source_path=source_video)
    video_info = sv.VideoInfo.from_video_path(video_path=source_video)

    # Where to start and end inference
    if start_at <= 0:
        start_at = 0
    if end_at <= -1:
        end_at = video_info.total_frames

    # Frame count
    f_idx = 0

    # Loop through all the frames
    with sv.VideoSink(target_path=target_video_path, video_info=video_info) as sink:
        for frame in tqdm(frame_generator, total=video_info.total_frames):

            # Only make predictions within range
            if start_at < f_idx < end_at:

                if smol:
                    # Run the frame through the SAHI slicer, then SAM to get prediction
                    sliced_predictions = get_sliced_prediction(frame,
                                                               yolo_model,
                                                               overlap_height_ratio=0.75,
                                                               overlap_width_ratio=0.75,
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

                        if task == 'segment':
                            # Run the boxes through SAM as prompts
                            masks = sam_model(frame, bboxes=bboxes, show=False)[0]
                            masks = masks.masks.data.cpu().numpy()
                            detections.mask = masks

                    else:
                        # If there are no detections, make dummy
                        detections = sv.Detections.empty()

                else:
                    # Run the frame through the YOLO model to get predictions
                    result = yolo_model(frame,
                                        conf=conf,
                                        iou=iou,
                                        half=True,
                                        augment=augment,
                                        max_det=2000,
                                        verbose=False,
                                        retina_masks=retina_masks,
                                        show=False)[0]

                    # Version issues
                    result.obb = None

                    # Convert the results
                    detections = sv.Detections.from_ultralytics(result)

                # Use NMS
                detections = detections.with_nms(iou, class_agnostic=True)

                if task == 'detect':
                    # Track the detections
                    detections = tracker.update_with_detections(detections)
                    labels = [f"#{tracker_id}" for tracker_id in detections.tracker_id]

                    # Create an annotated version of the frame (boxes)
                    frame = box_annotator.annotate(scene=frame, detections=detections)
                    frame = labeler.annotate(scene=frame, detections=detections, labels=labels)

                else:
                    # Create an annotated version of the frame (masks and boxes)
                    frame = mask_annotator.annotate(scene=frame, detections=detections)
                    frame = box_annotator.annotate(scene=frame, detections=detections)

                # Write the frame to the video
                sink.write_frame(frame=frame)

                if show:
                    Results(names={}, path="", orig_img=frame).show()

            # Increase frame count
            f_idx += 1

            # Break early as needed
            if f_idx >= end_at:
                break


# ----------------------------------------------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Video Processing with YOLO and ByteTrack")

    parser.add_argument(
        "--source_weights",
        required=True,
        help="Path to the source weights file",
        type=str,
    )
    parser.add_argument(
        "--source_video",
        required=True,
        help="Path to the source video file",
        type=str,
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Path to the target video directory (output)",
        type=str,
    )
    parser.add_argument(
        "--task",
        required=True,
        help="Task to perform [detect, segment], if applicable",
        type=str,
    )
    parser.add_argument(
        "--start_at",
        default=0,
        help="Frame to start inference",
        type=int,
    )
    parser.add_argument(
        "--end_at",
        default=-1,
        help="Frame to end inference",
        type=int,
    )
    parser.add_argument(
        "--conf",
        default=0.5,
        help="Confidence threshold for the model",
        type=float,
    )
    parser.add_argument(
        "--iou",
        default=0.75,
        help="IOU threshold for the model",
        type=float
    )
    parser.add_argument(
        "--smol",
        action='store_true',
        help="Uses SAHI to find smaller objects (takes longer)",
    )
    parser.add_argument(
        "--show",
        action='store_true',
        help="Display the inference video",
    )

    args = parser.parse_args()

    inference(
        source_weights=args.source_weights,
        source_video=args.source_video,
        output_dir=args.output_dir,
        task=args.task,
        start_at=args.start_at,
        end_at=args.end_at,
        conf=args.conf,
        iou=args.iou,
        smol=args.smol,
        show=args.show
    )
