import os.path
import argparse
from tqdm import tqdm

import cv2

import torch
import supervision as sv
from ultralytics import SAM
from ultralytics import YOLO


# ----------------------------------------------------------------------------------------------------------------------
# Functions
# ----------------------------------------------------------------------------------------------------------------------
def calculate_slice_parameters(video_width, video_height, slices_x=2, slices_y=2, overlap_percentage=0.1):
    """

    :param video_width:
    :param video_height:
    :param slices_x:
    :param slices_y:
    :param overlap_percentage:
    :return:
    """
    # Calculate base slice dimensions
    slice_width = video_width // slices_x
    slice_height = video_height // slices_y

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


def inference(source_weights, source_video, output_dir, start_at, end_at, conf, iou, track, smol, show):
    """

    :param source_weights:
    :param source_video:
    :param output_dir:
    :param start_at:
    :param end_at:
    :param conf:
    :param iou:
    :param track:
    :param smol:
    :param show:
    :return:
    """
    def slicer_callback(image_slice):
        """Callback function to perform SAHI using supervision"""
        # Get the results of the slice
        results = yolo_model(image_slice)[0]
        # Convert results to supervision standards
        results = sv.Detections.from_ultralytics(results)

        return results

    # Check for CUDA
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # Create the target path
    target_video_path = f"{output_dir}/{os.path.basename(source_video)}"
    # Create the target directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Create the video generators
    frame_generator = sv.get_video_frames_generator(source_path=source_video)
    video_info = sv.VideoInfo.from_video_path(video_path=source_video)

    try:  # Load the model weights
        yolo_model = YOLO(source_weights)

        # Load the SAM model
        sam_model = SAM('sam_l.pt')

        # Create the annotator for detection
        box_annotator = sv.BoundingBoxAnnotator()
        # Create the annotators for segmentation
        mask_annotator = sv.MaskAnnotator()
        # Adds label to annotation (tracking)
        labeler = sv.LabelAnnotator()

    except Exception as e:
        raise Exception(f"ERROR: Could not load model!\n{e}")

    if track:  # Set up the tracker
        tracker = sv.ByteTrack()

    if smol:  # Set up the slicer
        # Get the slicer parameter values (defaults 2 x 2 with 10 % overlap)
        slice_wh, overlap_ratio_wh = calculate_slice_parameters(video_info.width, video_info.height)
        # Create the slicer
        slicer = sv.InferenceSlicer(callback=slicer_callback,
                                    slice_wh=slice_wh,
                                    iou_threshold=0.10,
                                    overlap_ratio_wh=overlap_ratio_wh,
                                    overlap_filter_strategy=sv.OverlapFilter.NON_MAX_MERGE)

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

                # Perform predictions normally
                detections = yolo_model(frame,
                                        iou=iou,
                                        conf=conf,
                                        device=device)[0]

                # Convert results to supervision standards
                detections = sv.Detections.from_ultralytics(detections)

                if smol:
                    # Run the frame through the slicer
                    smol_detections = slicer(frame)
                    # Merge the results
                    detections = sv.Detections.merge([detections, smol_detections])

                # Perform NMS, and filter based on confidence scores
                detections = detections.with_nms(iou, class_agnostic=True)
                detections = detections[detections.confidence > conf]
                labels = detections.tracker_id

                if True:
                    bboxes = detections.xyxy
                    masks = sam_model(frame, bboxes=bboxes)[0]
                    masks = masks.masks.data.cpu().numpy()
                    detections.mask = masks

                if track:  # Track the detections
                    detections = tracker.update_with_detections(detections)
                    labels = detections.tracker_id.astype(str)

                # Create an annotated version of the frame
                frame = mask_annotator.annotate(scene=frame, detections=detections)
                frame = box_annotator.annotate(scene=frame, detections=detections)

                # Adds labels or track IDs
                frame = labeler.annotate(scene=frame,
                                         detections=detections,
                                         labels=labels)

                # Write the frame to the video
                sink.write_frame(frame=frame)

                if show:
                    cv2.imshow('Predictions', frame)

                    # Allow the frame to show, and update
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

            # Increase frame count
            f_idx += 1

            # Break early as needed
            if f_idx >= end_at:
                cv2.destroyAllWindows()
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
        default=0.65,
        help="Confidence threshold for the model",
        type=float,
    )
    parser.add_argument(
        "--iou",
        default=0.25,
        help="IOU threshold for the model",
        type=float
    )
    parser.add_argument(
        "--track",
        action='store_true',
        help="Uses ByteTrack to track detections",
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
        track=args.track,
        start_at=args.start_at,
        end_at=args.end_at,
        conf=args.conf,
        iou=args.iou,
        smol=args.smol,
        show=args.show
    )
