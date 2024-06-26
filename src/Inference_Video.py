import os.path
import argparse
import traceback
from tqdm import tqdm

import cv2
import numpy as np

import torch
import supervision as sv
from ultralytics import SAM
from ultralytics import YOLO


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------

class VideoInferencer:
    def __init__(self, weights_path: str, video_path: str, output_dir: str, start_at: int, end_at: int,
                 conf: float, iou: float, track: bool, segment: bool, smol: bool, show: bool):
        """

        :param weights_path:
        :param video_path:
        :param output_dir:
        :param start_at:
        :param end_at:
        :param conf:
        :param iou:
        :param track:
        :param smol:
        :param show:
        """
        self.source_weights = weights_path
        self.source_video = video_path
        self.output_dir = output_dir
        self.start_at = start_at
        self.end_at = end_at
        self.conf = conf
        self.iou = iou
        self.track = track
        self.segment = segment
        self.smol = smol
        self.show = show

        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

        self.yolo_model = None
        self.sam_model = None
        self.box_annotator = None
        self.mask_annotator = None
        self.labeler = None
        self.tracker = None
        self.slicer = None

    @staticmethod
    def calculate_slice_parameters(width: int, height: int, slices_x: int = 2, slices_y: int = 2, overlap: float = 0.1):
        """
        Calculate slice parameters for video frames; defaults to 2x2, 10% overlap

        :param width:
        :param height:
        :param slices_x:
        :param slices_y:
        :param overlap:
        :return:
        """
        slice_width = width // slices_x
        slice_height = height // slices_y
        overlap_width = int(slice_width * overlap)
        overlap_height = int(slice_height * overlap)
        adjusted_slice_width = slice_width + overlap_width
        adjusted_slice_height = slice_height + overlap_height
        overlap_ratio_w = overlap_width / adjusted_slice_width
        overlap_ratio_h = overlap_height / adjusted_slice_height

        return (adjusted_slice_width, adjusted_slice_height), (overlap_ratio_w, overlap_ratio_h)

    def slicer_callback(self, image_slice: np.ndarray):
        """
        Prepares a callback to be used with SAHI

        :param image_slice:
        :return:
        """
        results = self.yolo_model(image_slice)[0]
        results = sv.Detections.from_ultralytics(results)
        return results

    def load_models(self):
        """
        Loads the models

        :return:
        """
        try:
            self.yolo_model = YOLO(self.source_weights)
            self.sam_model = SAM('sam_l.pt')

            self.box_annotator = sv.BoundingBoxAnnotator()
            self.mask_annotator = sv.MaskAnnotator()
            self.labeler = sv.LabelAnnotator()

            if self.track:
                self.tracker = sv.ByteTrack()

        except Exception as e:
            raise Exception(f"ERROR: Could not load model!\n{e}")

    def setup_slicer(self, video_info):
        """
        Creates the SAHI slicer to be used within slicer callback;
        uses the video dimensions to determine slicer parameters.

        :param video_info:
        :return:
        """
        slice_wh, overlap_ratio_wh = self.calculate_slice_parameters(video_info.width, video_info.height)

        self.slicer = sv.InferenceSlicer(callback=self.slicer_callback,
                                         slice_wh=slice_wh,
                                         iou_threshold=0.10,
                                         overlap_ratio_wh=overlap_ratio_wh,
                                         overlap_filter_strategy=sv.OverlapFilter.NON_MAX_MERGE)

    def run_inference(self):
        """
        Runs inference on the video, range of frames specified.

        :return:
        """
        os.makedirs(self.output_dir, exist_ok=True)
        target_video_path = f"{self.output_dir}/{os.path.basename(self.source_video)}"
        frame_generator = sv.get_video_frames_generator(source_path=self.source_video)
        video_info = sv.VideoInfo.from_video_path(video_path=self.source_video)

        self.load_models()

        if self.smol:
            self.setup_slicer(video_info)
        if self.start_at <= 0:
            self.start_at = 0
        if self.end_at <= -1:
            self.end_at = video_info.total_frames

        f_idx = 0
        with sv.VideoSink(target_path=target_video_path, video_info=video_info) as sink:
            for frame in tqdm(frame_generator, total=video_info.total_frames):

                if self.start_at < f_idx < self.end_at:
                    detections = self.yolo_model(frame, iou=self.iou, conf=self.conf, device=self.device)[0]
                    detections = sv.Detections.from_ultralytics(detections)

                    if self.smol:
                        smol_detections = self.slicer(frame)
                        detections = sv.Detections.merge([detections, smol_detections])

                    detections = detections.with_nms(self.iou, class_agnostic=True)
                    detections = detections[detections.confidence > self.conf]
                    labels = detections.tracker_id

                    if self.segment:
                        bboxes = detections.xyxy
                        masks = self.sam_model(frame, bboxes=bboxes)[0]
                        masks = masks.masks.data.cpu().numpy()
                        detections.mask = masks
                    if self.track:
                        detections = self.tracker.update_with_detections(detections)
                        labels = detections.tracker_id.astype(str)

                    frame = self.mask_annotator.annotate(scene=frame, detections=detections)
                    frame = self.box_annotator.annotate(scene=frame, detections=detections)
                    frame = self.labeler.annotate(scene=frame, detections=detections, labels=labels)

                    sink.write_frame(frame=frame)

                    if self.show:
                        cv2.imshow('Predictions', frame)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break

                f_idx += 1
                if f_idx >= self.end_at:
                    cv2.destroyAllWindows()
                    break


# ----------------------------------------------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------------------------------------------

def main():

    parser = argparse.ArgumentParser(description="Video Inferencing with YOLO, SAM, and ByteTrack")

    parser.add_argument("--source_weights", required=True, type=str,
                        help="Path to the source weights file")

    parser.add_argument("--source_video", required=True, type=str,
                        help="Path to the source video file")

    parser.add_argument("--output_dir", required=True, type=str,
                        help="Path to the target video directory (output)")

    parser.add_argument("--start_at", default=0, type=int,
                        help="Frame to start inference")

    parser.add_argument("--end_at", default=-1, type=int,
                        help="Frame to end inference")

    parser.add_argument("--conf", default=0.65, type=float,
                        help="Confidence threshold for the model")

    parser.add_argument("--iou", default=0.25, type=float,
                        help="IOU threshold for the model")

    parser.add_argument("--track", action='store_true',
                        help="Uses ByteTrack to track detections")

    parser.add_argument("--segment", action='store_true',
                        help="Uses SAM to create masks on detections (takes longer)")

    parser.add_argument("--smol", action='store_true',
                        help="Uses SAHI to find smaller objects (takes longer)")

    parser.add_argument("--show", action='store_true',
                        help="Display the inference video")

    args = parser.parse_args()

    try:
        inference = VideoInferencer(
            weights_path=args.weights_path,
            video_path=args.video_path,
            output_dir=args.output_dir,
            start_at=args.start_at,
            end_at=args.end_at,
            conf=args.conf,
            iou=args.iou,
            track=args.track,
            segment=args.segment,
            smol=args.smol,
            show=args.show
        )
        inference.run_inference()
        print("Done.")
    except Exception as e:
        print(f"ERROR: Could not finish process.\n{e}")
        print(traceback.format_exc())


if __name__ == "__main__":
    main()
