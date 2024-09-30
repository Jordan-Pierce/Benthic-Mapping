import os
import traceback

import gradio as gr
import tator
import cv2

from Rocks.rock_algorithm import RockAlgorithm
from Coral.coral_algorithm import CoralAlgorithm


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------

class TatorOperator:
    def __init__(self, token, project_id, media_id):
        self.token = token
        self.project_id = project_id
        self.media_id = media_id

        try:
            self.api = tator.get_api(host='https://cloud.tator.io', token=token)
            self.version_type = self.api.whoami().id
            self.media = self.api.get_media(media_id)
            print(f"NOTE: Authentication successful for {self.api.whoami().email}")

        except Exception as e:
            raise Exception(f"ERROR: Could not connect to TATOR.\n{e}")

    def download_media(self, output_dir):
        """
        Download the media (video) from Tator.

        :param output_dir: Directory to save the downloaded video.
        :return: Path to the downloaded video file.
        """
        os.makedirs(output_dir, exist_ok=True)
        media_name = self.media.name.replace(":", "_")
        output_video_path = os.path.join(output_dir, f"{media_name}")

        try:
            print(f"NOTE: Downloading {self.media.name}...")
            for progress in tator.util.download_media(self.api,
                                                      self.media,
                                                      output_video_path,
                                                      self.media.height,
                                                      "streaming"):
                print(f"NOTE: Download progress: {progress}%")

            if os.path.exists(output_video_path):
                print(f"NOTE: Media {self.media.name} downloaded successfully")
                return output_video_path
            else:
                raise Exception(f"ERROR: Media {self.media.name} did not download successfully")

        except Exception as e:
            raise Exception(f"ERROR: Could not download media {self.media_id}: {e}")

    def extract_frame(self, video_path, frame_idx):
        """
        Get a specific frame from the downloaded video.

        :param video_path: Path to the downloaded video file.
        :param frame_idx: Index of the frame to retrieve.
        :return: The specified frame.
        """
        try:
            cap = cv2.VideoCapture(video_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            cap.release()

            if not ret:
                raise Exception(f"ERROR: Could not read frame {frame_idx} from {video_path}")

            return frame

        except Exception as e:
            raise Exception(f"ERROR: Could not get frame {frame_idx} from {video_path}.\n{e}")

    def delete_media(self, media_path):
        """
        Delete the downloaded media file.

        :param media_path: Path to the downloaded video file.
        """
        try:
            if os.path.exists(media_path):
                os.remove(media_path)
                print(f"NOTE: Deleted media file {media_path}")
            else:
                print(f"WARNING: Media file {media_path} does not exist")

        except Exception as e:
            raise Exception(f"ERROR: Could not delete media file {media_path}: {e}")

    @staticmethod
    def parse_frame_ranges(frame_ranges_str):
        """
        Parse the frame ranges string into a list of frame numbers.

        :param frame_ranges_str:
        """
        frames = []
        ranges = frame_ranges_str.split(',')
        for r in ranges:
            r = r.strip()
            if '-' in r:
                start, end = map(int, r.split('-'))
                frames.extend(range(start, end + 1))
            else:
                frames.append(int(r))

        return sorted(set(frames))

    def download_frame(self, frame_idx):
        """

        :param frame_idx:
        :return:
        """
        try:
            # Get the frame from TATOR
            temp = self.api.get_frame(id=self.media_id,
                                      tile=f"{self.media.width}x{self.media.height}",
                                      frames=[frame_idx])

            # Store the frame in memory, delete the actual file
            frame = cv2.imread(temp)
            os.remove(temp)

            return frame

        except Exception as e:
            raise Exception(f"ERROR: Could not get frame {frame_idx} from {self.media_id}.\n{e}")

    def get_spec(self, frame_idx, polygon, conf):
        """

        :return:
        """
        # GL Project
        if self.project_id == 155:

            points, conf = polygon, conf

            return {
                'type': 460,  # rock poly (mask) type
                'media_id': self.media_id,
                'version_id': 545,  # Imported Data
                'points': points,
                'frame': frame_idx,
                'attributes': {
                    "Label": "Rock"
                },
            }

        # MDBC Project
        elif self.project_id == 70:

            # Extract the data
            bbox, score = polygon, conf
            x, y, w, h = bbox

            return {
                'type': 247,  # Detection Box (440), Coral / Inverts (247)
                'media_id': self.media_id,
                'version_id': 408,  # Imported Data
                'x': x,
                'y': y,
                'width': w,
                'height': h,
                'frame': frame_idx,
                'attributes': {
                    "ScientificName": "",
                    "CommonName": "",
                    "Score": score,
                    "Needs Review": True
                },
            }

        else:
            raise Exception(f"ERROR: Project ID {self.project_id} is not valid.")

    def upload_predictions(self, frame_idx, polygons, confs=None):
        """

        :param predictions:
        :param frame_id:
        :return:
        """
        try:
            localizations = []

            if not confs:
                confs = [1.0 for _ in range(len(polygons))]

            # Add each of the polygon points to the localization list
            for (polygon, conf) in list(zip(polygons, confs)):
                # Specify spec
                spec = self.get_spec(frame_idx, polygon, conf)
                localizations.append(spec)

            print(f"NOTE: Uploading {len(localizations)} predictions to TATOR")

            # Total localizations uploaded
            num_uploaded = 0

            # Loop through and upload to TATOR
            for response in tator.util.chunked_create(func=self.api.create_localization_list,
                                                      project=self.project_id,
                                                      body=localizations):
                num_uploaded += len(response.id)

            print(f"NOTE: Successfully uploaded {num_uploaded} localizations for frame {frame_idx}")

        except Exception as e:
            print(f"ERROR: {e}")
            print(traceback.format_exc())


# ----------------------------------------------------------------------------------------------------------------------
# Gradio
# ----------------------------------------------------------------------------------------------------------------------


def run_rock_algorithm(token,
                       remember_token,
                       project_id,
                       media_id,
                       frame_ranges,
                       conf,
                       iou,
                       smol,
                       model_type,
                       model_weights,
                       progress=gr.Progress()):
    """
    Run the rock algorithm on specified frame ranges.

    :param token: Tator API token
    :param remember_token: Whether to remember the token
    :param project_id: Tator project ID
    :param media_id: Tator media ID
    :param frame_ranges: String specifying frame ranges to process
    :param conf: Confidence threshold
    :param iou: IoU threshold
    :param smol: Use SAHI
    :param model_type: Type of model (YOLO, or RTDETR)
    :param model_weights: Path to model weights file
    :param progress: Gradio progress bar
    :return: Status message
    """
    try:
        if remember_token:
            os.environ['TATOR_TOKEN'] = str(token)

        tator_operator = TatorOperator(token, project_id, media_id)

        config = {
            "model_confidence_threshold": float(conf),
            "iou_threshold": float(iou),
            "smol": bool(smol),
            "model_type": str(model_type),
            "model_path": str(model_weights),
            "sam_model_path": "sam2_b.pt"
        }

        rock_algo = RockAlgorithm(config)
        progress(0, "Initializing algorithm...")
        rock_algo.initialize()

        frames_to_process = tator_operator.parse_frame_ranges(frame_ranges)
        total_frames = len(frames_to_process)

        frames = []
        predictions = []

        # Download frames
        for i, frame_idx in enumerate(frames_to_process):
            progress((i + 1) / (3 * total_frames), f"Downloading frame {frame_idx}")
            frame = tator_operator.download_frame(frame_idx)
            frames.append((frame_idx, frame))

        # Perform inference
        for i, (frame_idx, frame) in enumerate(frames):
            progress((i + 1 + total_frames) / (3 * total_frames), f"Making predictions for frame {frame_idx}")
            polygons = rock_algo.infer(frame)
            predictions.append([frame_idx, polygons])

        # Upload predictions
        for i, (frame_idx, polygons) in enumerate(predictions):
            progress((i + 1 + 2 * total_frames) / (3 * total_frames), f"Uploading predictions for frame {frame_idx}")
            tator_operator.upload_predictions(frame_idx, polygons, confs=None)

        gr.Info("Processing completed successfully!")
        return "Done."

    except Exception as e:
        error_msg = f"ERROR: Failed to complete inference!\n{e}\n{traceback.format_exc()}"
        gr.Error(error_msg)
        return error_msg


def run_coral_algorithm(token,
                        remember_token,
                        project_id,
                        media_id,
                        frame_ranges,
                        conf,
                        iou,
                        model_type,
                        model_weights,
                        progress=gr.Progress()):
    """
    Run the coral algorithm on specified frame ranges.

    :param token: Tator API token
    :param remember_token: Whether to remember the token
    :param project_id: Tator project ID
    :param media_id: Tator media ID
    :param frame_ranges: String specifying frame ranges to process
    :param conf: Confidence threshold
    :param iou: IoU threshold
    :param model_type: Type of model (YOLO, or RTDETR)
    :param model_weights: Path to model weights file
    :param progress: Gradio progress bar
    :return: Status message
    """
    try:
        if remember_token:
            os.environ['TATOR_TOKEN'] = str(token)

        tator_operator = TatorOperator(token, project_id, media_id)

        config = {
            "model_confidence_threshold": float(conf),
            "iou_threshold": float(iou),
            "model_type": str(model_type),
            "model_path": str(model_weights),
        }
        # Initialize the Coral Algorithm
        coral_algo = CoralAlgorithm(config)
        progress(0, "Initializing algorithm...")
        coral_algo.initialize()

        # Parse the frame ranges
        frames_to_process = tator_operator.parse_frame_ranges(frame_ranges)
        total_frames = len(frames_to_process)

        predictions = []

        # Download media
        video_path = tator_operator.download_media(output_dir="temp_videos")

        # Extract frames and perform inference
        for i, frame_idx in enumerate(frames_to_process):
            frame = tator_operator.extract_frame(video_path, frame_idx)
            progress((i + 1 + total_frames) / (3 * total_frames), f"Making predictions for frame {frame_idx}")
            polygons, confs = coral_algo.infer(frame)
            predictions.append([frame_idx, polygons, confs])

        # Upload predictions
        for i, (frame_idx, polygons, confs) in enumerate(predictions):
            progress((i + 1 + 2 * total_frames) / (3 * total_frames), f"Uploading predictions for frame {frame_idx}")
            tator_operator.upload_predictions(frame_idx, polygons, confs)

        # Delete media
        tator_operator.delete_media(video_path)

        gr.Info("Processing completed successfully!")
        return "Done."

    except Exception as e:
        error_msg = f"ERROR: Failed to complete inference!\n{e}\n{traceback.format_exc()}"
        gr.Error(error_msg)
        return error_msg


def launch_gui():
    """
    Launches the Benthic-Mapping GUI.

    :return: None
    """
    with gr.Blocks(title="Benthic-Mapping") as app:
        gr.Markdown("# Benthic-Mapping")

        with gr.Tab("RockAlgorithm"):
            with gr.Group():
                gr.Markdown("## Upload Rock Predictions")
                gr.Markdown("Enter your Tator API token here. This is required for authentication.")
                token = gr.Textbox(label="Token", type="password", value=os.getenv("TATOR_TOKEN"))

                gr.Markdown("Check this box to save your token for future use.")
                remember_token = gr.Checkbox(label="Remember Token", value=True)

                with gr.Row():
                    with gr.Column():
                        gr.Markdown("Enter the ID of the Tator project you're working on.")
                        project_id = gr.Number(label="Project ID", value=155)

                        gr.Markdown("Use commas to separate ranges, dashes for inclusive ranges, "
                                    "and single numbers for individual frames: 25-30, 45, 50")
                        frame_ranges = gr.Textbox(label="Frame Ranges")

                    with gr.Column():
                        gr.Markdown("Enter the ID of the media file you want to process.")
                        media_id = gr.Number(label="Media ID", value=None)

            with gr.Group():
                gr.Markdown("## Model Parameters")

                with gr.Row():
                    with gr.Column():
                        gr.Markdown("Higher values mean stricter detection.")
                        conf = gr.Slider(label="Confidence Threshold", minimum=0, maximum=1, value=0.5)

                        gr.Markdown("Use SAHI to detect smaller instances (slower).")
                        smol = gr.Radio(choices=[True, False], label="SMOL Mode", value=False)

                    with gr.Column():
                        gr.Markdown("Lower values mean less overlap allowed between detections.")
                        iou = gr.Slider(label="IoU Threshold", minimum=0, maximum=1, value=0.7)

                        gr.Markdown("Specify the model architecture, either YOLO or RTDETR.")
                        model_type = gr.Radio(choices=["YOLO", "RTDETR"], value="RTDETR", label="Model Type")

                gr.Markdown("Upload the file containing the trained model weights.")
                model_weights = gr.File(label="Model Weights")

            gr.Markdown("### Model Weights")
            gr.Markdown("[YOLO - 06/26/2024](https://drive.google.com/file/d/1vcsO9rQr0lScHuBLISBR72Xgr1kpYIec/view?usp=drive_link)")
            gr.Markdown("[RTDETR - 06/30/2024](https://drive.google.com/file/d/1qotY6xEF5Y3cOknseGROEqtpUa3AnVZ2/view?usp=drive_link)")

            run_button = gr.Button("Run")

            gr.Markdown("The results and any messages from the algorithm will be displayed here.")
            output = gr.Textbox(label="Output", lines=10)

            run_button.click(
                run_rock_algorithm,
                inputs=[token,
                        remember_token,
                        project_id,
                        media_id,
                        frame_ranges,
                        conf,
                        iou,
                        smol,
                        model_type,
                        model_weights],
                outputs=output
            )

        with gr.Tab("CoralAlgorithm"):
            with gr.Group():
                gr.Markdown("## Upload Coral Predictions")
                gr.Markdown("Enter your Tator API token here. This is required for authentication.")
                token = gr.Textbox(label="Token", type="password", value=os.getenv("TATOR_TOKEN"))

                gr.Markdown("Check this box to save your token for future use.")
                remember_token = gr.Checkbox(label="Remember Token", value=True)

                with gr.Row():
                    with gr.Column():
                        gr.Markdown("Enter the ID of the Tator project you're working on.")
                        project_id = gr.Number(label="Project ID", value=70)

                        gr.Markdown("Use commas to separate ranges, dashes for inclusive ranges, "
                                    "and single numbers for individual frames: 25-30, 45, 50")
                        frame_ranges = gr.Textbox(label="Frame Ranges")

                    with gr.Column():
                        gr.Markdown("Enter the ID of the media file you want to process.")
                        media_id = gr.Number(label="Media ID", value=None)

            with gr.Group():
                gr.Markdown("## Model Parameters")

                with gr.Row():
                    with gr.Column():
                        gr.Markdown("Higher values mean stricter detection.")
                        conf = gr.Slider(label="Confidence Threshold", minimum=0, maximum=1, value=0.5)

                    with gr.Column():
                        gr.Markdown("Lower values mean less overlap allowed between detections.")
                        iou = gr.Slider(label="IoU Threshold", minimum=0, maximum=1, value=0.7)

                        gr.Markdown("Specify the model architecture, either YOLO or RTDETR.")
                        model_type = gr.Radio(choices=["YOLO", "RTDETR"], value="RTDETR", label="Model Type")

                gr.Markdown("Upload the file containing the trained model weights.")
                model_weights = gr.File(label="Model Weights")

            gr.Markdown("### Model Weights")
            gr.Markdown("[RTDETR - 07/10/2024](https://drive.google.com/file/d/1PQFi6a1hOASMs1LTn2I3_-2mrk0qMeLw/view?usp=drive_link)")
            gr.Markdown("[RTDETR - 09/26/2024](https://drive.google.com/file/d/12zxtizsQhwINYMlm0C-hc6gFKAAbG1h5/view?usp=drive_link)")

            run_button = gr.Button("Run")

            gr.Markdown("The results and any messages from the algorithm will be displayed here.")
            output = gr.Textbox(label="Output", lines=10)

            run_button.click(
                run_coral_algorithm,
                inputs=[token,
                        remember_token,
                        project_id,
                        media_id,
                        frame_ranges,
                        conf,
                        iou,
                        model_type,
                        model_weights],
                outputs=output
            )

    app.launch(share=False)


if __name__ == "__main__":
    launch_gui()
