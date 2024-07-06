import os
import traceback

import gradio as gr
import tator
import cv2

from Rocks.rock_algorithm import RockAlgorithm


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------

class TatorOperator:
    def __init__(self, token, project_id, media_id, start_at, end_at):
        """

        :param token:
        :param project_id:
        :param media_id:
        :param start_at:
        :param end_at:
        """
        self.token = token
        self.project_id = project_id
        self.media_id = media_id
        self.start_at = start_at
        self.end_at = end_at

        try:
            self.api = tator.get_api(host='https://cloud.tator.io', token=token)
            self.version_type = self.api.whoami().id
            self.media = self.api.get_media(media_id)
            print(f"NOTE: Authentication successful for {self.api.whoami().email}")

        except Exception as e:
            raise Exception(f"ERROR: Could not connect to TATOR.\n{e}")

    def upload_predictions(self, frame_idx, polygon_points):
        """

        :param polygon_points:
        :param frame_id:
        :return:
        """
        try:
            localizations = []

            # Add each of the polygon points to the localization list
            for points in polygon_points:
                # Specify spec
                spec = {
                    'type': 460,                      # rock poly type
                    'media_id': self.media_id,
                    'version_id': 546,                # rock anno ver
                    'points': points,
                    'frame': frame_idx,
                    'attributes': {"Label": "Rock"},
                }
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


# ----------------------------------------------------------------------------------------------------------------------
# Gradio
# ----------------------------------------------------------------------------------------------------------------------

def parse_frame_ranges(frame_ranges_str):
    """
    Parse the frame ranges string into a list of frame numbers.

    :param frame_ranges_str:
    :return:
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


def run_rock_algorithm(token,
                       remember_token,
                       project_id,
                       media_id,
                       frame_ranges,
                       conf,
                       iou,
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
            "smol": False,
            "model_type": "yolov10",
            "model_path": model_weights,
            "sam_model_path": "sam_l.pt"
        }

        rock_algo = RockAlgorithm(config)
        progress(0, "Initializing algorithm...")
        rock_algo.initialize()

        frames_to_process = parse_frame_ranges(frame_ranges)
        total_frames = len(frames_to_process)

        for i, frame_idx in enumerate(frames_to_process):
            progress((i + 1) / total_frames, f"Processing frame {frame_idx}")
            progress((i + 1) / total_frames, f"Downloading frame {frame_idx}")
            frame = tator_operator.download_frame(frame_idx)
            progress((i + 1) / total_frames, f"Making predictions for frame {frame_idx}")
            predictions = rock_algo.infer(frame)
            progress((i + 1) / total_frames, f"Uploading predictions for frame {frame_idx}")
            tator_operator.upload_predictions(frame_idx, predictions)

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
                        media_id = gr.Number(label="Media ID")

            with gr.Group():
                gr.Markdown("## Model Parameters")
                gr.Markdown("Set the confidence threshold for object detection. Higher values mean stricter detection.")
                conf = gr.Slider(label="Confidence Threshold", minimum=0, maximum=1, value=0.5)

                gr.Markdown("Set the Intersection over Union threshold for object detection. "
                            "Higher values mean less overlap allowed between detections.")
                iou = gr.Slider(label="IoU Threshold", minimum=0, maximum=1, value=0.7)

                gr.Markdown("Upload the file containing the trained model weights.")
                model_weights = gr.File(label="Model Weights")

            run_button = gr.Button("Run")

            gr.Markdown("The results and any messages from the algorithm will be displayed here.")
            output = gr.Textbox(label="Output", lines=10)

        run_button.click(
            run_rock_algorithm,
            inputs=[token, remember_token, project_id, media_id, frame_ranges, conf, iou, model_weights],
            outputs=output
        )

    app.launch(share=True)


if __name__ == "__main__":
    launch_gui()
