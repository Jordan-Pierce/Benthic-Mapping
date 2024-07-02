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

def run_rock_algorithm(token, remember_token, project_id, media_id, start_at, end_at, conf, iou, model_weights):
    """

    :param token:
    :param remember_token:
    :param project_id:
    :param media_id:
    :param start_at:
    :param end_at:
    :param conf:
    :param iou:
    :param model_weights:
    :return:
    """
    try:
        if remember_token:
            os.environ['TATOR_TOKEN'] = str(token)

        tator_operator = TatorOperator(token, project_id, media_id, start_at, end_at)

        config = {
            "model_confidence_threshold": float(conf),
            "iou_threshold": float(iou),
            "smol": False,
            "model_type": "yolov10",
            "model_path": model_weights,
            "sam_model_path": "sam_l.pt"
        }

        rock_algo = RockAlgorithm(config)
        rock_algo.initialize()

        for i, frame_idx in enumerate(range(start_at, end_at + 1)):
            frame = tator_operator.download_frame(frame_idx)
            predictions = rock_algo.infer(frame)
            tator_operator.upload_predictions(frame_idx, predictions)

        return "Done."

    except Exception as e:
        error_msg = f"ERROR: Failed to complete inference!\n{e}\n{traceback.format_exc()}"
        return error_msg


def launch_gui():
    """

    :return:
    """
    with gr.Blocks(title="Benthic-Mapping") as app:
        gr.Markdown("# Benthic-Mapping")

        with gr.Tab("RockAlgorithm"):
            with gr.Group():
                gr.Markdown("## Upload Rock Predictions")
                token = gr.Textbox(label="Token", type="password", value=os.getenv("TATOR_TOKEN"))
                remember_token = gr.Checkbox(label="Remember Token", value=True)
                project_id = gr.Number(label="Project ID", value=155)
                media_id = gr.Number(label="Media ID")
                start_at = gr.Number(label="Start Frame")
                end_at = gr.Number(label="End Frame")

            with gr.Group():
                gr.Markdown("## Model Parameters")
                conf = gr.Slider(label="Confidence Threshold", minimum=0, maximum=1, value=0.5)
                iou = gr.Slider(label="IoU Threshold", minimum=0, maximum=1, value=0.7)
                model_weights = gr.File(label="Model Weights")

            run_button = gr.Button("Run")
            output = gr.Textbox(label="Output", lines=10)

        run_button.click(
            run_rock_algorithm,
            inputs=[token, remember_token, project_id, media_id, start_at, end_at, conf, iou, model_weights],
            outputs=output
        )

    app.queue()
    app.launch()


if __name__ == "__main__":
    launch_gui()
