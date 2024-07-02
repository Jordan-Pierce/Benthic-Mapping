import os
import traceback
from gooey import Gooey, GooeyParser

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
# Main
# ----------------------------------------------------------------------------------------------------------------------

@Gooey(dump_build_config=True,
       program_name="Benthic-Mapping",
       default_size=(900, 600),  # width, height
       console=True,
       richtext_controls=True,
       progress_regex=r"^progress: (?P<current>\d+)/(?P<total>\d+)$",
       progress_expr="current / total * 100",
       hide_progress_msg=True,
       timing_options={
           'show_time_remaining': True,
           'hide_time_remaining_on_complete': True,
       })
def main():
    """

    :return:
    """
    parser = GooeyParser(description="")

    parser.add_argument('--verbose', help='be verbose', dest='verbose', action='store_true', default=False)
    subs = parser.add_subparsers(help='commands', dest='command')
    upload_parser = subs.add_parser('RockAlgorithm')

    upload_parser_panel_1 = upload_parser.add_argument_group('Upload Rock Predictions',
                                                             'Provide your TATOR API Token, and specify the '
                                                             'Media (by ID) and frames to make segments for',
                                                             gooey_options={'show_border': True})

    upload_parser_panel_1.add_argument("--token", required=True, type=str,
                                       default=os.getenv("TATOR_TOKEN"),
                                       metavar='Token',
                                       help="TATOR API Token",
                                       widget="PasswordField")

    upload_parser_panel_1.add_argument('--remember_token', action="store_false",
                                       metavar="Remember Token",
                                       help='Store Token as a Local Environmental Variable',
                                       widget="BlockCheckbox")

    upload_parser_panel_1.add_argument("--project_id", required=True, type=int,
                                       metavar='Project ID',
                                       default=155,
                                       help="ID for the Project")

    upload_parser_panel_1.add_argument("--media_id", required=True, type=int,
                                       metavar='Media ID',
                                       help="ID for the Media (video)")

    upload_parser_panel_1.add_argument("--start_at", required=True, type=int,
                                       metavar='Start Frame',
                                       help="Starting Frame")

    upload_parser_panel_1.add_argument("--end_at", required=True, type=int,
                                       metavar='End Frame',
                                       help="Ending Frame (inclusive)")

    upload_parser_panel_2 = upload_parser.add_argument_group('Model Parameters',
                                                             'Provide the path to the model (.pth) file',
                                                             gooey_options={'show_border': True})

    upload_parser_panel_2.add_argument("--conf", type=float, default=0.5,
                                       metavar='Confidence Threshold',
                                       help="Higher values filter less confident predictions; [0. - 1.]")

    upload_parser_panel_2.add_argument("--iou", type=float, default=0.7,
                                       metavar='IoU Threshold',
                                       help="Lower values filter overlapping predictions; [0. - 1.]")

    upload_parser_panel_2.add_argument("--model_weights", required=True,
                                       metavar='Model Weights',
                                       help="Path to Model Weights (.pt)",
                                       widget="FileChooser")

    args = upload_parser.parse_args()

    try:

        if not args.remember_token:
            os.environ['TATOR_TOKEN'] = str(args.token)

        # Initialize the TATOR operator given user criteria
        tator_operator = TatorOperator(args.token,
                                       args.project_id,
                                       args.media_id,
                                       args.start_at,
                                       args.end_at)

        # Create the config dict
        config = {
            "model_confidence_threshold": float(args.conf),
            "iou_threshold": float(args.iou),
            "smol": False,
            "model_type": "yolov10",
            "model_path": args.model_weights,
            "sam_model_path": "sam_l.pt"
        }

        # Initialize the rock algorithm
        rock_algo = RockAlgorithm(config)
        rock_algo.initialize()

        # Loop through each frame
        for i, frame_idx in enumerate(range(int(args.start_at), int(args.end_at) + 1, 1)):
            print(f"NOTE: Making predictions on frame {frame_idx}")
            # Download the frame, get numpy array
            frame = tator_operator.download_frame(frame_idx)
            # Pass to rock algorithm
            predictions = rock_algo.infer(frame)
            # Upload to tator
            tator_operator.upload_predictions(frame_idx, predictions)
            # Update Gooey progress bar for user
            print(f"progress: {i + 1}/{(args.end_at + 1) - args.start_at}")

        print("Done.")

    except Exception as e:
        print(f"ERROR: Failed to complete inference!\n{e}")
        print(traceback.format_exc())


if __name__ == "__main__":
    main()
