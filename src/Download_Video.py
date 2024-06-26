import os
import argparse
import traceback
import subprocess
import tator
from typing import List


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------

class VideoDownloader:
    def __init__(self, api_token: str, project_id: int, output_dir: str):
        """

        :param api_token:
        :param project_id:
        :param output_dir:
        """
        self.api_token = api_token
        self.project_id = project_id
        self.api = self._authenticate()

        self.root = output_dir
        self.raw_video_dir = f"{self.root}/Raw_Videos"
        self.converted_video_dir = f"{self.root}/Converted_Videos"

    def _authenticate(self):
        """
        Authenticate with Tator API

        :return:
        """
        try:
            api = tator.get_api(host='https://cloud.tator.io', token=self.api_token)
            print(f"NOTE: Authentication successful for {api.whoami().username}")
            return api
        except Exception as e:
            raise Exception(f"ERROR: Could not authenticate with provided API Token\n{e}")

    @staticmethod
    def convert_video(input_file: str, output_dir: str):
        """
        Convert the video file to a specific, and consistent format

        :param input_file:
        :param output_dir:
        :return:
        """
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, os.path.splitext(os.path.basename(input_file))[0] + '.mp4')

        if os.path.exists(output_file):
            return

        ffmpeg_cmd = [
            'ffmpeg',
            '-i', input_file,
            '-c:v', 'libx264',
            '-crf', '23',
            '-c:a', 'aac',
            '-strict', 'experimental',
            output_file
        ]

        try:
            subprocess.run(ffmpeg_cmd, check=True)
            print(f'Successfully converted {input_file} to {output_file}')
        except subprocess.CalledProcessError as e:
            print(f'Error converting {input_file}: {e}')

    def download_and_convert(self, media_ids: List[str]):
        """
        Download and convert a set of video files

        :param media_ids:
        :return:
        """
        if not self.project_id:
            raise ValueError("ERROR: Project ID provided is invalid; please check input")

        if not media_ids:
            raise ValueError("ERROR: Media IDs provided are invalid; please check input")

        os.makedirs(self.raw_video_dir, exist_ok=True)
        os.makedirs(self.converted_video_dir, exist_ok=True)

        for media_id in media_ids:
            try:
                media = self.api.get_media(media_id)
                media_name, ext = media.name.split(".")
                media_name = media_name.replace(":", "_")
                output_video_path = f"{self.raw_video_dir}/{media_name}_converted.{ext}"
                print(f"NOTE: Downloading {media.name}...")

                for progress in tator.util.download_media(self.api, media, output_video_path):
                    print(f"NOTE: Download progress: {progress}%")

                if os.path.exists(output_video_path):
                    print(f"NOTE: Media {media.name} downloaded successfully")
                    self.convert_video(output_video_path, self.converted_video_dir)
                else:
                    print(f"ERROR: Media {media.name} did not download successfully; skipping")

            except Exception as e:
                print(f"ERROR: Could not process media {media_id}: {e}")


# ----------------------------------------------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------------------------------------------

def main():

    parser = argparse.ArgumentParser(description="Download and convert videos from TATOR")

    parser.add_argument("--api_token", type=str,
                        default=os.getenv('TATOR_TOKEN'),
                        help="Tator API Token")

    parser.add_argument("--project_id", type=int, default=155,
                        help="Project ID for desired media")

    parser.add_argument("--media_ids", type=int, nargs='+',
                        help="ID for desired media(s)")

    parser.add_argument("--output_dir", type=str,
                        default=os.path.dirname(os.path.dirname(os.path.realpath(__file__))) + "/Data",
                        help="ID for desired media(s)")

    args = parser.parse_args()

    try:
        downloader = VideoDownloader(args.api_token, args.project_id, args.output_dir)
        downloader.download_and_convert(args.media_ids)
        print("Done.")
    except Exception as e:
        print(f"ERROR: Could not finish process.\n{e}")
        print(traceback.format_exc())


if __name__ == "__main__":
    main()
