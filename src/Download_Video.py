import os
import sys
import argparse
import traceback
import subprocess

import tator


# ----------------------------------------------------------------------------------------------------------------------
# Functions
# ----------------------------------------------------------------------------------------------------------------------

def convert_videos(input_files, output_dir):
    """

    :param input_files:
    :param output_dir:
    :return:
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    for input_file in input_files:

        # Output file path
        output_file = os.path.join(output_dir, os.path.splitext(os.path.basename(input_file))[0] + '.mp4')

        # If the file already exists, skip it
        if os.path.exists(output_file):
            continue

        # FFmpeg command to convert video
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
            # Run FFmpeg command
            subprocess.run(ffmpeg_cmd, check=True)
            print(f'Successfully converted {input_file} to {output_file}')
        except subprocess.CalledProcessError as e:
            print(f'Error converting {input_file}: {e}')


def download_video(api_token, project_id, media_ids):
    """

    :param api_token:
    :param project_id:
    :param media_ids:
    :return:
    """
    try:
        # Setting the api given the token, authentication
        token = api_token
        api = tator.get_api(host='https://cloud.tator.io', token=token)
        print(f"NOTE: Authentication successful for {api.whoami().username}")

    except Exception as e:
        print(f"ERROR: Could not authenticate with provided API Token\n{e}")
        sys.exit(1)

    # Project id containing media
    if not project_id:
        print(f"ERROR: Project ID provided is invalid; please check input")
        sys.exit(1)

    # List of media
    if not len(media_ids):
        print(f"ERROR: Medias provided is invalid; please check input")
        sys.exit(1)

    # Get the root data directory (Data); OCD
    root = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) + "\\Data"
    root = root.replace("\\", "/")

    # Raw video location (from TATOR)
    raw_video_dir = f"{root}/Raw_Videos"
    os.makedirs(raw_video_dir, exist_ok=True)

    # Converted videos
    converted_video_dir = f"{root}/Converted_Videos"
    os.makedirs(converted_video_dir, exist_ok=True)

    for media_id in media_ids:

        # ------------------------------------------------
        # Download media
        # ------------------------------------------------
        try:
            # Get the video handler
            media = api.get_media(media_id)
            media_name, ext = media.name.split(".")
            media_name = media_name.replace(":", "_")
            output_video_path = f"{raw_video_dir}/{media_name}_converted.{ext}"
            os.makedirs(raw_video_dir, exist_ok=True)
            print(f"NOTE: Downloading {media.name}...")

        except Exception as e:
            print(f"ERROR: Could not get media from {media_id}")
            sys.exit(1)

        # Download the video
        for progress in tator.util.download_media(api, media, output_video_path):
            print(f"NOTE: Download progress: {progress}%")

        # Check that video was downloaded
        if os.path.exists(output_video_path):
            print(f"NOTE: Media {media.name} downloaded successfully")
        else:
            print(f"ERROR: Media {media.name} did not download successfully; skipping")
            continue

        # Convert the videos to MP4 readable files
        convert_videos([output_video_path], converted_video_dir)


# ----------------------------------------------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------------------------------------------

def main():
    """

    """

    parser = argparse.ArgumentParser(description="Download video from TATOR")

    parser.add_argument("--api_token", type=str,
                        default=os.getenv('TATOR_TOKEN'),
                        help="Tator API Token")

    parser.add_argument("--project_id", type=int, default=70,
                        help="Project ID for desired media")

    parser.add_argument("--media_ids", type=int, nargs='+',
                        help="ID for desired media(s)")

    args = parser.parse_args()

    try:
        download_video(api_token=args.api_token,
                       project_id=args.project_id,
                       media_ids=args.media_ids)

        print("Done.")

    except Exception as e:
        print(f"ERROR: {e}")
        print(traceback.format_exc())


if __name__ == "__main__":
    main()
