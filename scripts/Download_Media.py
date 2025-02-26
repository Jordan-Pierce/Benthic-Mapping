import os
import sys
import argparse
import traceback
import subprocess
import tator
from typing import List, Optional, Tuple

try:
    from dotenv import load_dotenv
    load_dotenv()

    assert os.getenv("TATOR_TOKEN"), print("Error: Missing Tator Token API")
except Exception as e:
    print(e)
    sys.exit()

# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------

class VideoDownloader:
    def __init__(self, api_token: str, project_id: int, output_dir: str):
        """
        :param api_token: API token used for authentication
        :param project_id: The project identifier
        :param output_dir: The root directory for all outputs.
        """
        self.api_token = api_token
        self.project_id = project_id
        self.api = self._authenticate()

        self.root = output_dir
        self.raw_video_dir = os.path.join(self.root, "Raw_Videos")
        self.converted_video_dir = os.path.join(self.root, "Converted_Videos")
        self.extracted_frames_dir = os.path.join(self.root, "Extracted_Frames")

    def _authenticate(self):
        """
        Authenticate with Tator API
        :return: API client instance
        """
        try:
            api = tator.get_api(host='https://cloud.tator.io', token=self.api_token)
            print(f"NOTE: Authentication successful for {api.whoami().username}")
            return api
        except Exception as e:
            raise Exception(f"ERROR: Could not authenticate with provided API Token\n{e}")

    @staticmethod
    def convert_video(input_file: str, output_dir: str) -> Optional[str]:
        """
        Convert the video file to a specific, consistent format
        :param input_file: Path to the input file
        :param output_dir: Output directory to store converted video
        :return: Path to the converted video file, or None if failed
        """
        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(input_file))[0]
        output_file = os.path.join(output_dir, f"{base_name}.mp4")

        if os.path.exists(output_file):
            return output_file

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
            output_file = None

        return output_file

    def extract_frames(self, video_file: str, every_n_seconds: float = 1.0, crop_region: Optional[Tuple[int, int, int, int]] = None) -> str:
        """
        Extracts (and optionally crops) frames from a video file at specified intervals.
        The extracted frames are saved as JPEG images in a subfolder within
        the Extracted_Frames directory, where the subfolder is derived from the video file's name.
        
        :param video_file: The path to the video file.
        :param every_n_seconds: Number of seconds between frame extractions (default: 1.0).
        :param crop_region: Optional tuple (width, height, x, y) specifying the region to crop.
        :return: The directory where the frames have been saved.
        """
        # Create a subfolder in Extracted_Frames using the video file's base name
        base_name = os.path.splitext(os.path.basename(video_file))[0]
        output_subfolder = os.path.join(self.extracted_frames_dir, base_name)
        os.makedirs(output_subfolder, exist_ok=True)
    
        # Build the filter parameters
        # Set the frame extraction rate based on the interval
        fps = f"fps=1/{every_n_seconds}"  # This creates the specified interval between frames
        vf_filters = fps
    
        if crop_region:
            width, height, x, y = crop_region
            vf_filters += f",crop={width}:{height}:{x}:{y}"
    
        output_pattern = os.path.join(output_subfolder, "frame_%04d.jpg")
        ffmpeg_cmd = [
            'ffmpeg',
            '-i', video_file,
            '-vf', vf_filters,
            '-qscale:v', '2',
            output_pattern
        ]
    
        try:
            subprocess.run(ffmpeg_cmd, check=True)
            print(f"Frames extracted successfully to {output_subfolder}")
        except subprocess.CalledProcessError as e:
            print(f"Error extracting frames from {video_file}: {e}")
        
        return output_subfolder

    def download(self, media_ids: List[str], convert: bool = False, extract: bool = False,
                 every_n_seconds: float = 1.0, crop_region: Optional[Tuple[int, int, int, int]] = None) -> List[str]:
        """
        Download video files, optionally converting them and extracting frames.
        
        :param media_ids: A list of media IDs to download.
        :param convert: Boolean flag for video conversion.
        :param extract: Boolean flag for extracting frames.
        :param crop_region: Optional tuple (width, height, x, y) for cropping frames.
        :return: List of paths to the processed video files.
        """
        if not self.project_id:
            raise ValueError("ERROR: Project ID provided is invalid; please check input")

        if not media_ids:
            raise ValueError("ERROR: Media IDs provided are invalid; please check input")

        os.makedirs(self.raw_video_dir, exist_ok=True)
        
        video_paths = []

        for media_id in media_ids:
            try:
                media = self.api.get_media(media_id)
                # Assume media.name includes an extension
                media_name, ext = media.name.split(".")
                media_name = media_name.replace(":", "_")
                output_video_path = os.path.join(self.raw_video_dir, f"{media_name}_converted.{ext}")
                print(f"NOTE: Downloading {media.name}...")

                for progress in tator.util.download_media(self.api,
                                                          media,
                                                          output_video_path,
                                                          media.height,
                                                          "streaming"):
                    print(f"NOTE: Download progress: {progress}%")

                if os.path.exists(output_video_path):
                    print(f"NOTE: Media {media.name} downloaded successfully")
                    if convert:
                        converted_path = self.convert_video(output_video_path, self.converted_video_dir)
                        if converted_path:
                            output_video_path = converted_path
                    if extract:
                        self.extract_frames(output_video_path, every_n_seconds, crop_region)
                else:
                    print(f"ERROR: Media {media.name} did not download successfully; skipping")

                video_paths.append(output_video_path)

            except Exception as e:
                print(f"ERROR: Could not process media {media_id}: {e}")

        return video_paths

# ----------------------------------------------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Download, convert videos and extract frames from TATOR",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Download videos using media IDs
    python video_downloader.py --media_ids 1234 5678
    
    # Download and convert videos to MP4
    python video_downloader.py --media_ids 1234 5678 --convert
    
    # Download, convert, and extract frames every 2 seconds
    python video_downloader.py --media_ids 1234 --convert --extract --frame_interval 2.0
    
    # Download, convert, and extract cropped frames
    python video_downloader.py --media_ids 1234 --convert --extract --crop 1920 1080 0 0
    """
    )

    # Authentication and Project Settings
    auth_group = parser.add_argument_group('Authentication and Project Settings')
    auth_group.add_argument(
        "--api_token", 
        type=str,
        default=os.getenv('TATOR_TOKEN'),
        help="Tator API Token (default: environment variable TATOR_TOKEN)"
    )
    auth_group.add_argument(
        "--project_id", 
        type=int, 
        default=155,
        help="Project ID for desired media (default: 155)"
    )

    # Media Selection
    media_group = parser.add_argument_group('Media Selection')
    media_group.add_argument(
        "--media_ids", 
        type=int, 
        nargs='+',
        required=True,
        help="One or more media IDs to download"
    )

    # Output Settings
    output_group = parser.add_argument_group('Output Settings')
    output_group.add_argument(
        "--output_dir", 
        type=str,
        default=os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "data"),
        help="Root directory for output files (default: ../data)"
    )

    # Processing Options
    process_group = parser.add_argument_group('Processing Options')
    process_group.add_argument(
        "--convert", 
        action="store_true",
        help="Convert downloaded videos to MP4 format"
    )
    process_group.add_argument(
        "--extract", 
        action="store_true",
        help="Extract frames from videos"
    )
    process_group.add_argument(
        "--every_n_seconds", 
        type=float,
        default=1.0,
        help="Interval in seconds between extracted frames (default: 1.0)"
    )
    process_group.add_argument(
        "--crop", 
        type=int,
        nargs=4,
        metavar=('WIDTH', 'HEIGHT', 'X', 'Y'),
        help="Crop region for extracted frames: WIDTH HEIGHT X Y"
    )

    args = parser.parse_args()

    try:
        downloader = VideoDownloader(args.api_token, args.project_id, args.output_dir)
        downloader.download(
            args.media_ids, 
            convert=args.convert,
            extract=args.extract,
            every_n_seconds=args.every_n_seconds,
            crop_region=args.crop if args.crop else None
        )
        print("Done.")
    except Exception as e:
        print(f"ERROR: Could not finish process.\n{e}")
        print(traceback.format_exc())

if __name__ == "__main__":
    main()