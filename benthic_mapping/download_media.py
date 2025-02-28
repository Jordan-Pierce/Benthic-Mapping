import os
import argparse
import traceback
import subprocess
from typing import List, Optional, Tuple

import tator
import multiprocessing

from concurrent.futures import ProcessPoolExecutor


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class MediaDownloader:
    def __init__(self, api_token: str, project_id: int, output_dir: str):
        """
        :param api_token: API token used for authentication
        :param project_id: The project identifier
        :param output_dir: The output_dir directory for all outputs.
        """
        self.api_token = api_token
        self.project_id = project_id
        self.api = self._authenticate()

        self.output_dir = output_dir
        self.raw_video_dir = os.path.join(self.output_dir, "Raw_Videos")
        self.converted_video_dir = os.path.join(self.output_dir, "Converted_Videos")
        self.extracted_frames_dir = os.path.join(self.output_dir, "Extracted_Frames")
                
        # Create directories
        os.makedirs(self.raw_video_dir, exist_ok=True)
        os.makedirs(self.converted_video_dir, exist_ok=True)
        os.makedirs(self.extracted_frames_dir, exist_ok=True)
        
        # Track paths to all processed files
        self.original_video_paths = []  # Paths to downloaded original videos
        self.converted_video_paths = []  # Paths to converted videos
        self.extracted_frame_dirs = []  # Paths to directories containing extracted frames
        self.media_path_map = {}  # Maps media_id to both original and converted paths

    def _authenticate(self):
        """
        Authenticate with Tator API
        :return: API client instance
        """
        try:
            api = tator.get_api(
                host='https://cloud.tator.io', token=self.api_token)
            print(
                f"NOTE: Authentication successful for {api.whoami().username}")
            return api
        except Exception as e:
            raise Exception(
                f"ERROR: Could not authenticate with provided API Token\n{e}")

    @staticmethod
    def convert_video(input_file: str, output_dir: str) -> Optional[str]:
        """
        Convert the video file to a specific, consistent format
        :param input_file: Path to the input file
        :param output_dir: Output directory to store converted video
        :return: Path to the converted video file, or None if failed
        """
        # Check to make sure ffmpeg is installed
        try:
            subprocess.run(['ffmpeg', '-version'], check=True)
        except FileNotFoundError:
            raise FileNotFoundError("ERROR: ffmpeg is not installed; please install ffmpeg before continuing")
        
        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(input_file))[0]
        output_file = os.path.join(output_dir, f"{base_name}.mp4")
        
        print(f"NOTE: Converting {input_file} to MP4...")

        if os.path.exists(output_file):
            # Skip conversion if the file already exists
            if output_file not in self.converted_video_paths:
                self.converted_video_paths.append(output_file)
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
            # Track the converted video path
            if output_file not in self.converted_video_paths:
                self.converted_video_paths.append(output_file)
        except subprocess.CalledProcessError as e:
            print(f'Error converting {input_file}: {e}')
            output_file = None

        return output_file

    @staticmethod
    def _extract_single_frame(args):
        timestamp, frame_number, video_file, output_subfolder, crop_region = args
        output_file = os.path.join(output_subfolder, f"frame_{frame_number:04d}.jpg")
        
        # Build filter for optional cropping
        vf_filter = ""
        if crop_region:
            width, height, x, y = crop_region
            vf_filter = f"-vf crop={width}:{height}:{x}:{y}"
        
        cmd = [
            'ffmpeg',
            '-ss', str(timestamp),
            '-i', video_file,
            '-frames:v', '1',
            *([vf_filter] if vf_filter else []),
            '-q:v', '2',
            '-y',
            output_file
        ]
        
        # Remove empty elements
        cmd = [part for part in cmd if part]
        
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return True
        except subprocess.CalledProcessError:
            print(f"Error extracting frame at {timestamp}s")
            return False

    def extract_frames(self, video_file: str, every_n_seconds: float = 1.0, 
                       crop_region: Optional[Tuple[int, int, int, int]] = None) -> str:
        """
        Extracts (and optionally crops) frames from a video file at specified intervals using multiprocessing.
        Each worker extracts a single frame to maximize parallelism.
        
        :param video_file: The path to the video file.
        :param every_n_seconds: Number of seconds between frame extractions (default: 1.0).
        :param crop_region: Optional tuple (width, height, x, y) specifying the region to crop.
        :return: The directory where the frames have been saved.
        """
        
        # Create a subfolder in Extracted_Frames using the video file's base name
        base_name = os.path.splitext(os.path.basename(video_file))[0]
        output_subfolder = os.path.join(self.extracted_frames_dir, base_name)
        os.makedirs(output_subfolder, exist_ok=True)
        
        # Get video duration using ffprobe
        duration_cmd = [
            'ffprobe', 
            '-v', 'error', 
            '-show_entries', 'format=duration', 
            '-of', 'default=noprint_wrappers=1:nokey=1', 
            video_file
        ]
        
        try:
            duration = float(subprocess.check_output(duration_cmd).decode().strip())
            print(f"Video duration: {duration} seconds")
            
            # Calculate frame timestamps
            timestamps = [i for i in range(0, int(duration), int(every_n_seconds))]
            
            # Prepare arguments for parallel processing
            tasks = [
                (timestamp, i+1, video_file, output_subfolder, crop_region) 
                for i, timestamp in enumerate(timestamps)
            ]
            
            # Use multiprocessing to extract frames in parallel
            max_workers = min(multiprocessing.cpu_count(), len(timestamps))
            print(f"Extracting {len(timestamps)} frames using {max_workers} workers...")
            
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                results = list(executor.map(self._extract_single_frame, tasks))
            
            print(f"Successfully extracted {sum(results)} frames to {output_subfolder}")
            
            # Track this extracted frames directory
            if output_subfolder not in self.extracted_frame_dirs:
                self.extracted_frame_dirs.append(output_subfolder)
            
        except subprocess.CalledProcessError as e:
            print(f"Error processing video {video_file}: {e}")
        
        return output_subfolder
    
    def download_data(self, media_ids: List[str], convert: bool = False, extract: bool = False,
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
            raise ValueError(
                "ERROR: Project ID provided is invalid; please check input")

        if not media_ids:
            raise ValueError(
                "ERROR: Media IDs provided are invalid; please check input")

        os.makedirs(self.raw_video_dir, exist_ok=True)

        video_paths = []

        for media_id in media_ids:
            try:
                media = self.api.get_media(media_id)
                # Assume media.name includes an extension
                media_name, ext = media.name.split(".")
                media_name = media_name.replace(":", "_")
                output_video_path = os.path.join(self.raw_video_dir, f"{media_name}_converted.{ext}")
                
                if not os.path.exists(output_video_path):
                    print(f"NOTE: Downloading {media.name}...")
                    for progress in tator.util.download_media(self.api,
                                                              media,
                                                              output_video_path,
                                                              media.height,
                                                              "streaming"):
                        print(f"NOTE: Download progress: {progress}%")

                if os.path.exists(output_video_path):
                    print(f"NOTE: Media {media.name} downloaded successfully to {output_video_path}")
                    
                    # Track the original video path
                    if output_video_path not in self.original_video_paths:
                        self.original_video_paths.append(output_video_path)
                    
                    # Initialize media path tracking
                    if media_id not in self.media_path_map:
                        self.media_path_map[media_id] = {
                            "original": output_video_path, 
                            "converted": None, 
                            "frames": None
                        }
                    else:
                        self.media_path_map[media_id]["original"] = output_video_path
                    
                    if convert:
                        print(f"NOTE: Converting {media.name} to MP4...")
                        converted_path = self.convert_video(output_video_path, self.converted_video_dir)
                        if converted_path:
                            output_video_path = converted_path
                            # Update the conversion path in the media path map
                            self.media_path_map[media_id]["converted"] = converted_path
                            
                    if extract:
                        print(f"NOTE: Extracting frames from {media.name}...")
                        frames_dir = self.extract_frames(output_video_path, every_n_seconds, crop_region)
                        # Update the frames directory in the media path map
                        self.media_path_map[media_id]["frames"] = frames_dir
                else:
                    print(f"ERROR: Media {media.name} did not download successfully; skipping")

                video_paths.append(output_video_path)

            except Exception as e:
                print(f"ERROR: Could not process media {media_id}: {e}")

        return video_paths
    
    def get_all_media_paths(self) -> dict:
        """
        Returns a dictionary with all tracked media paths
        
        :return: Dictionary with lists of original videos, converted videos, and frame directories
        """
        return {
            "original_videos": self.original_video_paths,
            "converted_videos": self.converted_video_paths,
            "frame_directories": self.extracted_frame_dirs,
            "media_map": self.media_path_map
        }
    
    def get_paths_for_media_id(self, media_id: str) -> dict:
        """
        Get all paths associated with a specific media ID
        
        :param media_id: The media ID to query
        :return: Dictionary with paths for that media ID
        """
        if media_id in self.media_path_map:
            return self.media_path_map[media_id]
        return None


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
        python Download_Media.py --media_ids 1234 5678
        
        # Download and convert videos to MP4
        python Download_Media.py --media_ids 1234 5678 --convert
        
        # Download, convert, and extract frames every 2 seconds
        python Download_Media.py --media_ids 1234 --convert --extract --frame_interval 2.0
        
        # Download, convert, and extract cropped frames
        python Download_Media.py --media_ids 1234 --convert --extract --crop 1920 1080 0 0
        """
    )

    # Authentication and Project Settings
    auth_group = parser.add_argument_group(
        'Authentication and Project Settings')
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
        default=os.path.join(os.path.dirname(
            os.path.dirname(os.path.realpath(__file__))), "data"),
        help="output_dir directory for output files (default: ../data)"
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
        # Initialize the video downloader
        downloader = MediaDownloader(args.api_token, 
                                     args.project_id, 
                                     args.output_dir)
        # Download, convert, and extract
        downloader.download_data(
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
