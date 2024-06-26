import os
import shutil
import argparse
import traceback
from tqdm import tqdm
import concurrent.futures
from typing import List, Dict

import tator

import cv2
import numpy as np
import pandas as pd

import supervision as sv
from autodistill import helpers

from Common import get_now
from Common import render_dataset


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------

class DataDownloader:
    def __init__(self, api_token: str, project_id: int, search_string: str, dataset_name: str, output_dir: str):
        """

        :param api_token:
        :param project_id:
        :param search_string:
        :param dataset_name:
        :param output_dir:
        """
        self.api = self._authenticate(api_token)
        self.project_id = project_id
        self.search_string = search_string

        self.dataset_name = dataset_name

        self.root = output_dir
        self.data_dir = f"{self.root}/Data/Training_Data"
        self.dataset_dir = f"{self.data_dir}/{get_now()}_{self.dataset_name}"

        self._setup_directories()

    @staticmethod
    def _authenticate(api_token: str):
        """

        :param api_token:
        :return:
        """
        try:
            api = tator.get_api(host='https://cloud.tator.io', token=api_token)
            print(f"NOTE: Authentication successful for {api.whoami().username}")
            return api
        except Exception as e:
            raise Exception(f"ERROR: Could not authenticate with provided API Token\n{e}")

    def _setup_directories(self):
        """

        :return:
        """
        os.makedirs(f"{self.dataset_dir}/images", exist_ok=True)

    def _save_query(self):
        """
        Write the search string to text file

        :return:
        """
        try:
            file_path = f"{self.dataset_dir}/search_string.txt"
            with open(file_path, 'w') as file:
                file.write(self.search_string)
        except Exception as e:
            print(f"WARNING: An error occurred while writing search string to file:\n{e}")

    def download_frame(self, media, frame_id: int):
        """

        :param media:
        :param frame_id:
        :return:
        """
        frame_path = f"{self.dataset_dir}/images/{str(media.id)}_{str(frame_id)}.jpg"
        if not os.path.exists(frame_path):
            temp = self.api.get_frame(id=media.id, tile=f"{media.width}x{media.height}", frames=[int(frame_id)])
            shutil.move(temp, frame_path)
        return frame_id, frame_path

    def download_frames(self, media, frames: List[int]):
        """

        :param media:
        :param frames:
        :return:
        """
        with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count() - 1) as executor:
            future_to_frame = {executor.submit(self.download_frame, media, frame): frame for frame in frames}
            return [future.result()[1] for future in concurrent.futures.as_completed(future_to_frame)]

    @staticmethod
    def process_query(query):
        """

        :param query:
        :return:
        """
        data = []
        for q in query:

            if 'ScientificName' in q.attributes:
                media = q.media
                frame = q.frame
                frame_name = f"{q.media}_{q.frame}.jpg"
                x = q.x
                y = q.y
                width = q.width
                height = q.height

                try:
                    label = str(q.attributes['ScientificName'])
                except:
                    label = 'Unknown'

                data.append([media, frame_name, frame, x, y, width, height, label])

        return pd.DataFrame(data, columns=['media', 'name', 'frame', 'x', 'y', 'width', 'height', 'label'])

    def process_media(self, media_id: int, data: pd.DataFrame):
        """

        :param media_id:
        :param data:
        :return:
        """
        media = self.api.get_media(id=int(media_id))
        subset = data[data['media'] == media_id].copy()
        subset.loc[:, 'media_width'] = media.width
        subset.loc[:, 'media_height'] = media.height
        frames = self.download_frames(media, subset['frame'].unique())
        return subset, frames

    @staticmethod
    def process_frame(frame_path: str, data: pd.DataFrame, class_to_id: Dict[str, int]):
        """
        Process a single frame and create detections.

        :param frame_path: Path to the frame image
        :param data: DataFrame containing annotation data
        :param class_to_id: Dictionary mapping class names to class IDs
        :return: Tuple of frame name and dict containing image and detection
        """
        name = os.path.basename(frame_path)
        subset = data[data['name'] == name]
        width, height = subset['media_width'].iloc[0], subset['media_height'].iloc[0]

        xyxy = np.column_stack([
            subset['x'].values * width,
            subset['y'].values * height,
            (subset['x'].values + subset['width'].values) * width,
            (subset['y'].values + subset['height'].values) * height
        ])

        class_ids = np.array([class_to_id[label] for label in subset['label']], dtype=int)

        detection = sv.Detections(
            xyxy=xyxy.astype(int),
            class_id=class_ids
        )

        return name, {'image': cv2.imread(frame_path), 'detection': detection}

    def download_labels(self):
        """
        Download and process labels, creating a YOLO-format dataset.

        :return:
        """
        # Query TATOR given the search string
        query = self.api.get_localization_list(project=self.project_id, encoded_search=self.search_string)
        print(f"NOTE: Found {len(query)} Localizations")

        # Extract needed information from data
        data = self.process_query(query).head(1000)
        classes = data['label'].unique().tolist()
        class_to_id = {class_name: i for i, class_name in enumerate(classes)}

        # Download frames (multithreading)
        results = []
        for media_id in tqdm(data['media'].unique(), desc="Processing media"):
            subset, frames = self.process_media(media_id, data[data['media'] == media_id])
            results.append((subset, frames))

        data = pd.concat([result[0] for result in results])
        frame_paths = [path for result in results for path in result[1]]

        # Prepare for YOLO format
        images = {}
        detections = {}
        for frame_path in tqdm(frame_paths, desc="Processing frames"):
            name, result = self.process_frame(frame_path, data, class_to_id)
            images[name] = result['image']
            detections[name] = result['detection']

        # Convert to YOLO dataset
        dataset = sv.DetectionDataset(classes=classes, images=images, annotations=detections)
        dataset.as_yolo(
            self.dataset_dir + "/images",
            self.dataset_dir + "/annotations",
            min_image_area_percentage=0.01,
            data_yaml_path=self.dataset_dir + "/data.yaml"
        )

        # Render results, split to training / validation sets
        render_dataset(dataset, f"{self.dataset_dir}/rendered")
        helpers.split_data(self.dataset_dir, record_confidence=False)


# ----------------------------------------------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Download frames and labels from TATOR")

    parser.add_argument("--api_token", type=str,
                        default=os.getenv('TATOR_TOKEN'),
                        help="Tator API Token")

    parser.add_argument("--project_id", type=int,
                        default=70,
                        help="Project ID for desired media")

    parser.add_argument("--search_string", type=str, required=True,
                        help="Search string for localizations")

    parser.add_argument("--dataset_name", type=str, default="Test",
                        help="Name of the dataset")

    parser.add_argument("--output_dir", type=str,
                        default=os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
                        help="Search string for localizations")

    args = parser.parse_args()

    try:
        downloader = DataDownloader(api_token=args.api_token,
                                    project_id=args.project_id,
                                    search_string=args.search_string,
                                    dataset_name=args.dataset_name,
                                    output_dir=args.output_dir)
        downloader.download_labels()
        print("Done.")
    except Exception as e:
        print(f"ERROR: Could not finish process.\n{e}")
        print(traceback.format_exc())


if __name__ == "__main__":
    main()
