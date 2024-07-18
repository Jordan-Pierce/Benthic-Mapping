import os
import random
import shutil
import argparse
import traceback
import concurrent.futures
from functools import partial

import tator
import pandas as pd

import supervision as sv

from Common import render_dataset


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------

class DataDownloader:
    def __init__(self,
                 api_token: str,
                 project_id: int,
                 search_string: str,
                 frac: float,
                 code_as: str,
                 dataset_name: str,
                 draw_bboxes: bool,
                 output_dir: str):
        """

        :param api_token:
        :param project_id:
        :param search_string:
        :param frac:
        :param code_as:
        :param dataset_name:
        :param draw_bboxes:
        :param output_dir:
        """
        self.api = None

        self.token = api_token
        self.project_id = project_id
        self.search_string = search_string
        self.frac = frac

        self.code_as = code_as
        self.dataset_name = dataset_name
        self.draw_bboxes = draw_bboxes

        self.dataset_dir = f"{output_dir}/{self.dataset_name}"

        self.query = None
        self.data = None
        self.classes = None
        self.class_to_id = None

        self.authenticate()
        self.create_directories()
        self.save_query_string()

    def authenticate(self):
        """

        :return:
        """
        try:

            self.api = tator.get_api(host='https://cloud.tator.io', token=self.token)
            print(f"NOTE: Authentication successful for {self.api.whoami().username}")

        except Exception as e:
            raise Exception(f"ERROR: Could not authenticate with provided API Token\n{e}")

    def create_directories(self):
        """

        :return:
        """
        os.makedirs(self.dataset_dir, exist_ok=True)
        os.makedirs(f"{self.dataset_dir}/images", exist_ok=True)
        os.makedirs(f"{self.dataset_dir}/labels", exist_ok=True)

    def save_query_string(self):
        """
        Write the search string to text file

        :return:
        """
        try:

            with open(f"{self.dataset_dir}/search_string.txt", 'w') as file:
                file.write(self.search_string)

        except Exception as e:
            print(f"WARNING: An error occurred while writing search string to file:\n{e}")

    def write_yaml(self):
        """
        Writes a YOLO-formatted dataset yaml file

        :return:
        """
        self.classes = self.data['label'].unique().tolist()
        self.class_to_id = {class_name: i for i, class_name in enumerate(self.classes)}

        # Create data.yaml
        with open(f"{self.dataset_dir}/data.yaml", 'w') as f:
            f.write(f"names: {self.classes}\n")
            f.write(f"nc: {len(self.classes)}\n")
            f.write(f"train: {os.path.join(self.dataset_dir, 'images')}\n")
            f.write(f"val: {os.path.join(self.dataset_dir, 'images')}\n")

    def process_query(self):
        """
        Process the query by restructuring the data into a DataFrame;
        each row contains a localization, and the media it belongs to.
        Then group by image_path and randomly sample.

        :return: None
        """
        data = []

        for q in self.query:
            frame_name = f"{q.media}_{q.frame}.jpg"
            label_name = f"{q.media}_{q.frame}.txt"

            try:
                original_label = str(q.attributes['ScientificName'])
            except Exception as e:
                original_label = 'Unknown'

            # Encode the label as something different
            label = self.code_as if self.code_as else original_label

            # Create a dictionary for each row
            row_dict = {
                'media': q.media,
                'frame': q.frame,
                'frame_name': frame_name,
                'frame_path': f"{self.dataset_dir}/images/{frame_name}",
                'label_name': label_name,
                'label_path': f"{self.dataset_dir}/labels/{label_name}",
                'x': q.x,
                'y': q.y,
                'width': q.width,
                'height': q.height,
                'label': label
            }

            # Add the dictionary to the list
            data.append(row_dict)

        # Create DataFrame from the list of dictionaries
        self.data = pd.DataFrame(data)

        if self.frac < 1:

            # Group by frame_path
            frames = self.data['frame'].unique().tolist()
            sampled_frames = random.sample(frames, int(len(frames) * self.frac))
            sampled_data = self.data[self.data['frame'].isin(sampled_frames)]

            # Reset index after sampling
            self.data = sampled_data.reset_index(drop=True)

        self.data.dropna(axis=0, how='any', inplace=True)
        self.data.reset_index(drop=True, inplace=True)

    def download_frames(self):
        """
        Multithreading function for downloading multiple frames in parallel.

        """
        # Get unique frame paths
        paths = self.data['frame_path'].unique()

        # Create a partial function with self as the first argument
        func = partial(self.download_frame)

        # Use ThreadPoolExecutor for parallel downloads
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            # Submit download tasks for each unique frame path
            futures = [executor.submit(func, self.data[self.data['frame_path'] == path].head(1)) for path in paths]

            # Wait for all tasks to complete
            concurrent.futures.wait(futures)

    def download_frame(self, row):
        """
        Takes in a row from a dataframe, downloads corresponding
        frame and saves it to disk.

        :param row:
        """
        # Download the frame if it doesn't already exist
        if not os.path.exists(row['frame_path'].item()):
            try:
                temp = self.api.get_frame(id=row['media'].item(),
                                          tile=f"{row['width'].item()}x{row['height'].item()}",
                                          frames=[int(row['frame'].item())])

                shutil.move(temp, row['frame_path'].item())
                print(f"Downloaded: {row['frame_path'].item()}")
            except Exception as e:
                print(f"Error downloading {row['frame_path'].item()}: {str(e)}")

    def write_labels(self):
        """
        Write YOLO-formatted labels to text files.

        :return
        """
        for label_path in self.data['label_path'].unique():
            # Get all labels that correspond to this label / image file
            label_df = self.data[self.data['label_path'] == label_path]

            yolo_annotations = []
            for _, row in label_df.iterrows():
                class_id = self.class_to_id[row['label']]
                x_center = (row['x'] + row['width'] / 2)
                y_center = (row['y'] + row['height'] / 2)
                w = row['width']
                h = row['height']

                # Convert to YOLO format (normalized)
                x_center /= 1.0
                y_center /= 1.0
                w /= 1.0
                h /= 1.0

                # Create YOLO-formatted annotation string
                yolo_annotation = f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}"
                yolo_annotations.append(yolo_annotation)

            # Save annotations
            with open(label_path, 'w') as f:
                f.write('\n'.join(yolo_annotations))

            print("Downloaded: ", label_path)

    def render(self):
        """

        :return:
        """
        try:

            if self.draw_bboxes:
                dataset = sv.DetectionDataset.from_yolo(
                    images_directory_path=f"{self.dataset_dir}/images",
                    annotations_directory_path=f"{self.dataset_dir}/labels",
                    data_yaml_path=f"{self.dataset_dir}/data.yaml",
                )

                render_dataset(dataset, f"{self.dataset_dir}/render")

        except Exception as e:
            print(f"ERROR: Failed to render dataset\n{e}")

    def download_data(self):
        """
        Download and process labels, creating a YOLO-format dataset.
        """
        # Query TATOR given the search string
        print("NOTE: Querying TATOR for data")
        self.query = self.api.get_localization_list(project=self.project_id,
                                                    encoded_search=self.search_string)

        # Extract data from query
        self.process_query()
        print(f"NOTE: Found {len(self.query)} localizations, sampled {len(self.data)}")

        # Write the dataset yaml file
        self.write_yaml()

        # Write all the label files
        self.write_labels()

        # Download all the frames
        self.download_frames()

        # Render the dataset
        self.render()

        print(f"NOTE: Dataset created at {self.dataset_dir}")


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

    parser.add_argument("--frac", type=float, default=1.0,
                        help="Sub-sample the amount of images being downloaded")

    parser.add_argument("--code_as", type=str, default="",
                        help="Change all labels to _, else original labels are kept")

    parser.add_argument("--dataset_name", type=str, required=True,
                        help="Name of the dataset")

    parser.add_argument("--draw_bboxes", action="store_true",
                        help="Render the annotations superimposed on images")

    parser.add_argument("--output_dir", type=str, required=True,
                        help="Where to download data to")

    args = parser.parse_args()

    try:

        DataDownloader(api_token=args.api_token,
                       project_id=args.project_id,
                       search_string=args.search_string,
                       frac=args.frac,
                       code_as=args.code_as,
                       dataset_name=args.dataset_name,
                       draw_bboxes=args.draw_bboxes,
                       output_dir=args.output_dir).download_data()

        print("Done.")

    except Exception as e:
        print(f"ERROR: Could not finish process.\n{e}")
        print(traceback.format_exc())


if __name__ == "__main__":
    main()
