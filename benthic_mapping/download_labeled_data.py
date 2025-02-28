import argparse
import concurrent.futures
import json
import os
import random
import shutil
import traceback
from functools import partial

import pandas as pd
import supervision as sv
from Common import render_dataset

import tator

# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class LabeledDataDownloader:
    def __init__(self,
                 api_token: str,
                 project_id: int,
                 media_id: int,
                 search_string: str,
                 frac: float,
                 code_as: dict,
                 dataset_name: str,
                 draw_bboxes: bool,
                 output_dir: str,
                 label_field: str):
        """

        :param api_token:
        :param project_id:
        :param media_id:
        :param search_string:
        :param frac:
        :param dataset_name:
        :param output_dir:
        :param label_field: Field name to use as the label
        """
        self.api = None

        self.token = api_token
        self.project_id = project_id
        self.media_id = media_id
        self.search_string = search_string
        self.frac = frac
        self.label_field = label_field

        self.dataset_name = dataset_name

        self.dataset_dir = f"{output_dir}/{self.dataset_name}"
        self.image_dir = f"{self.dataset_dir}/images"
        self.label_dir = f"{self.dataset_dir}/labels"

        os.makedirs(self.dataset_dir, exist_ok=True)
        os.makedirs(self.image_dir, exist_ok=True)
        os.makedirs(self.label_dir, exist_ok=True)

        self.query = None
        self.data = None
        self.classes = None

        self.authenticate()
        self.create_directories()
        self.save_query_string()

    def authenticate(self):
        """

        :return:
        """
        try:
            # Authenticate with TATOR
            self.api = tator.get_api(
                host='https://cloud.tator.io', token=self.token)
            print(
                f"NOTE: Authentication successful for {self.api.whoami().username}")

        except Exception as e:
            raise Exception(
                f"ERROR: Could not authenticate with provided API Token\n{e}")

    def save_query_string(self):
        """
        Write the search string to text file

        :return:
        """
        try:
            # Write search string to file for the posterity
            if self.search_string:
                with open(f"{self.dataset_dir}/search_string.txt", 'w') as file:
                    file.write(self.search_string)

        except Exception as e:
            print(
                f"WARNING: An error occurred while writing search string to file:\n{e}")

    def process_query(self):
        """
        Process the query by restructuring the data into a DataFrame;
        each row contains a localization, and the media it belongs to.
        Then group by image_path and randomly sample.

        :return: None
        """
        data = []

        for q in self.query:

            image_name = f"{q.media}_{q.frame}.jpg"
            label_name = f"{q.media}_{q.frame}.txt"

            try:
                label = str(q.attributes[self.label_field])
            except Exception as e:
                raise Exception(f"ERROR: Query includes instances without '{self.label_field}' field")

            # Create a dictionary for each row
            row_dict = {
                'media': q.media,
                'frame': q.frame,
                'image_name': image_name,
                'image_path': f"{self.dataset_dir}/images/{image_name}",
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

        # QA / QC
        self.data.dropna(axis=0, how='any', inplace=True)
        self.data.drop_duplicates(inplace=True)
        self.data.reset_index(drop=True, inplace=True)

        if self.frac < 1:
            # Group by image_path
            frames = self.data['image_path'].unique().tolist()
            sampled_frames = random.sample(
                frames, int(len(frames) * self.frac))
            sampled_data = self.data[self.data['image_path'].isin(
                sampled_frames)]

            # Reset index after sampling
            self.data = sampled_data.reset_index(drop=True)

    def download_frames(self):
        """
        Multithreading function for downloading multiple frames in parallel.

        """
        # Get unique image paths
        paths = self.data['image_path'].unique()

        # Create a partial function with self as the first argument
        func = partial(self.download_frame)

        # Use ThreadPoolExecutor for parallel downloads
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            # Submit download tasks for each unique image path
            futures = [executor.submit(
                func, self.data[self.data['image_path'] == path].head(1)) for path in paths]

            # Wait for all tasks to complete
            concurrent.futures.wait(futures)

    def download_frame(self, row):
        """
        Takes in a row from a dataframe, downloads corresponding
        frame and saves it to disk.

        :param row:
        """
        # Download the frame if it doesn't already exist
        if not os.path.exists(row['image_path'].item()):
            try:
                # Get the image from Tator at full resolution
                temp = self.api.get_frame(id=row['media'].item(),
                                          tile=f"{row['width'].item()}x{row['height'].item()}",
                                          frames=[int(row['frame'].item())])

                # Move the image to the correct directory
                shutil.move(temp, row['image_path'].item())
                print(f"Downloaded: {row['image_path'].item()}")

            except Exception as e:
                print(
                    f"Error downloading {row['image_path'].item()}: {str(e)}")

    def download_data(self):
        """
        Download and process labels, creating a YOLO-format dataset.
        """
        # Query TATOR given the search string
        print("NOTE: Querying TATOR for data")
        self.query = self.api.get_localization_list(
            project=self.project_id, encoded_search=self.search_string)

        # Extract data from query
        self.process_query()
        print(
            f"NOTE: Found {len(self.query)} localizations, (keeping {len(self.data)})")

        # Write the dataset yaml file
        print("NOTE: Writing dataset yaml file")
        self.write_yaml()

        # Write all the label files
        print("NOTE: Writing label files")
        self.write_labels()

        # Download all the frames
        print("NOTE: Downloading frames")
        self.download_frames()

        print(f"NOTE: Dataset created at {self.dataset_dir}")


# ----------------------------------------------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Download frames and labels from TATOR")

    parser.add_argument("--api_token", type=str, required=True,
                        default=os.getenv('TATOR_TOKEN'),
                        help="Tator API Token")

    parser.add_argument("--project_id", type=int, required=True, default=70,
                        help="Project ID for desired media")

    parser.add_argument("--search_string", type=str, required=True,
                        help="Search string for localizations")

    parser.add_argument("--frac", type=float, default=1.0,
                        help="Sub-sample the amount of images being downloaded")

    parser.add_argument("--dataset_name", type=str, required=True,
                        help="Name of the dataset")

    parser.add_argument("--output_dir", type=str, required=True,
                        help="Where to download data to")

    parser.add_argument("--label_field", type=str, required=True,
                        help="Field name to use as the label")
    
    args = parser.parse_args()

    try:
        # Download the data
        downloader = LabeledDataDownloader(api_token=args.api_token,
                                           project_id=args.project_id,
                                           search_string=args.search_string,
                                           frac=args.frac,
                                           dataset_name=args.dataset_name,
                                           output_dir=args.output_dir,
                                           label_field=args.label_field)

        # Download the data
        downloader.download_data()

        print("Done.")

    except Exception as e:
        print(f"ERROR: Could not finish process.\n{e}")
        print(traceback.format_exc())


if __name__ == "__main__":
    main()
