import os
import json
import random
import shutil
import argparse
import traceback

from tqdm import tqdm
import concurrent.futures
from functools import partial

import pandas as pd
import supervision as sv

import tator


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------

# TODO allow user to specify the resolution of the images being downloaded, keep aspect resolution
class LabeledDataDownloader:
    def __init__(self,
                 api_token: str,
                 project_id: int,
                 search_string: str,
                 frac: float,
                 dataset_name: str,
                 output_dir: str,
                 label_field: str or list):
        """

        :param api_token:
        :param project_id:
        :param media_id:
        :param search_string:
        :param frac:
        :param dataset_name:
        :param output_dir:
        :param label_field: Field name(s) to use as the label(s), can be a single field or a list
        """
        self.api = None

        self.token = api_token
        self.project_id = project_id
        self.search_string = search_string
        self.frac = frac
        
        # Support both single field and list of fields
        if isinstance(label_field, list):
            self.label_field = label_field
        else:
            self.label_field = [label_field]

        self.dataset_name = dataset_name

        # Convert paths to absolute paths
        self.output_dir = os.path.abspath(output_dir)
        self.dataset_dir = os.path.abspath(f"{self.output_dir}/{self.dataset_name}")
        self.image_dir = os.path.abspath(f"{self.dataset_dir}/images")

        os.makedirs(self.dataset_dir, exist_ok=True)
        os.makedirs(self.image_dir, exist_ok=True)

        self.query = None
        self.data = None
        self.data_dict = {}

        self.authenticate()
        self.save_query_string()

    def authenticate(self):
        """

        :return:
        """
        try:
            # Authenticate with TATOR
            self.api = tator.get_api(host='https://cloud.tator.io', token=self.token)
            print(f"NOTE: Authentication successful for {self.api.whoami().username}")

        except Exception as e:
            raise Exception(f"ERROR: Could not authenticate with provided API Token\n{e}")

    def save_query_string(self):
        """
        Write the search string to text file

        :return:
        """
        try:
            # Write search string to file for the posterity
            if self.search_string:
                query_string_path = os.path.abspath(f"{self.dataset_dir}/search_string.txt")
                with open(query_string_path, 'w') as file:
                    file.write(self.search_string)
                    
            print(f"NOTE: Search string saved to {query_string_path}")

        except Exception as e:
            print(f"WARNING: An error occurred while writing search string to file:\n{e}")

    def query_tator(self):
        """
        Query TATOR for the desired data

        :return:
        """
        try:
            # Query TATOR for the desired data
            print(f"NOTE: Querying Tator for labeled data")
            self.query = self.api.get_localization_list(project=self.project_id, encoded_search=self.search_string)

        except Exception as e:
            raise Exception(f"ERROR: Could not query TATOR for data\n{e}")
    
    def process_query(self):
        """
        Process the query by restructuring the data into a DataFrame;
        each row contains a localization, and the media it belongs to.
        Then group by image_path and randomly sample.

        :return: None
        """
        print(f"NOTE: Found {len(self.query)} localizations")
        
        data = []
        
        # Loop through the queries (Class - tator localization)
        for q in tqdm(self.query, desc="Processing query"):

            image_name = f"{q.media}_{q.frame}.jpg"
            label_name = f"{q.media}_{q.frame}.txt"

            # Handle labels differently based on whether we have one field or multiple
            if len(self.label_field) == 1:
                # Single field - store as string
                label = q.to_dict().get(self.label_field[0], None)
            else:
                # Multiple fields - store as nested dictionary
                label = {}
                for field in self.label_field:
                    label[field] = q.to_dict().get(field, None)

            # Determine if the localization is a bounding box or a polygon
            if q.to_dict().get("points", None):
                # Get the polygon from the query
                polygon = q.points
                # Get the bounding box from the polygon
                x, y, width, height = sv.polygon_to_xyxy(polygon)
            elif (q.x is not None) and (q.y is not None) and (q.width is not None) and (q.height is not None):
                # No polygon, set to None
                polygon = None
                # Get the bounding box from the query
                x, y, width, height = q.x, q.y, q.width, q.height
            else:    
                # No polygon or bounding box, set to None
                polygon = None
                x, y, width, height = None, None, None, None
            
            # Create a dictionary for each row with absolute paths
            row_dict = {
                'media': q.media,
                'frame': q.frame,
                'image_name': image_name,
                'image_path': os.path.abspath(f"{self.image_dir}/{image_name}"),
                'label_name': label_name,
                'x': x,
                'y': y,
                'width': width,
                'height': height,
                'polygon': polygon,
                'label': label
            }
             
            # Add the dictionary to the list
            data.append(row_dict)

        # Create DataFrame from the list of dictionaries
        self.data = pd.DataFrame(data)

        if self.frac < 1:
            # Group by image_path
            image_paths = self.data['image_path'].unique().tolist()
            sampled_images = random.sample(image_paths, int(len(image_paths) * self.frac))
            sampled_data = self.data[self.data['image_path'].isin(sampled_images)]

            # Reset index after sampling
            self.data = sampled_data.reset_index(drop=True)
        
        print(f"NOTE: Found {len(self.data)} localizations after sampling")
            
        # Convert to dict
        self.data_dict = self.data.to_dict('records')
            
        # Save the dataframe to disk as JSON with absolute path
        json_path = os.path.abspath(f"{self.dataset_dir}/data.json")
        with open(json_path, 'w') as f:
            # Convert DataFrame to JSON with proper formatting
            json_data = json.dumps(self.data_dict, indent=2)
            f.write(json_data)
            
        print(f"NOTE: Data saved to {json_path}")

    def download_images(self, max_workers=os.cpu_count() // 3, max_retries=3):
        """
        Multithreading function for downloading multiple images in parallel.
        
        :param max_workers: Maximum number of concurrent downloads
        :param max_retries: Maximum number of retry attempts per download
        """
        print(f"NOTE: Downloading images to {self.image_dir}")
        
        # Get unique combinations of media ID and frame number
        media_images = self.data[['media', 'frame', 'image_path']].drop_duplicates().to_dict('records')
        
        # Use ThreadPoolExecutor for parallel downloads
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit download tasks for each unique frame
            futures = [executor.submit(self.download_image, item, max_retries) for item in media_images]
            
            # Process results as they complete with a progress bar
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Downloading images"):
                try:
                    future.result()  # This will raise any exceptions that occurred during execution
                except Exception as e:
                    print(f"Failed download task: {str(e)}")
                    
        print(f"NOTE: Images downloaded to {self.image_dir}")

    def download_image(self, item, max_retries=3):
        """
        Downloads a frame and saves it to disk with retry logic.
        
        :param item: Dictionary with media, frame, and image_path keys
        :param max_retries: Maximum number of retry attempts
        :return: Path to saved image or None if failed
        """
        # Skip download if the file already exists
        if os.path.exists(item['image_path']):
            return item['image_path']
            
        for attempt in range(max_retries):
            try:
                # Get image dimensions from first matching row
                frame_data = self.data[(self.data['media'] == item['media']) & 
                                      (self.data['frame'] == item['frame'])].iloc[0]
                
                # Get the image from Tator at full resolution
                temp = self.api.get_frame(id=item['media'],
                                          tile=f"{frame_data['width']}x{frame_data['height']}",
                                          frames=[int(item['frame'])])

                # Move the image to the correct directory
                shutil.move(temp, item['image_path'])
                return item['image_path']
                
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"Retrying download for media {item['media']}, frame {item['frame']} again...")
                else:
                    print(f"Error downloading {item['image_path']}: {str(e)}")
        
    def download_data(self):
        """
        Download and process labels, creating a YOLO-format dataset.
        """
        # Query TATOR given the search string
        self.query_tator()

        # Extract data from query
        self.process_query()

        # Download all the images
        self.download_images()


# ----------------------------------------------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Download images and labels from TATOR")

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

    parser.add_argument("--label_field", type=str, required=True, nargs='+',
                        help="Field name(s) to use as the label; can be single field or a list of fields")
    
    args = parser.parse_args()

    try:
        # Convert output_dir to absolute path
        output_dir = os.path.abspath(args.output_dir)
        
        # Handle label_field as either single field or list
        label_field = args.label_field
        if len(label_field) == 1:
            label_field = label_field[0]  # If only one field provided, convert from list to string
        
        # Download the data
        downloader = LabeledDataDownloader(api_token=args.api_token,
                                           project_id=args.project_id,
                                           search_string=args.search_string,
                                           frac=args.frac,
                                           dataset_name=args.dataset_name,
                                           output_dir=output_dir,
                                           label_field=label_field)

        # Download the data
        downloader.download_data()

        print("Done.")

    except Exception as e:
        print(f"ERROR: Could not finish process.\n{e}")
        print(traceback.format_exc())


if __name__ == "__main__":
    main()
