import os
import yaml
import random
import argparse
from tqdm import tqdm

import cv2
import numpy as np
from PIL import Image

import supervision as sv


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------

class ChipCreator:
    def __init__(self, dataset_path, output_dir):

        self.classes = None
        self.dataset = None
        self.dataset_path = dataset_path
        self.train_paths = []
        self.valid_paths = []
        self.num_datasets = None

        self.detection_dataset = None

        self.output_dir = f"{output_dir}/Chips"

    def load_dataset(self):
        """

        :return:
        """
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError("Dataset not found.")

        with open(self.dataset_path, 'r') as file:
            self.dataset = yaml.safe_load(file)

        self.classes = self.dataset['names']

        # Create all the sub folders
        for split in ['train', 'valid', 'test']:
            for name in self.classes:
                os.makedirs(f"{self.output_dir}/{split}/{name}", exist_ok=True)

        # Process train paths
        if isinstance(self.dataset.get('train'), str):
            self.train_paths = [self.dataset['train']]
        elif isinstance(self.dataset.get('train'), list):
            self.train_paths = self.dataset['train']

        # Process validation paths
        if isinstance(self.dataset.get('val'), str):
            self.valid_paths = [self.dataset['val']]
        elif isinstance(self.dataset.get('val'), list):
            self.valid_paths = self.dataset['val']

        # Total number of datasets
        self.num_datasets = max(len(self.train_paths), len(self.valid_paths))

    def load_detection_dataset(self, index):
        """
        Load detection dataset, handling cases where train or valid datasets might be missing.

        :param index: Index of the dataset to load
        :return: None
        """
        images = {}
        annotations = {}

        # Load train dataset if available
        if index < len(self.train_paths):
            train_path = self.train_paths[index]
            train_path = f"{train_path}/images" if not train_path.endswith("images") else train_path
            try:
                train = sv.DetectionDataset.from_yolo(
                    images_directory_path=train_path,
                    annotations_directory_path=train_path.replace('images', 'labels'),
                    data_yaml_path=self.dataset_path,
                )
                images.update(train.images)
                annotations.update(train.annotations)
            except Exception as e:
                print(f"Warning: Failed to load train dataset at index {index}. Error: {str(e)}")

        # Load valid dataset if available
        if index < len(self.valid_paths):
            valid_path = self.valid_paths[index]
            valid_path = f"{valid_path}/images" if not valid_path.endswith("images") else valid_path
            try:
                valid = sv.DetectionDataset.from_yolo(
                    images_directory_path=valid_path,
                    annotations_directory_path=valid_path.replace('images', 'labels'),
                    data_yaml_path=self.dataset_path,
                )
                images.update(valid.images)
                annotations.update(valid.annotations)
            except Exception as e:
                print(f"Warning: Failed to load valid dataset at index {index}. Error: {str(e)}")

        # Check if any data was loaded
        if not images or not annotations:
            raise ValueError(f"No data could be loaded for index {index}. Please check your dataset paths.")

        # Create the detection dataset
        self.detection_dataset = sv.DetectionDataset(classes=self.classes,
                                                     images=images,
                                                     annotations=annotations)

    def chip_image(self, image, xyxy):
        """
        Extract a chip (sub-image) from a NumPy array image based on bounding box coordinates.

        :param image: NumPy array representing the image
        :param xyxy: List or tuple of bounding box coordinates [x1, y1, x2, y2]
        :return: NumPy array of the extracted chip, or None if the chip has no area
        """
        x1, y1, x2, y2 = map(int, xyxy)

        # Ensure coordinates are within image boundaries
        height, width = image.shape[:2]
        x1, x2 = max(0, x1), min(width, x2)
        y1, y2 = max(0, y1), min(height, y2)

        # Extract the chip
        chip = image[y1:y2, x1:x2]

        if chip.shape[0] > 0 and chip.shape[1] > 0:
            return chip

        return None

    def save_chip(self, split, class_name, chip_name, chip):
        """
        Save a chip as an RGB image.

        :param split: Dataset split (e.g., 'train', 'valid')
        :param class_name: Name of the class
        :param chip_name: Name of the chip
        :param chip: Numpy array representing the image
        :return: Path to the saved chip
        """
        chip_dir = f"{self.output_dir}/{split}/{class_name}"
        os.makedirs(chip_dir, exist_ok=True)

        chip_path = f"{chip_dir}/{chip_name}"

        # Ensure the chip is in RGB format
        if len(chip.shape) == 2:  # If it's a grayscale image
            chip = cv2.cvtColor(chip, cv2.COLOR_GRAY2RGB)
        elif chip.shape[2] == 4:  # If it's RGBA
            chip = cv2.cvtColor(chip, cv2.COLOR_RGBA2RGB)
        elif chip.shape[2] == 3:
            # If it's already RGB, ensure it's in the correct order (OpenCV uses BGR by default)
            chip = cv2.cvtColor(chip, cv2.COLOR_BGR2RGB)

        # Ensure the data type is uint8
        if chip.dtype != np.uint8:
            chip = (chip * 255).astype(np.uint8)

        # Save the image
        Image.fromarray(chip).save(chip_path)

        return str(chip_path)

    def create_chips(self):
        """

        :return:
        """
        # Loop through all the images in the detection dataset
        for image_path, image in self.detection_dataset.images.items():

            # Get the image basename, corresponding detections
            image_name = os.path.basename(image_path).split(".")[0]
            detections = self.detection_dataset.annotations[image_path]

            # Loop through detections, chip, and then save in split folder
            for i, (xyxy, class_id) in enumerate(zip(detections.xyxy, detections.class_id)):

                # Randomly assign the chip to train, valid or test
                split = random.choices(['train', 'valid', 'test'], weights=[70, 20, 10])[0]

                # Get the chip
                chip = self.chip_image(image, xyxy)

                if chip is not None:
                    class_name = self.detection_dataset.classes[class_id]
                    chip_name = f"{class_name}_{i}_{image_name}.jpg"
                    self.save_chip(split, class_name, chip_name, chip)

    def run(self):
        """

        :return:
        """
        # Load the data.yaml file
        self.load_dataset()

        # Loop through each of the datasets, chip bboxes
        for _ in tqdm(range(self.num_datasets)):
            self.load_detection_dataset(_)
            self.create_chips()

        print(f"NOTE: Created classification dataset in {self.output_dir}")


# ----------------------------------------------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------------------------------------------


def main():
    """

    :return:
    """
    parser = argparse.ArgumentParser(description="Create a classification dataset from a detection dataset")

    parser.add_argument("--dataset_path", type=str, required=True,
                        help="Path to the detection dataset's data.yaml file")

    parser.add_argument("--output_dir", type=str, required=True,
                        help="Path to the output directory for the classification dataset")

    args = parser.parse_args()

    dataset_path = args.dataset_path
    output_dir = args.output_dir

    ChipCreator(dataset_path, output_dir).run()


if __name__ == "__main__":
    main()
