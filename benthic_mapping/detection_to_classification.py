import os
import yaml
import random
import argparse
from tqdm import tqdm

import cv2
import numpy as np
from PIL import Image

import supervision as sv

random.seed(42)


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class DetectionToClassifier:
    def __init__(self, dataset_path=None, output_dir=None, yolo_dataset=None):
        """
        Initialize the DetectionToClassifier with either a dataset path or a YOLODataset object.
        
        :param dataset_path: Path to the YAML dataset configuration file (optional if yolo_dataset provided)
        :param output_dir: Directory where the classification dataset will be saved
        :param yolo_dataset: YOLODataset object (optional if dataset_path provided)
        """
        self.classes = None
        self.dataset = None
        self.dataset_path = dataset_path
        self.train_paths = []
        self.valid_paths = []
        self.num_datasets = None
        self.yolo_dataset = yolo_dataset

        self.detection_dataset = None

        if output_dir:
            self.output_dir = f"{output_dir}/Crops"
        else:
            # If no output_dir and using yolo_dataset, create in the yolo dataset directory
            if yolo_dataset:
                self.output_dir = f"{yolo_dataset.dataset_dir}/Crops"
            else:
                raise ValueError("Either output_dir or yolo_dataset must be provided")

    def load_dataset(self):
        """
        Load the YAML dataset configuration file and create output directories.
        
        :return: None
        """
        # If we have a YOLODataset object, extract info from it
        if self.yolo_dataset:
            self.classes = self.yolo_dataset.classes
            
            # Create train/val paths based on the YOLODataset structure
            train_dir = os.path.join(self.yolo_dataset.dataset_dir, 'images/train')
            val_dir = os.path.join(self.yolo_dataset.dataset_dir, 'images/val') 
            
            self.train_paths = [train_dir]
            self.valid_paths = [val_dir]
            self.num_datasets = 1
            
        # Otherwise use the dataset path
        elif self.dataset_path:
            if not os.path.exists(self.dataset_path):
                raise FileNotFoundError("Dataset not found.")

            with open(self.dataset_path, 'r') as file:
                self.dataset = yaml.safe_load(file)

            self.classes = self.dataset['names']

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
        else:
            raise ValueError("Either dataset_path or yolo_dataset must be provided")

        # Create all the sub folders
        for split in ['train', 'val', 'test']:
            for name in self.classes:
                os.makedirs(f"{self.output_dir}/{split}/{name}", exist_ok=True)

    def load_detection_dataset_from_yolo_dataset(self):
        """
        Load detection dataset from a YOLODataset object.
        
        :return: None
        """
        images = {}
        annotations = {}
        
        # Process data from the YOLODataset
        grouped_data = self.yolo_dataset.data.groupby('image_path')
        
        for image_path, group in tqdm(grouped_data, desc="Loading from YOLODataset"):
            try:
                # Load the image
                image = cv2.imread(image_path)
                if image is None:
                    print(f"Warning: Could not load image {image_path}")
                    continue
                    
                # Store the image
                images[image_path] = image
                
                # Process annotations
                xyxy_list = []
                class_id_list = []
                
                for _, row in group.iterrows():
                    # Get annotation coordinates
                    x1 = row[self.yolo_dataset.standard_columns['x']]
                    y1 = row[self.yolo_dataset.standard_columns['y']]
                    w = row[self.yolo_dataset.standard_columns['width']]
                    h = row[self.yolo_dataset.standard_columns['height']]
                    
                    x2 = x1 + w
                    y2 = y1 + h
                    
                    xyxy_list.append([x1, y1, x2, y2])
                    
                    # Get class ID
                    class_name = row[self.yolo_dataset.standard_columns['label']]
                    class_id = self.yolo_dataset.class_to_id.get(class_name, 0)
                    class_id_list.append(class_id)
                
                # Create annotation object
                annotation = sv.Detections(
                    xyxy=np.array(xyxy_list),
                    class_id=np.array(class_id_list)
                )
                
                # Store annotations
                annotations[image_path] = annotation
                
            except Exception as e:
                print(f"Error processing {image_path}: {str(e)}")
        
        # Create the detection dataset
        self.detection_dataset = sv.DetectionDataset(
            classes=self.classes,
            images=images,
            annotations=annotations
        )
        
        print(f"NOTE: Loaded dataset from YOLODataset - {len(self.detection_dataset)} detections found")

    def load_detection_dataset(self, index):
        """
        Load detection dataset, handling cases where train or valid datasets might be missing.

        :param index: Index of the dataset to load
        :return: None
        """
        # If we're using a YOLODataset, use the specialized method
        if self.yolo_dataset:
            self.load_detection_dataset_from_yolo_dataset()
            return
        
        images = {}
        annotations = {}

        # Load train dataset if available
        if index < len(self.train_paths):
            train_path = self.train_paths[index]
            images_path = f"{train_path}/images" if not train_path.endswith("images") else train_path
            labels_path = images_path.replace('images', 'labels')
            try:
                train = sv.DetectionDataset.from_yolo(
                    images_directory_path=images_path,
                    annotations_directory_path=labels_path,
                    data_yaml_path=self.dataset_path,
                )
                images.update(train.images)
                annotations.update(train.annotations)
            except Exception as e:
                print(f"Warning: Failed to load train dataset at index {index}. Error: {str(e)}")

        # Load valid dataset if available
        if index < len(self.valid_paths):
            valid_path = self.valid_paths[index]
            images_path = f"{valid_path}/images" if not valid_path.endswith("images") else valid_path
            labels_path = images_path.replace('images', 'labels')
            try:
                valid = sv.DetectionDataset.from_yolo(
                    images_directory_path=images_path,
                    annotations_directory_path=labels_path,
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

        print(f"NOTE: Loaded dataset - {len(self.detection_dataset)} detections found")

    def extract_crop(self, image, xyxy):
        """
        Extract a crop (sub-image) from a NumPy array image based on bounding box coordinates.

        :param image: NumPy array representing the image
        :param xyxy: List or tuple of bounding box coordinates [x1, y1, x2, y2]
        :return: NumPy array of the extracted crop, or None if the crop has no area
        """
        x1, y1, x2, y2 = map(int, xyxy)

        # Ensure coordinates are within image boundaries
        height, width = image.shape[:2]
        x1, x2 = max(0, x1), min(width, x2)
        y1, y2 = max(0, y1), min(height, y2)

        # Extract the crop
        crop = image[y1:y2, x1:x2]

        if crop.shape[0] > 0 and crop.shape[1] > 0:
            return crop

        return None

    def save_crop(self, split, class_name, crop_name, crop):
        """
        Save a crop as an RGB image.

        :param split: Dataset split (e.g., 'train', 'val', 'test')
        :param class_name: Name of the class
        :param crop_name: Name of the crop
        :param crop: Numpy array representing the image
        :return: Path to the saved crop
        """
        crop_dir = f"{self.output_dir}/{split}/{class_name}"
        os.makedirs(crop_dir, exist_ok=True)

        crop_path = f"{crop_dir}/{crop_name}"

        # Ensure the crop is in RGB format
        if len(crop.shape) == 2:  # If it's a grayscale image
            crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2RGB)
        elif crop.shape[2] == 4:  # If it's RGBA
            crop = cv2.cvtColor(crop, cv2.COLOR_RGBA2RGB)
        elif crop.shape[2] == 3:
            # If it's already RGB, ensure it's in the correct order (OpenCV uses BGR by default)
            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

        # Ensure the data type is uint8
        if crop.dtype != np.uint8:
            crop = (crop * 255).astype(np.uint8)

        # Save the image
        Image.fromarray(crop).save(crop_path)

        return str(crop_path)

    def create_crops(self):
        """
        Creates classification crops from detection bounding boxes and 
        organizes them into train/val/test splits by class.

        :return: None
        """
        # Loop through all the images in the detection dataset
        for image_path, image in tqdm(self.detection_dataset.images.items(), desc="Creating crops"):

            # Get the image basename, corresponding detections
            image_name = os.path.basename(image_path).split(".")[0]
            detections = self.detection_dataset.annotations[image_path]

            # Loop through detections, crop, and then save in split folder
            for i, (xyxy, class_id) in enumerate(zip(detections.xyxy, detections.class_id)):

                # Randomly assign the crop to train, valid or test
                split = random.choices(['train', 'val', 'test'], weights=[70, 20, 10])[0]

                # Get the crop
                crop = self.extract_crop(image, xyxy)

                if crop is not None:
                    class_name = self.detection_dataset.classes[class_id]
                    crop_name = f"{class_name}_{i}_{image_name}.jpeg"
                    self.save_crop(split, class_name, crop_name, crop)

    def write_classification_yaml(self):
        """
        Create a YOLO-formatted classification dataset YAML file.
        
        :return: None
        """
        # Count classes and items per class
        class_counts = {class_name: {'train': 0, 'val': 0, 'test': 0} for class_name in self.classes}
        
        for split in ['train', 'val', 'test']:
            for class_name in self.classes:
                class_dir = f"{self.output_dir}/{split}/{class_name}"
                if os.path.exists(class_dir):
                    count = len([f for f in os.listdir(class_dir) if os.path.isfile(os.path.join(class_dir, f))])
                    class_counts[class_name][split] = count
        
        # Create the YAML content
        yaml_content = {
            'path': self.output_dir,
            'train': f"{self.output_dir}/train",
            'val': f"{self.output_dir}/val",
            'test': f"{self.output_dir}/test",
            'nc': len(self.classes),
            'names': {i: name for i, name in enumerate(self.classes)},
        }
        
        # Write the YAML file
        yaml_path = f"{self.output_dir}/data.yaml"
        with open(yaml_path, 'w') as f:
            yaml.dump(yaml_content, f, default_flow_style=False)
            
        # Print summary
        print(f"Classification dataset YAML written to {yaml_path}")
        print("Class distribution:")
        for class_name, counts in class_counts.items():
            print(f"  {class_name}: train={counts['train']}, val={counts['val']}, test={counts['test']}")

    def process_dataset(self):
        """
        Main execution method that loads datasets and creates classification crops.

        :return: None
        """
        # Load the data.yaml file or YOLODataset
        self.load_dataset()

        if self.yolo_dataset:
            # Use the YOLODataset object directly
            self.load_detection_dataset(0)  # Index doesn't matter, will use yolo_dataset
            self.create_crops()
        else:
            # Loop through each of the datasets, crop bboxes
            for i in range(self.num_datasets):
                self.load_detection_dataset(i)
                self.create_crops()

        # Create the classification dataset YAML file
        self.write_classification_yaml()

        print(f"NOTE: Created classification dataset in {self.output_dir}")


# ----------------------------------------------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------------------------------------------


def main():
    """
    Parse command line arguments and run the crop creation process.

    :return: None
    """
    parser = argparse.ArgumentParser(description="Create a classification dataset from a detection dataset")

    parser.add_argument("--dataset_path", type=str, required=True,
                        help="Path to the detection dataset's data.yaml file")

    parser.add_argument("--output_dir", type=str, required=True,
                        help="Path to the output directory for the classification dataset")

    args = parser.parse_args()

    try:
        # Run the conversion process
        converter = DetectionToClassifier(dataset_path=args.dataset_path, 
                                          output_dir=args.output_dir)
        converter.process_dataset()
        print("NOTE: Process completed successfully")
        print("Done.")
        
    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
    except ValueError as e:
        print(f"Error: Invalid value - {e}")
    except Exception as e:
        print(f"Error: An unexpected error occurred - {e}")


if __name__ == "__main__":
    main()
