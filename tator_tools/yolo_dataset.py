import os
import glob
import yaml
import shutil
import argparse
from tqdm.auto import tqdm

import cv2
import numpy as np
import pandas as pd
from PIL import Image

import supervision as sv


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class YOLODataset:
    """
    Class to create a YOLO-formatted dataset for object detection.
    Takes a pandas DataFrame with annotation data and generates the necessary directory structure,
    labels, and configuration files.

    Required DataFrame columns:
    - label: class name for the object
    - x, y, width, height: bounding box coordinates (can be absolute or normalized)
    - image_path: path to the image file
    - image_name: filename of the image
    """

    def __init__(self, data, output_dir, dataset_name="yolo_dataset", train_ratio=0.8, test_ratio=0, task='detect'):
        """
        Initialize the YOLO dataset.

        :param data: pandas DataFrame containing annotation data
        :param output_dir: directory where datasets will be created
        :param dataset_name: name of the dataset (subdirectory will be created)
        :param train_ratio: train/val split ratio (default: 0.8)
        :param test_ratio: test split ratio (default: 0) - if > 0, train_ratio becomes train ratio
        :param task: task type ('detect' or 'segment')
        """
        # Define the minimal set of required columns
        needed_columns = ['label', 'x', 'y', 'width', 'height', 'image_path', 'image_name', 'polygon']
        
        # Validate that all required columns exist in the dataframe
        missing_columns = [col for col in needed_columns if col not in data.columns]
        
        if missing_columns:
            raise ValueError(f"DataFrame is missing required columns: {missing_columns}")
        
        has_polygon = data['polygon'].notnull().any()
        has_x = data['x'].notnull().any()
        has_y = data['y'].notnull().any()
        has_width = data['width'].notnull().any()
        has_height = data['height'].notnull().any()
        
        tasks = []
        
        # Check if polygon values are present
        if has_polygon:
            tasks.extend(['segment', 'detect'])
            
        # Check if bounding box values are present
        elif has_x and has_y and has_width and has_height:
            tasks.append('detect')
        
        # Set the task based on the available data
        if task in tasks:
            self.task = task
        else:
            raise ValueError(f"Only the following tasks are available for this dataset: {tasks}")

        self.data = data
        
        self.output_dir = output_dir
        self.dataset_name = dataset_name
        self.dataset_dir = os.path.join(output_dir, dataset_name)
        
        self.train_ratio = train_ratio
        self.test_ratio = test_ratio
        self.classes = None
        self.class_to_id = None

        # Create dataset directories if they don't exist
        os.makedirs(f"{self.dataset_dir}/images/train", exist_ok=True)
        os.makedirs(f"{self.dataset_dir}/images/val", exist_ok=True)
        os.makedirs(f"{self.dataset_dir}/labels/train", exist_ok=True)
        os.makedirs(f"{self.dataset_dir}/labels/val", exist_ok=True)
        
        # Create test directories if test_ratio > 0
        if self.test_ratio > 0:
            os.makedirs(f"{self.dataset_dir}/images/test", exist_ok=True)
            os.makedirs(f"{self.dataset_dir}/labels/test", exist_ok=True)
            
    def split_dataset(self):
        """
        Split the dataset into training, validation, and test sets based on train_ratio and test_ratio.

        :return: None
        """
        # Get unique images
        unique_images = self.data['image_name'].unique()

        # Shuffle
        np.random.shuffle(unique_images)
        
        # Calculate split indices
        if self.test_ratio > 0:
            # If test_ratio is provided, split into train/val/test
            test_idx = int(len(unique_images) * self.test_ratio)
            train_idx = int(len(unique_images) * self.train_ratio)
            
            test_images = unique_images[:test_idx]
            train_images = unique_images[test_idx:test_idx + train_idx]
            val_images = unique_images[test_idx + train_idx:]
        else:
            # Original behavior: split into train/val only
            split_idx = int(len(unique_images) * self.train_ratio)
            train_images = unique_images[:split_idx]
            val_images = unique_images[split_idx:]
            test_images = []  # Empty, no test set

        # Initialize yolo path columns if they don't exist
        if 'yolo_image_path' not in self.data.columns:
            self.data['yolo_image_path'] = ''
        if 'yolo_label_path' not in self.data.columns:
            self.data['yolo_label_path'] = ''

        # Update paths in the DataFrame
        for image in train_images:
            # Get the image extension   
            ext = os.path.splitext(image)[-1]
            # Create a mask for the image
            mask = self.data['image_name'] == image
            # Update paths for training set
            image_paths = f"{self.dataset_dir}/images/train/{image}"
            self.data.loc[mask, 'yolo_image_path'] = image_paths
            label_paths = f"{self.dataset_dir}/labels/train/{image.replace(ext, '.txt')}"
            self.data.loc[mask, 'yolo_label_path'] = label_paths

        for image in val_images:
            # Get the image extension
            ext = os.path.splitext(image)[-1]
            # Create a mask for the image
            mask = self.data['image_name'] == image
            # Update paths for validation set
            image_paths = f"{self.dataset_dir}/images/val/{image}"
            self.data.loc[mask, 'yolo_image_path'] = image_paths
            label_paths = f"{self.dataset_dir}/labels/val/{image.replace(ext, '.txt')}"
            self.data.loc[mask, 'yolo_label_path'] = label_paths
            
        for image in test_images:
            # Get the image extension
            ext = os.path.splitext(image)[-1]
            # Create a mask for the image
            mask = self.data['image_name'] == image
            # Update paths for test set
            image_paths = f"{self.dataset_dir}/images/test/{image}"
            self.data.loc[mask, 'yolo_image_path'] = image_paths
            label_paths = f"{self.dataset_dir}/labels/test/{image.replace(ext, '.txt')}"
            self.data.loc[mask, 'yolo_label_path'] = label_paths
            
        # Print split information
        print(f"Dataset split: {len(train_images)} train, {len(val_images)} val, {len(test_images)} test images")

    def write_yaml(self):
        """
        Writes a YOLO-formatted dataset yaml file

        :return: None
        """
        self.classes = self.data['label'].unique().tolist()
        self.class_to_id = {class_name: i for i, class_name in enumerate(self.classes)}

        # Create data.yaml with support for test set
        with open(f"{self.dataset_dir}/data.yaml", 'w') as f:
            f.write(f"train: {os.path.join(self.dataset_dir, 'images/train')}\n")
            f.write(f"val: {os.path.join(self.dataset_dir, 'images/val')}\n")
            
            # Add test path if test_ratio > 0
            if self.test_ratio > 0:
                f.write(f"test: {os.path.join(self.dataset_dir, 'images/test')}\n")
                
            f.write(f"nc: {len(self.classes)}\n")
            f.write(f"names: {self.classes}\n")

    def write_labels(self):
        """
        Write YOLO-formatted labels to text files based on the task type.

        :return: None
        """
        if self.task == 'detect':
            self.write_detection_labels()
        elif self.task == 'segment':
            self.write_segmentation_labels()
        else:
            raise ValueError(f"Unsupported task type: {self.task}")

    def write_detection_labels(self):
        """
        Write YOLO-formatted bounding box labels to text files for object detection.

        :return: None
        """
        # Group annotations by label path
        for label_path, label_group in self.data.groupby('yolo_label_path'):
            # Ensure the directory exists
            os.makedirs(os.path.dirname(label_path), exist_ok=True)

            yolo_annotations = []
            for _, row in label_group.iterrows():
                class_id = self.class_to_id[row['label']]

                # Extract image width and height
                image = Image.open(row['image_path'])
                img_width, img_height = image.size

                # Convert to YOLO format (normalized center coordinates) from normalized coordinates
                x_center = row['x'] + row['width'] / 2
                y_center = row['y'] + row['height'] / 2
                w = row['width']
                h = row['height']

                # Create YOLO-formatted annotation string
                yolo_annotation = f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}"
                yolo_annotations.append(yolo_annotation)

            # Save annotations
            with open(label_path, 'w') as f:
                f.write('\n'.join(yolo_annotations))

            print(f"Created detection label: {label_path}")

    def write_segmentation_labels(self):
        """
        Write YOLO-formatted polygon labels to text files for instance segmentation.

        :return: None
        """
        # Group annotations by label path
        for label_path, label_group in self.data.groupby('yolo_label_path'):
            # Ensure the directory exists
            os.makedirs(os.path.dirname(label_path), exist_ok=True)

            yolo_annotations = []
            for _, row in label_group.iterrows():
                class_id = self.class_to_id[row['label']]

                # Extract image width and height
                image = Image.open(row['image_path'])
                img_width, img_height = image.size

                # Check if polygon data exists
                if pd.isnull(row['polygon']):
                    continue

                try:
                    # Polygon data stored as list
                    normalized_points = row['polygon']
                    
                    # Format for YOLO segmentation: class_id x1 y1 x2 y2 ... xn yn
                    points_str = " ".join([f"{p:.6f}" for p in normalized_points])
                    yolo_annotation = f"{class_id} {points_str}"
                    yolo_annotations.append(yolo_annotation)
                    
                except Exception as e:
                    print(f"Error processing polygon for {row['image_name']}: {e}")
                    continue

            # Save annotations
            with open(label_path, 'w') as f:
                f.write('\n'.join(yolo_annotations))

            print(f"Created segmentation label: {label_path}")
        
    def copy_images(self, move_instead_of_copy=False):
        """
        Copy or move images to the appropriate train/val/test directories based on the split.

        :param move_instead_of_copy: If True, move images instead of copying them
        :return: None
        """
        # Check if the DataFrame has the required columns
        if not all(col in self.data.columns for col in ['image_path', 'image_name', 'yolo_image_path']):
            raise ValueError("DataFrame must contain 'image_path', 'image_name' and 'yolo_image_path' columns")

        action_name = "Moving" if move_instead_of_copy else "Copying"

        # Copy/move images to train, val, and test directories
        for _, row in tqdm(self.data.iterrows(), total=len(self.data), desc=f"{action_name} images"):
            # Get source and destination paths
            src_path = row['image_path']
            dst_path = os.path.dirname(row['yolo_image_path'])

            dst_dir = os.path.dirname(dst_path)
            os.makedirs(dst_dir, exist_ok=True)

            # Check if source exists and isn't already at destination
            if os.path.exists(src_path) and src_path != dst_path:
                if move_instead_of_copy:
                    shutil.move(src_path, dst_path)
                else:
                    shutil.copy(src_path, dst_path)

    def render_examples(self, output_dir, num_examples=10, include_boxes=True, include_labels=True):
        """
        Render examples from the dataset with annotations and save the images.

        :param output_dir: Directory to save rendered images
        :param num_examples: Number of examples to render (default: 10)
        :param include_boxes: Whether to include bounding boxes in the rendering
        :param include_labels: Whether to include class labels in the rendering
        :return: None
        """
        os.makedirs(output_dir, exist_ok=True)

        # Get unique image paths
        unique_images = self.data['image_path'].unique()

        # Select a subset of images
        if len(unique_images) > num_examples:
            import random
            selected_images = random.sample(list(unique_images), num_examples)
        else:
            selected_images = unique_images

        # Create the annotation objects
        box_annotator = sv.BoundingBoxAnnotator()
        label_annotator = sv.LabelAnnotator()

        # Process each selected image
        for image_path in tqdm(selected_images, desc="Rendering Examples"):
            try:
                # Load image
                image = cv2.imread(image_path)
                if image is None:
                    print(f"Warning: Could not read image {image_path}")
                    continue
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                # Get image dimensions
                img_height, img_width = image.shape[:2]

                # Get annotations for this image
                annotations_df = self.data[self.data['image_path'] == image_path]

                # Create detections object
                boxes = []
                class_ids = []

                for _, row in annotations_df.iterrows():
                    # Convert YOLO format back to pixel coordinates
                    x_center = row['x'] + row['width'] / 2
                    y_center = row['y'] + row['height'] / 2
                    w = row['width']
                    h = row['height']

                    # Calculate xyxy format
                    x1 = x_center - w / 2
                    y1 = y_center - h / 2
                    x2 = x1 + w
                    y2 = y1 + h

                    boxes.append([x1, y1, x2, y2])
                    class_ids.append(self.class_to_id[row['label']])

                # Create supervision detection object
                detections = sv.Detections(
                    xyxy=np.array(boxes),
                    class_id=np.array(class_ids)
                )

                # Annotate image
                if include_boxes:
                    image = box_annotator.annotate(
                        scene=image, detections=detections)

                if include_labels:
                    # Map class IDs back to class names
                    id_to_class = {v: k for k, v in self.class_to_id.items()}
                    labels = [id_to_class[class_id] for class_id in detections.class_id]
                    image = label_annotator.annotate(scene=image, detections=detections, labels=labels)

                # Save output image
                output_file = os.path.join(output_dir, os.path.basename(image_path))
                cv2.imwrite(output_file, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

            except Exception as e:
                print(f"Error processing {image_path}: {e}")

        print(f"Rendered {len(selected_images)} examples to {output_dir}")

    def process_dataset(self, move_images=False):
        """
        Process the entire dataset: split, write labels and generate YAML.

        :param move_images: If True, move images instead of copying them
        :return: None
        """
        print(f"Processing YOLO dataset with {len(self.data)} annotations...")
        # Split the dataset, create paths
        self.split_dataset()
        # Write YAML file for dataset
        self.write_yaml()
        # Write labels based on task type
        self.write_labels()
        # Move or copy images 
        self.copy_images(move_instead_of_copy=move_images)
        print(f"Dataset created at {self.dataset_dir}")
        print(f"Classes: {self.classes}")

        # Optional: Render examples
        examples_dir = os.path.join(self.dataset_dir, "examples")
        self.render_examples(examples_dir)
        
        
# ----------------------------------------------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------------------------------------------


def main():
    """
    Main function to process the YOLO dataset from command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Create a YOLO-formatted dataset for object detection")
    
    parser.add_argument("--dataframe", "-d", type=str, required=True, 
                        help="Path to the CSV file containing annotation data")
    
    parser.add_argument("--output_dir", "-o", type=str, required=True, 
                        help="Base directory where datasets will be created/stored")
    
    parser.add_argument("--dataset_name", "-n", type=str, default="yolo_dataset",
                        help="Name of the dataset (subdirectory inside output_dir)")
    
    parser.add_argument("--train_ratio", "-s", type=float, default=0.8, 
                        help="Train/val split ratio (default: 0.8). If test_ratio > 0, this becomes train ratio.")
    
    parser.add_argument("--test_ratio", "-t", type=float, default=0.0,
                        help="Test split ratio (default: 0.0). If > 0, data will be split into train/val/test.")
    
    parser.add_argument("--task", "-k", type=str, default="detect",
                        help="Task type ('detect' or 'segment')")
    
    parser.add_argument("--move_images", "-m", action="store_true", 
                        help="Move images instead of copying them")
                        
    args = parser.parse_args()

    # Load dataframe
    print(f"Loading annotations from {args.dataframe}")
    try:
        df = pd.read_csv(args.dataframe)
    except Exception as e:
        print(f"Error loading dataframe: {e}")
        return
        
    print(f"Loaded dataframe with {len(df)} rows")
    
    # Create and process dataset
    dataset = YOLODataset(
        data=df,
        output_dir=args.output_dir,
        dataset_name=args.dataset_name,
        train_ratio=args.train_ratio,
        test_ratio=args.test_ratio,
        task=args.task
    )
    
    # Process the dataset
    dataset.process_dataset(move_images=args.move_images)
    print(f"YOLO dataset processed successfully. Output directory: {os.path.join(args.output_dir, args.dataset_name)}")
    print("Done.")


if __name__ == "__main__":
    main()
