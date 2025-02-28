import argparse
import glob
import os
import shutil

import cv2
import numpy as np
import pandas as pd
import supervision as sv
import yaml
from tqdm.auto import tqdm

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

    def __init__(self, data, dataset_dir, split_ratio=0.8, required_columns=None):
        """
        Initialize the YOLO dataset.

        :param data: pandas DataFrame containing annotation data
        :param dataset_dir: directory where the dataset will be created
        :param split_ratio: train/val split ratio (default: 0.8)
        :param required_columns: optional dict mapping standard column names to actual df column names
        """
        # Define the minimal set of required columns
        self.standard_columns = {
            'label': 'label',           # Class name
            'x': 'x',                   # X coordinate (top-left)
            'y': 'y',                   # Y coordinate (top-left)
            'width': 'width',           # Width of bounding box
            'height': 'height',         # Height of bounding box
            'image_path': 'image_path',  # Path to image file
            'image_name': 'image_name',  # Name of image file
        }

        # Handle backward compatibility for frame_path and frame_name
        if required_columns:
            # If provided mapping uses old names, update them to new names
            if 'frame_path' in required_columns:
                required_columns['image_path'] = required_columns.pop('frame_path')
            if 'frame_name' in required_columns:
                required_columns['image_name'] = required_columns.pop('frame_name')
            self.standard_columns.update(required_columns)
        
        # For backward compatibility, rename frame_path/frame_name to image_path/image_name if they exist
        if 'frame_path' in data.columns and 'image_path' not in data.columns:
            data = data.rename(columns={'frame_path': 'image_path'})
        if 'frame_name' in data.columns and 'image_name' not in data.columns:
            data = data.rename(columns={'frame_name': 'image_name'})

        # Validate that all required columns exist in the dataframe
        missing_columns = [col for col in self.standard_columns.values()
                           if col not in data.columns]
        if missing_columns:
            raise ValueError(
                f"DataFrame is missing required columns: {missing_columns}")

        self.data = data
        self.dataset_dir = dataset_dir
        self.split_ratio = split_ratio
        self.classes = None
        self.class_to_id = None

        # Create dataset directories if they don't exist
        os.makedirs(f"{dataset_dir}/images/train", exist_ok=True)
        os.makedirs(f"{dataset_dir}/images/val", exist_ok=True)
        os.makedirs(f"{dataset_dir}/labels/train", exist_ok=True)
        os.makedirs(f"{dataset_dir}/labels/val", exist_ok=True)

    @classmethod
    def from_existing(cls, dataset_dir):
        """
        Initialize a YOLODataset from an existing YOLO dataset directory.

        :param dataset_dir: Path to the existing YOLO dataset directory
        :return: YOLODataset instance
        """
        # Check if the directory exists and contains required files
        if not os.path.exists(dataset_dir):
            raise ValueError(f"Dataset directory {dataset_dir} does not exist")

        yaml_path = os.path.join(dataset_dir, "data.yaml")
        if not os.path.exists(yaml_path):
            raise ValueError(f"YAML file not found at {yaml_path}")

        # Load the YAML config
        with open(yaml_path, 'r') as f:
            yaml_data = yaml.safe_load(f)

        # Extract class information
        classes = yaml_data.get('names', [])
        if not classes:
            raise ValueError("No class names found in the YAML file")

        class_to_id = {class_name: i for i, class_name in enumerate(classes)}

        # Find the train and val directories
        train_dir = yaml_data.get('train')
        val_dir = yaml_data.get('val')

        # Process train and val directories
        rows = []

        for data_split, img_dir in [('train', train_dir), ('val', val_dir)]:
            if not img_dir or not os.path.exists(img_dir):
                continue

            # Get all image files
            image_files = []
            for ext in ['*.jpg', '*.jpeg', '*.png']:
                image_files.extend(glob.glob(os.path.join(img_dir, ext)))

            for img_path in tqdm(image_files, desc=f"Loading {data_split} images"):
                img_filename = os.path.basename(img_path)
                img_name_no_ext = os.path.splitext(img_filename)[0]

                # Find corresponding label file
                if 'images' in img_dir:
                    label_dir = img_dir.replace('images', 'labels')
                else:
                    label_dir = os.path.join(dataset_dir, 'labels', data_split)

                label_path = os.path.join(label_dir, f"{img_name_no_ext}.txt")

                # Load image to get dimensions
                img = cv2.imread(img_path)
                if img is None:
                    continue
                img_height, img_width = img.shape[:2]

                # Check if label file exists
                if os.path.exists(label_path):
                    # Read annotations
                    with open(label_path, 'r') as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) >= 5:
                                class_id = int(parts[0])
                                x_center = float(parts[1]) * img_width
                                y_center = float(parts[2]) * img_height
                                w = float(parts[3]) * img_width
                                h = float(parts[4]) * img_height

                                # Convert to top-left corner format
                                x = x_center - w / 2
                                y = y_center - h / 2

                                # Find class label from ID
                                id_to_class = {v: k for k,
                                               v in class_to_id.items()}
                                label = id_to_class.get(class_id, "unknown")

                                # Create row with only required fields
                                row_dict = {
                                    'image_name': img_filename,
                                    'image_path': img_path,
                                    'label_path': label_path,
                                    'x': x,
                                    'y': y,
                                    'width': w,
                                    'height': h,
                                    'label': label
                                }
                                rows.append(row_dict)

        # Create DataFrame from rows
        if rows:
            df = pd.DataFrame(rows)
        else:
            # Create an empty DataFrame with required columns
            df = pd.DataFrame(columns=['image_name', 'image_path', 'label_path',
                                       'x', 'y', 'width', 'height', 'label'])

        # Create the YOLODataset instance
        dataset = cls(data=df, dataset_dir=dataset_dir)
        dataset.classes = classes
        dataset.class_to_id = class_to_id

        return dataset

    @classmethod
    def merge_datasets(cls, datasets, output_dir, split_ratio=0.8, move_images=False):
        """
        Merge multiple YOLODataset objects into a single dataset.
        
        :param datasets: List of YOLODataset objects to merge
        :param output_dir: Directory where the merged dataset will be created
        :param split_ratio: Train/val split ratio for the merged dataset
        :param move_images: If True, move images instead of copying them
        :return: New YOLODataset instance with merged data
        """
        if not datasets:
            raise ValueError("No datasets provided for merging")
            
        # Create combined DataFrame
        combined_df = pd.DataFrame()
        for i, dataset in enumerate(datasets):
            # Make a copy to avoid modifying original
            df_copy = dataset.data.copy()
            
            # Add source dataset identifier (useful for debugging)
            df_copy['source_dataset'] = i
            
            # Update paths to point to original image locations
            df_copy['original_image_path'] = df_copy['image_path'].copy()
            
            # Append to combined DataFrame
            combined_df = pd.concat([combined_df, df_copy], ignore_index=True)
        
        print(f"Merged {len(datasets)} datasets with a total of {len(combined_df)} annotations")
        
        # Create new dataset
        merged_dataset = cls(
            data=combined_df,
            dataset_dir=output_dir,
            split_ratio=split_ratio
        )
        
        # Process the merged dataset
        merged_dataset.process_dataset(move_images=move_images)
        
        return merged_dataset

    def write_yaml(self):
        """
        Writes a YOLO-formatted dataset yaml file

        :return: None
        """
        self.classes = self.data['label'].unique().tolist()
        self.class_to_id = {class_name: i for i,
                            class_name in enumerate(self.classes)}

        # Create data.yaml
        with open(f"{self.dataset_dir}/data.yaml", 'w') as f:
            f.write(
                f"train: {os.path.join(self.dataset_dir, 'images/train')}\n")
            f.write(f"val: {os.path.join(self.dataset_dir, 'images/val')}\n")
            f.write(f"nc: {len(self.classes)}\n")
            f.write(f"names: {self.classes}\n")

    def write_labels(self):
        """
        Write YOLO-formatted labels to text files.

        :return: None
        """
        # Get column name mappings
        label_col = self.standard_columns['label']
        x_col = self.standard_columns['x']
        y_col = self.standard_columns['y']
        width_col = self.standard_columns['width']
        height_col = self.standard_columns['height']

        # Group annotations by label path
        for label_path, label_group in self.data.groupby('label_path'):
            # Ensure the directory exists
            os.makedirs(os.path.dirname(label_path), exist_ok=True)

            yolo_annotations = []
            for _, row in label_group.iterrows():
                class_id = self.class_to_id[row[label_col]]

                # Extract image width and height (you might need to add this to your DataFrame)
                # For now, assuming the coordinates are already in the right format
                img_width = 1.0  # replace with actual width if available
                img_height = 1.0  # replace with actual height if available

                # Convert to YOLO format (normalized center coordinates)
                x_center = (row[x_col] + row[width_col] / 2) / img_width
                y_center = (row[y_col] + row[height_col] / 2) / img_height
                w = row[width_col] / img_width
                h = row[height_col] / img_height

                # Create YOLO-formatted annotation string
                yolo_annotation = f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}"
                yolo_annotations.append(yolo_annotation)

            # Save annotations
            with open(label_path, 'w') as f:
                f.write('\n'.join(yolo_annotations))

            print(f"Created label: {label_path}")

    def split_dataset(self):
        """
        Split the dataset into training and validation sets.

        :return: None
        """
        # Get column name for image_name
        image_name_col = self.standard_columns['image_name']

        # Get unique images
        unique_images = self.data[image_name_col].unique()

        # Shuffle and split
        np.random.shuffle(unique_images)
        split_idx = int(len(unique_images) * self.split_ratio)

        train_images = unique_images[:split_idx]
        val_images = unique_images[split_idx:]

        # Update paths in the DataFrame
        for image in train_images:
            mask = self.data[image_name_col] == image
            # Update paths for training set
            image_paths = self.data.loc[mask, 'image_path'].str.replace('/images/', '/images/train/')
            self.data.loc[mask, 'image_path'] = image_paths

            label_paths = self.data.loc[mask, 'label_path'].str.replace('/labels/', '/labels/train/')
            self.data.loc[mask, 'label_path'] = label_paths

        for image in val_images:
            mask = self.data[image_name_col] == image
            # Update paths for validation set
            image_paths = self.data.loc[mask, 'image_path'].str.replace('/images/', '/images/val/')
            self.data.loc[mask, 'image_path'] = image_paths

            label_paths = self.data.loc[mask, 'label_path'].str.replace('/labels/', '/labels/val/')
            self.data.loc[mask, 'label_path'] = label_paths

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

        # Get column mappings
        image_path_col = self.standard_columns['image_path']
        label_col = self.standard_columns['label']
        x_col = self.standard_columns['x']
        y_col = self.standard_columns['y']
        width_col = self.standard_columns['width']
        height_col = self.standard_columns['height']

        # Get unique image paths
        unique_images = self.data[image_path_col].unique()

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
                annotations_df = self.data[self.data[image_path_col] == image_path]

                # Create detections object
                boxes = []
                class_ids = []

                for _, row in annotations_df.iterrows():
                    # Convert YOLO format back to pixel coordinates
                    x_center = row[x_col] + row[width_col] / 2
                    y_center = row[y_col] + row[height_col] / 2
                    w = row[width_col]
                    h = row[height_col]

                    # Calculate xyxy format
                    x1 = x_center - w / 2
                    y1 = y_center - h / 2
                    x2 = x1 + w
                    y2 = y1 + h

                    boxes.append([x1, y1, x2, y2])
                    class_ids.append(self.class_to_id[row[label_col]])

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
                    labels = [id_to_class[class_id]
                              for class_id in detections.class_id]
                    image = label_annotator.annotate(
                        scene=image, detections=detections, labels=labels)

                # Save output image
                output_file = os.path.join(
                    output_dir, os.path.basename(image_path))
                cv2.imwrite(output_file, cv2.cvtColor(
                    image, cv2.COLOR_RGB2BGR))

            except Exception as e:
                print(f"Error processing {image_path}: {e}")

        print(f"Rendered {len(selected_images)} examples to {output_dir}")

    def copy_images(self, move_instead_of_copy=False):
        """
        Copy or move images to the appropriate train/val directories based on the split.

        :param move_instead_of_copy: If True, move images instead of copying them
        :return: None
        """
        # Get column mappings
        image_path_col = self.standard_columns['image_path']
        image_name_col = self.standard_columns['image_name']

        # Check if the DataFrame has the required columns
        if not all(col in self.data.columns for col in [image_path_col, image_name_col]):
            raise ValueError(f"DataFrame must contain '{image_path_col}' and '{image_name_col}' columns")

        action_name = "Moving" if move_instead_of_copy else "Copying"

        # Copy/move images to train and val directories
        for _, row in tqdm(self.data.iterrows(), total=len(self.data), desc=f"{action_name} images"):
            # Get source and destination paths
            src_path = row[image_path_col]

            # Images should be in either train or val subdirectory
            if 'train' in src_path:
                dst_dir = os.path.join(self.dataset_dir, 'images', 'train')
            else:
                dst_dir = os.path.join(self.dataset_dir, 'images', 'val')

            os.makedirs(dst_dir, exist_ok=True)
            dst_path = os.path.join(dst_dir, row[image_name_col])

            # Check if source exists and isn't already at destination
            if os.path.exists(src_path) and src_path != dst_path:
                if move_instead_of_copy:
                    shutil.move(src_path, dst_path)
                else:
                    shutil.copy(src_path, dst_path)

    def process_dataset(self, move_images=False):
        """
        Process the entire dataset: split, write labels and generate YAML.

        :param move_images: If True, move images instead of copying them
        :return: None
        """
        print(f"Processing YOLO dataset with {len(self.data)} annotations...")
        self.split_dataset()
        self.write_yaml()
        self.write_labels()
        # Updated to support moving
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
    
    parser.add_argument("--dataframe", "-d", type=str, 
                        help="Path to the CSV file containing annotation data")
    
    parser.add_argument("--output_dir", "-o", type=str, 
                        help="Directory where the dataset will be created")
    
    parser.add_argument("--split_ratio", "-s", type=float, default=0.8, 
                        help="Train/val split ratio (default: 0.8)")
    
    parser.add_argument("--move_images", "-m", action="store_true", 
                        help="Move images instead of copying them")
    
    parser.add_argument("--column_map", "-c", type=str, 
                        help="JSON string for column mapping (optional)")
    
    parser.add_argument("--from_existing", "-e", action="store_true", 
                        help="Load from existing YOLO dataset directory") 
    
    # Add another argument for merging datasets
    parser.add_argument("--merge", "-M", type=str, nargs='+',
                        help="Merge multiple existing YOLO dataset directories")
                        
    args = parser.parse_args()

    # Check required arguments
    if args.from_existing:
        if not args.output_dir:
            parser.error("--output_dir is required when loading from existing dataset")
        
        # Load from existing dataset
        print(f"Loading existing YOLO dataset from {args.output_dir}")
        dataset = YOLODataset.from_existing(args.output_dir)
        
    elif args.merge:
        if not args.output_dir:
            parser.error("--output_dir is required when merging datasets")
            
        print(f"Merging {len(args.merge)} datasets...")
        datasets = []
        for dataset_path in args.merge:
            print(f"Loading dataset from {dataset_path}")
            datasets.append(YOLODataset.from_existing(dataset_path))
            
        # Merge datasets
        merged_dataset = YOLODataset.merge_datasets(
            datasets=datasets,
            output_dir=args.output_dir,
            split_ratio=args.split_ratio,
            move_images=args.move_images
        )
        
        print(f"Merged dataset created successfully at {args.output_dir}")
        return
        
    else:
        if not args.dataframe or not args.output_dir:
            parser.error("--dataframe and --output_dir are required when creating a new dataset")
        
        # Load dataframe
        print(f"Loading annotations from {args.dataframe}")
        try:
            df = pd.read_csv(args.dataframe)
        except Exception as e:
            print(f"Error loading dataframe: {e}")
            return
            
        print(f"Loaded dataframe with {len(df)} rows")
        
        # Parse column mapping if provided
        column_map = None
        if args.column_map:
            try:
                import json
                column_map = json.loads(args.column_map)
                print(f"Using column mapping: {column_map}")
            except json.JSONDecodeError:
                print("Warning: Invalid JSON for column mapping, using default")
        
        # Create and process dataset
        dataset = YOLODataset(
            data=df,
            dataset_dir=args.output_dir,
            split_ratio=args.split_ratio,
            required_columns=column_map
        )
    
    # Process the dataset
    dataset.process_dataset(move_images=args.move_images)
    print(f"YOLO dataset processed successfully. Output directory: {args.output_dir}")
    print("Done.")


if __name__ == "__main__":
    main()
