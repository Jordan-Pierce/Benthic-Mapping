import os
import glob
import argparse
from tqdm import tqdm
from datetime import datetime

import cv2
import numpy as np
import fiftyone as fo
import fiftyone.brain as fob


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class FiftyOneDatasetViewer:
    def __init__(self, image_dir, dataset_name=None, nickname=None):
        self.image_dir = image_dir
        self.dataset_name = dataset_name or os.path.basename(os.path.normpath(image_dir))
        self.nickname = nickname or self.dataset_name  # Use dataset_name as default nickname
        self.dataset = None
        self.brain_key = None

    def create_or_load_dataset(self):
        """Creates a new dataset or loads existing one"""
        if self.nickname in fo.list_datasets():
            overwrite = input(f"Dataset with nickname '{self.nickname}' already exists. Overwrite? (y/n): ").lower()
            if overwrite == 'y':
                print(f"Overwriting existing dataset: {self.nickname}")
                fo.delete_dataset(self.nickname)
                self.dataset = fo.Dataset(self.nickname)
            else:
                print("Loading existing dataset.")
                self.dataset = fo.load_dataset(self.nickname)
                return
        else:
            print(f"Creating new dataset: {self.nickname}")
            self.dataset = fo.Dataset(self.nickname)

        filepaths = glob.glob(os.path.join(self.image_dir, "*.*"))
        if not filepaths:
            raise ValueError(f"No files found in directory: {self.image_dir}")

        samples = []
        for filepath in tqdm(filepaths, desc="Processing images"):
            img = cv2.imread(filepath)
            if img is None:
                continue

            filename = os.path.basename(filepath)
            file_stats = os.stat(filepath)

            sample = fo.Sample(filepath=filepath)
            sample.metadata = fo.ImageMetadata(
                size_bytes=file_stats.st_size,
                mime_type=f"image/{os.path.splitext(filename)[1][1:]}",
                width=img.shape[1],
                height=img.shape[0],
                num_channels=img.shape[2] if len(img.shape) > 2 else 1,
            )

            sample["file_extension"] = os.path.splitext(filename)[1]
            sample["relative_path"] = os.path.relpath(filepath, self.image_dir)
            sample["creation_date"] = datetime.fromtimestamp(file_stats.st_ctime)
            sample["modification_date"] = datetime.fromtimestamp(file_stats.st_mtime)
            sample["mean_color"] = img.mean(axis=(0, 1)).tolist()
            sample["mean_brightness"] = img.mean()

            samples.append(sample)

        # Add fields to dataset schema
        self.dataset.add_sample_field("file_extension", fo.StringField)
        self.dataset.add_sample_field("relative_path", fo.StringField)
        self.dataset.add_sample_field("creation_date", fo.DateTimeField)
        self.dataset.add_sample_field("modification_date", fo.DateTimeField)
        self.dataset.add_sample_field("mean_color", fo.VectorField)
        self.dataset.add_sample_field("mean_brightness", fo.FloatField)

        self.dataset.add_samples(samples)

    def compute_embeddings(self):
        """Compute embeddings for all images in the dataset"""
        filepaths = [sample.filepath for sample in self.dataset]
        return np.array([
            cv2.resize(cv2.imread(f, cv2.IMREAD_UNCHANGED), (64, 64),
                       interpolation=cv2.INTER_AREA).ravel()
            for f in filepaths
        ])

    def create_visualization(self, embeddings):
        """Create UMAP visualization"""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        self.brain_key = f"{self.dataset_name}_umap_{timestamp}"

        return fob.compute_visualization(
            self.dataset,
            embeddings=embeddings,
            num_dims=2,
            method="umap",
            brain_key=self.brain_key,
            verbose=True,
            seed=51,
        )

    def process_dataset(self):
        """Main processing method"""
        # Create or load dataset
        self.create_or_load_dataset()

        print("Computing embeddings...")
        # Compute embeddings
        embeddings = self.compute_embeddings()

        print("Computing UMAP visualization...")
        # Create UMAP visualization
        self.create_visualization(embeddings)
        self.dataset.load_brain_results(self.brain_key)
        
    def visualize(self):
        """visualize the dataset"""
        # Process the dataset
        self.process_dataset()

        print(f"Launching FiftyOne App with visualization '{self.brain_key}'")
        # Launch FiftyOne App
        session = fo.launch_app(self.dataset)
        session.wait()


# ----------------------------------------------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description='Process images and create a FiftyOne dataset with UMAP visualization',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="""
        Examples:
        python FiftyOne_Clustering.py --image_dir /path/to/images
        python FiftyOne_Clustering.py --dataset_name existing_dataset
        python FiftyOne_Clustering.py --list_datasets
        """
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--image_dir', type=str,
                       help='Path to directory containing images')

    group.add_argument('--dataset_name', type=str,
                       help='Name of existing FiftyOne dataset to load')

    group.add_argument('--list_datasets', action='store_true',
                       help='List all available FiftyOne datasets and exit')
    
    group.add_argument('--delete_dataset', type=str,
                       help='Delete an existing dataset.')

    parser.add_argument('--nickname', type=str, 
                        help='Optional: A nickname for the dataset.')

    args = parser.parse_args()
    
    if args.delete_dataset:
        if args.delete_dataset in fo.list_datasets():
            print(f"Deleting dataset: {args.delete_dataset}")
            fo.delete_dataset(args.delete_dataset)
        else:
            print(f"Dataset not found: {args.delete_dataset}")
        return

    if args.list_datasets:
        datasets = fo.list_datasets()
        if datasets:
            print("Available datasets:")
            for dataset in datasets:
                print(f"  - {dataset}")
        else:
            print("No FiftyOne datasets available.")
        return

    if args.image_dir:
        if not os.path.isdir(args.image_dir):
            raise ValueError(f"Directory not found: {args.image_dir}")
        viewer = FiftyOneDatasetViewer(args.image_dir, nickname=args.nickname)
    else:
        if args.dataset_name not in fo.list_datasets():
            raise ValueError(f"Dataset not found: {args.dataset_name}")
        viewer = FiftyOneDatasetViewer(None, dataset_name=args.dataset_name, nickname=args.nickname)

    viewer.visualize()


if __name__ == "__main__":
    main()