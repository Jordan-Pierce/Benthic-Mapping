import os
import glob
import argparse
from datetime import datetime

import cv2
import numpy as np

import fiftyone as fo
import fiftyone.zoo as foz
import fiftyone.brain as fob
import fiftyone.utils.random as four

def create_or_load_dataset(dataset_name, image_dir):
    """Creates a new dataset or loads existing one"""
    if dataset_name in fo.list_datasets():
        print(f"Loading existing dataset: {dataset_name}")
        return fo.load_dataset(dataset_name)
    
    print(f"Creating new dataset: {dataset_name}")
    dataset = fo.Dataset(dataset_name)
    
    # Get all image files in directory
    filepaths = glob.glob(os.path.join(image_dir, "*.*"))
    if not filepaths:
        raise ValueError(f"No files found in directory: {image_dir}")
    
    # Create samples
    samples = []
    for filepath in filepaths:
        filename = os.path.basename(filepath)
        sample = fo.Sample(filepath=filepath)
        samples.append(sample)
    
    dataset.add_samples(samples)
    return dataset

def compute_embeddings(dataset):
    """Compute embeddings for all images in the dataset"""
    filepaths = [sample.filepath for sample in dataset]
    embeddings = np.array([
        cv2.resize(cv2.imread(f, cv2.IMREAD_UNCHANGED), (64, 64), interpolation=cv2.INTER_AREA).ravel()
        for f in filepaths
    ])
    return embeddings


def main():

    parser = argparse.ArgumentParser(description='Process images and create a FiftyOne dataset with UMAP visualization')
    
    parser.add_argument('--image_dir', type=str, required=True,
                      help='Path to directory containing images')
    
    args = parser.parse_args()
    
    # Validate directory path
    if not os.path.isdir(args.image_dir):
        raise ValueError(f"Directory not found: {args.image_dir}")
    
    # Create dataset name from directory name
    dataset_name = os.path.basename(os.path.normpath(args.image_dir))
    
    # Create or load dataset
    dataset = create_or_load_dataset(dataset_name, args.image_dir)
    
    # Generate a unique brain key using timestamp
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    brain_key = f"{dataset_name}_umap_{timestamp}"
    
    # Compute embeddings
    print("Computing embeddings...")
    embeddings = compute_embeddings(dataset)
    
    # Compute 2D representation
    print("Computing UMAP visualization...")
    results = fob.compute_visualization(
        dataset,
        embeddings=embeddings,
        num_dims=2,
        method="umap",
        brain_key=brain_key,
        verbose=True,
        seed=51,
    )
    
    dataset.load_brain_results(brain_key)
    
    print(f"Launching FiftyOne App with visualization '{brain_key}'")
    session = fo.launch_app(dataset)
    session.wait()

if __name__ == "__main__":
    main()