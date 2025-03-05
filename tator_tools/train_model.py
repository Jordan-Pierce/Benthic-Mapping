import os
import glob
import datetime
import argparse
import traceback

import numpy as np

from ultralytics import YOLO
from ultralytics import RTDETR
import ultralytics.data.build as build
from ultralytics.data.dataset import YOLODataset


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class YOLOWeightedDataset(YOLODataset):
    def __init__(self, *args, mode="train", **kwargs):
        """
        Initialize the WeightedDataset.

        Args:
            class_weights (list or numpy array): A list or array of weights corresponding to each class.
        """

        super(YOLOWeightedDataset, self).__init__(*args, **kwargs)

        self.train_mode = "train" in self.prefix

        # You can also specify weights manually instead
        self.count_instances()
        class_weights = np.sum(self.counts) / self.counts

        # Aggregation function
        self.agg_func = np.mean

        self.class_weights = np.array(class_weights)
        self.weights = self.calculate_weights()
        self.probabilities = self.calculate_probabilities()

    def count_instances(self):
        """
        Count the number of instances per class

        Returns:
            dict: A dict containing the counts for each class.
        """
        self.counts = [0 for i in range(len(self.data["names"]))]
        for label in self.labels:
            cls = label['cls'].reshape(-1).astype(int)
            for id in cls:
                self.counts[id] += 1

        self.counts = np.array(self.counts)
        self.counts = np.where(self.counts == 0, 1, self.counts)

    def calculate_weights(self):
        """
        Calculate the aggregated weight for each label based on class weights.

        Returns:
            list: A list of aggregated weights corresponding to each label.
        """
        weights = []
        for label in self.labels:
            cls = label['cls'].reshape(-1).astype(int)

            # Give a default weight to background class
            if cls.size == 0:
                weights.append(1)
                continue

            # Take mean of weights
            # You can change this weight aggregation function to aggregate weights differently
            weight = self.agg_func(self.class_weights[cls])
            weights.append(weight)
        return weights

    def calculate_probabilities(self):
        """
        Calculate and store the sampling probabilities based on the weights.

        Returns:
            list: A list of sampling probabilities corresponding to each label.
        """
        total_weight = sum(self.weights)
        probabilities = [w / total_weight for w in self.weights]
        return probabilities

    def __getitem__(self, index):
        """
        Return transformed label information based on the sampled index.
        """
        # Don't use for validation
        if not self.train_mode:
            return self.transforms(self.get_image_and_label(index))
        else:
            index = np.random.choice(len(self.labels), p=self.probabilities)
            return self.transforms(self.get_image_and_label(index))


# ----------------------------------------------------------------------------------------------------------------------
# ModelTrainer
# ----------------------------------------------------------------------------------------------------------------------
    

class ModelTrainer:
    def __init__(self, training_data, epochs=50, weights="yolov8m.pt", model_path="", output_dir=None, name=None,
                 task='detect', cache=False, device=0, half=False, imgsz=640, patience=10, batch=0.5, save_period=10, 
                 plots=False, single_cls=False, weighted=False):
        """
        Initialize the ModelTrainer with explicit parameters.

        :param training_data: Path to training data (YAML or folder)
        :param epochs: Number of training epochs
        :param weights: Initial weights file
        :param model_path: Path to pre-trained model
        :param output_dir: Output directory for the project
        :param name: Directory to save run results
        :param task: Task type (detect, classify, segment)
        :param cache: Use caching for datasets
        :param device: Device to run on
        :param half: Use half precision
        :param imgsz: Image size for training
        :param patience: Patience for early stopping
        :param batch: Batch size as a fraction of GPU memory
        :param save_period: Save checkpoint every x epochs
        :param plots: Generate plots
        :param single_cls: Train as single-class dataset
        :param weighted: Use weighted sampling for training
        """
        self.training_data = training_data
        self.num_epochs = epochs
        self.weights = weights
        self.model_path = model_path
        self.task = task
        self.cache = cache
        self.device = device
        self.half = half
        self.imgsz = imgsz if imgsz else 640
        self.patience = patience
        self.batch = batch
        self.save_period = save_period
        self.plots = plots
        self.single_cls = single_cls
        self.weighted = weighted

        # Set the output directory and run name
        self.name = name if name else self.get_run_name()
        self.output_dir = f"{output_dir}"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Set the training data
        if os.path.exists(self.training_data):
            self.training_data = training_data

        self.target_model = self.load_model()
        
    @staticmethod
    def get_now():
        """
        Get current datetime as a formatted string.

        :return: Formatted current datetime
        """
        return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    def get_run_name(self):
        """
        Generate a run name with an optional prefix.

        :return: Generated run name
        """
        return f"{self.get_now()}_{self.task}_{self.weights.split('.')[0]}"

    def load_model(self):
        """
        Load the model from the given path.

        :return: Loaded model
        """
        try:

            if "yolo" in self.weights.lower():
                model = YOLO(self.weights)
            elif "rtdetr" in self.weights.lower():
                model = RTDETR(self.weights)
            else:
                raise ValueError(f"Unrecognized weights type: {self.weights}")

            if os.path.exists(self.model_path):
                model.load_weights(self.model_path)

            return model

        except Exception as e:
            raise Exception(f"ERROR: Failed to load model.\n{e}")

    def train_model(self):
        """
        Train the model with the specified parameters.

        :return: Training results
        """
        if self.weighted:
            build.YOLODataset = YOLOWeightedDataset

        results = self.target_model.train(
            data=self.training_data,
            task=self.task,
            cache=self.cache,
            device=self.device,
            half=self.half,
            imgsz=self.imgsz,
            epochs=self.num_epochs,
            patience=self.patience,
            batch=self.batch,
            project=self.output_dir,
            name=self.name,
            save_period=self.save_period,
            plots=self.plots,
            single_cls=self.single_cls
        )
        print("Training completed.")
        return results

    def evaluate_model(self):
        try:
            self.target_model.val(
                data=self.training_data,
                split='test',
                save_json=True,
                plots=True
            )
        except Exception as e:
            print(f"WARNING: Failed to evaluate model.\n{e}")


# ----------------------------------------------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------------------------------------------

def main():
    """

    :return:
    """
    parser = argparse.ArgumentParser(description="Train a model for object detection.")

    parser.add_argument("--training_data", type=str, required=True,
                        help="Path to the training data (YAML, or Folder)")

    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs")

    parser.add_argument("--weights", type=str, default="yolov8m.pt",
                        help="Initial weights file")

    parser.add_argument("--model_path", type=str, default="",
                        help="Path to the pre-trained model")

    parser.add_argument("--root_dir", type=str, default=None,
                        help="Root directory for the project")

    parser.add_argument("--run_dir", type=str, default=None,
                        help="Directory to save run results")

    parser.add_argument("--task", type=str, default='detect',
                        help="Task type (e.g., 'detect', 'classify', 'segment')")

    parser.add_argument("--cache", action="store_true",
                        help="Use caching for datasets")

    parser.add_argument("--device", type=int, default=0,
                        help="Device to run on (e.g., cuda device)")

    parser.add_argument("--half", action="store_true",
                        help="Use half precision")
    
    parser.add_argument("--img_size", type=int, default=640,
                        help="Image size for training")

    parser.add_argument("--patience", type=int, default=10,
                        help="Patience for early stopping")

    parser.add_argument("--batch", type=float, default=0.5,
                        help="Batch size as a fraction of GPU memory")

    parser.add_argument("--save_period", type=int, default=10,
                        help="Save checkpoint every x epochs")

    parser.add_argument("--plots", action="store_true",
                        help="Generate plots")

    parser.add_argument("--single_cls", action="store_true",
                        help="Train as single-class dataset")

    parser.add_argument("--weighted", action="store_true",
                        help="Use weighted sampling for training")

    args = parser.parse_args()

    try:
        # Define and train the model
        trainer = ModelTrainer(
            training_data=args.training_data,
            epochs=args.epochs,
            weights=args.weights,
            model_path=args.model_path,
            root_dir=args.root_dir,
            run_dir=args.run_dir,
            task=args.task,
            cache=args.cache,
            device=args.device,
            half=args.half,
            img_size=args.img_size,
            patience=args.patience,
            batch=args.batch,
            save_period=args.save_period,
            plots=args.plots,
            single_cls=args.single_cls,
            weighted=args.weighted
        )
        trainer.train_model()
        print("Done.")
        
    except Exception as e:
        print(f"ERROR: Could not finish process.\n{e}")
        print(traceback.format_exc())


if __name__ == "__main__":
    main()

