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

from Common import create_training_yaml


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


class ModelTrainer:
    def __init__(self, args):
        """
        Initialize the ModelTrainer with command line arguments.

        :param args: Parsed command line arguments
        """
        self.args = args

        self.root = None
        self.run_dir = None

        self.num_epochs = args.epochs
        self.weights = args.weights

        # Updates root and run
        self.set_root_directory(args.root_dir)
        self.set_run_directory(args.run_dir)
        self.create_training_data(args.training_data)

        self.run_name = self.get_run_name()
        self.target_model = self.load_model()

    def set_root_directory(self, root_dir):
        """
        Get and validate the root directory.

        :param root_dir: Root directory path
        :return: Validated root directory path
        """
        root = root_dir or os.path.dirname(os.path.dirname(os.path.realpath(__file__))) + "/Data"
        root = root.replace("\\", "/")
        assert os.path.exists(root), f"Root directory not found: {root}"
        self.root = root

    def set_run_directory(self, run_dir):
        """
        Create the run directory if it doesn't exist.

        :param run_dir: Run directory path
        :return: Run directory path
        """
        run_dir = run_dir or f"{self.root}/Runs"
        os.makedirs(run_dir, exist_ok=True)
        self.run_dir = run_dir

    def get_run_name(self):
        """
        Generate a run name with an optional prefix.

        :return: Generated run name
        """
        return f"{self.get_now()}_{self.args.task}_{self.weights.split('.')[0]}"

    def create_training_data(self, training_data):
        """

        :return:
        """
        # Check if training data is already a YAML file
        if training_data.endswith(".yaml"):
            self.training_data = training_data
            return
        if os.path.isdir(training_data):
            yaml_files = glob.glob(f"{training_data}/**/data.yaml")
            if not yaml_files:
                raise FileNotFoundError(f"No 'data.yaml' files found in '{training_data}'")

            self.training_data = create_training_yaml(yaml_files, self.run_dir)

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

            if os.path.exists(self.args.model_path):
                model.load_weights(self.args.model_path)

            return model

        except Exception as e:
            raise Exception(f"ERROR: Failed to load model.\n{e}")

    @staticmethod
    def get_now():
        """
        Get current datetime as a formatted string.

        :return: Formatted current datetime
        """
        return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    def train_model(self):
        """
        Train the model with the specified parameters.

        :return: Training results
        """
        if self.args.weighted:
            build.YOLODataset = YOLOWeightedDataset

        results = self.target_model.train(
            data=self.training_data,
            task=self.args.task,
            cache=self.args.cache,
            device=self.args.device,
            half=self.args.half,
            epochs=self.num_epochs,
            patience=self.args.patience,
            batch=self.args.batch,
            project=self.run_dir,
            name=self.run_name,
            save_period=self.args.save_period,
            plots=self.args.plots,
            single_cls=self.args.single_cls
        )
        print("Training completed.")
        return results

    def evaluate_model(self):
        try:

            results = self.target_model.val(
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
        trainer = ModelTrainer(args)
        results = trainer.train_model()
        print("Done.")
    except Exception as e:
        print(f"ERROR: Could not finish process.\n{e}")
        print(traceback.format_exc())


if __name__ == "__main__":
    main()

