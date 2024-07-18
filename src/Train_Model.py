import os
import datetime
import argparse
import traceback

from ultralytics import YOLO
from ultralytics import RTDETR


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------

class ModelTrainer:
    def __init__(self, args):
        """
        Initialize the ModelTrainer with command line arguments.

        :param args: Parsed command line arguments
        """
        self.args = args

        self.root = None
        self.run_dir = None

        self.training_data = args.training_data
        self.num_epochs = args.epochs
        self.weights = args.weights

        # Updates root and run
        self.set_root_directory(args.root_dir)
        self.set_run_directory(args.run_dir)

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


# ----------------------------------------------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------------------------------------------

def main():
    """

    :return:
    """
    parser = argparse.ArgumentParser(description="Train a model for object detection.")

    parser.add_argument("--training_data", type=str, required=True,
                        help="Path to the training data YAML file")

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

    parser.add_argument("--run_prefix", type=str, default=None,
                        help="Prefix for the run name")

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

    parser.add_argument("--batch", type=float, default=0.8,
                        help="Batch size as a fraction of GPU memory")

    parser.add_argument("--save_period", type=int, default=10,
                        help="Save checkpoint every x epochs")

    parser.add_argument("--plots", action="store_true",
                        help="Generate plots")

    parser.add_argument("--single_cls", action="store_true",
                        help="Train as single-class dataset")

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

