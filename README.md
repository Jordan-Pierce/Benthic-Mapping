# Benthic-Mapping

A library for automating detection within benthic habitats (for finding rocks, coral, and other benthic features). This
library revolves around Tator.

<details>
<summary><h2>Tator Algorithms</h2></summary>

For production deployment in Tator

### Installation

```bash
# cmd

conda create --name bm python==3.10 -y
conda activate bm

pip install uv

uv pip install -r requirements.txt

conda install cuda-nvcc -c nvidia/label/cuda-11.8.0 -y
conda install cudatoolkit=11.8 -c nvidia/label/cuda-11.8.0 -y

# Example for torch 2.0.0 and torchvision 0.15.1 w/ CUDA 11.8
uv pip install torch==2.0.0+cu118 torchvision==0.15.1+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
```

</details>

<details>
<summary><h2>Local</h2></summary>

For local testing and debugging algorithms before deployment in Tator. Also useful for data visualization.

### Installation

```bash
# cmd

conda create --name bm python==3.10 -y
conda activate bm

pip install uv

uv pip install -e .

conda install cuda-nvcc -c nvidia/label/cuda-11.8.0 -y
conda install cudatoolkit=11.8 -c nvidia/label/cuda-11.8.0 -y

# Example for torch 2.0.0 and torchvision 0.15.1 w/ CUDA 11.8
uv pip install torch==2.0.0+cu118 torchvision==0.15.1+cu118 --extra-index-url https://download.pytorch.org/whl/cu118

conda install ffmpeg
```

### Scripts

- **Algorithm_Demo.py**: A Gradio-based web interface for running Rock and Coral detection algorithms on Tator media. Provides an interactive way to test models with configurable parameters.

- **Common.py**: Contains utility functions shared across multiple scripts, including dataset rendering, YAML configuration handling, and timestamp generation.

- **Crop_Bounding_Boxes.py**: Creates a classification dataset by cropping objects from detection datasets based on bounding boxes. Organizes extracted chips into train/val/test splits by class.

- **Download_Labeled_Data.py**: Downloads labeled data from Tator, converting annotations into YOLO format. Supports filtering by search string and random sampling.

- **Download_Media.py**: Batch downloads media files from Tator with options to convert videos to MP4 and extract frames at specified intervals.

- **Fiftyone_Clustering.py**: Utilizes FiftyOne and UMAP to create visual clusters of images for exploring datasets and identifying patterns.

- **Inference_Video.py**: Performs inference on videos using trained models (YOLO/RTDETR) with optional SAM segmentation, SAHI for small object detection, and tracking.

- **Train_Model.py**: Trains object detection models with customizable parameters including weighted datasets for class imbalance and various optimization options.
- 

</details>