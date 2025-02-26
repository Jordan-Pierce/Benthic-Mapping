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

# For using scripts/
conda install ffmpeg
```

### App

The `app.py` is a python script with a `gradio` interface that will allow you to test the functionality of the Great 
Lakes Rock Detector, and the MDBC Coral Automatic Recognition and Locator (CARL) model.

You'll be presented with a local URL that you can paste into your browser to access the app.

### Parameters

- `Token`: [TATOR API Token](https://www.tator.io/docs/developer-guide/getting-started/get-an-api-token); This is required for authentication.
- `Remember Token`: Check this box to save your token for future use; Stored as user variable on local machine.
- `Project ID`: Enter the ID of the Tator project you're working on; defaults to `155`.
- `Frame Ranges`: Use commas to separate ranges, dashes for inclusive ranges, and single numbers for individual frames: 25-30, 45, 50
- `Media ID`: Enter the ID of the media file you want to process.
- `Confidence Threshold`: Set the confidence threshold for object detection; Higher values mean stricter detection.
- `IoU Threshold`: Set the Intersection over Union threshold for object detection; Higher values mean less overlap allowed between detections.
- `SAHI Mode`: Use SAHI to tile the image, make predictions on each, and then merge the results.
- `Model Type`: Select the architecture of the model corresponding to the model weights; either YOLO or RTDETR
- `Model Weights`: Upload the file containing the trained model weights.
- `Output`: The results and any messages from the algorithm will be displayed here.

### Model Weights

Download the latest version of the weights (.pt) for each of the algorithms before running the app. The weights can be
found after launching the app; please download and upload the weights to the app before running the algorithm.

</details>