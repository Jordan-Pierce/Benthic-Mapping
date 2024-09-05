# App

The `app.py` is a python script with a `gradio` interface that will allow you to test the functionality of the Great 
Lakes Rock Detector, and the MDBC Coral Automatic Recognition and Locator (CARL) model.

### **How to Install**

##### GitHub Repository
First, clone the repo:
```bash
# cmd

# Clone and enter the repository
git clone https://github.com/Jordan-Pierce/Benthic-Mapping.git
cd Benthic-Mapping/Algorithms
```
##### Anaconda
Then set up an `Anaconda` environment:
```bash
# cmd

# Create and activate an environment
conda create --name benthic-mapping python=3.8 -y
conda activate benthic-mapping
```
##### CUDA
Once this has finished, if you have CUDA, you can install the versions of `cuda-nvcc` and `cudatoolkit` that you need,
and then install the corresponding versions `torch` and `torchvision`:
```bash
# cmd

# Example for CUDA 11.8
conda install cuda-nvcc -c nvidia/label/cuda-11.8.0 -y
conda install cudatoolkit=11.8 -c nvidia/label/cuda-11.8.0 -y

# Example for torch 2.0.0 and torchvision 0.15.1 w/ CUDA 11.8
pip install torch==2.0.0+cu118 torchvision==0.15.1+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
```
See here for more details on [PyTorch](https://pytorch.org/get-started/locally/) versions.
##### Install
Finally, run the setup script to install the application:
```bash
# cmd

pip install -e .

benthic-mapping
```

You'll be presented with a local URL that you can paste into your browser to access the app.

## Parameters

- `Token`: [TATOR API Token](https://www.tator.io/docs/developer-guide/getting-started/get-an-api-token); This is required for authentication.
- `Remember Token`: Check this box to save your token for future use; Stored as user variable on local machine.
- `Project ID`: Enter the ID of the Tator project you're working on; defaults to `155`.
- `Frame Ranges`: Use commas to separate ranges, dashes for inclusive ranges, and single numbers for individual frames: 25-30, 45, 50
- `Media ID`: Enter the ID of the media file you want to process.
- `Confidence Threshold`: Set the confidence threshold for object detection; Higher values mean stricter detection.
- `IoU Threshold`: Set the Intersection over Union threshold for object detection; Higher values mean less overlap allowed between detections.
- `SMOL Mode`: Use SAHI to tile the image, make predictions on each, and then merge the results.
- `Model Type`: Select the architecture of the model corresponding to the model weights; either YOLO or RTDETR
- `Model Weights`: Upload the file containing the trained model weights.
- `Output`: The results and any messages from the algorithm will be displayed here.


### Model Weights

Download the latest version of the weights (.pt) for each of the algorithms before running the app. The weights can be
found after launching the app; please download and upload the weights to the app before running the algorithm.