# App

The `app.py` is a python script with a `gradio` interface that will allow you to test the functionality of the Great 
Lakes Rock Detector, and the MDBC Coral Automatic Recognition and Locator (CARL) model.

## Installation

To install, ensure that [miniconda](https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe) is 
installed on your machine if it is not already (this can be done by the user). After it is installed, open a `Anaconda
terminal`, navigate to a directory of your choosing, and perform the following commands:
```bash
# cmd

git clone https://github.com/Jordan-Pierce/Benthic-Mapping.git

cd Benthic-Mapping/Algorithms
 
conda create --name benthic-mapping python=3.8 -y
conda activate benthic-mapping

pip install -e .
```

*Optionally*, if you have a `NVIDIA GPU`, you can install the GPU version of PyTorch by running the following command:
```bash
# cmd

conda install cuda-nvcc -c nvidia/label/cuda-11.8.0 -y
conda install cudatoolkit=11.8 -c nvidia/label/cuda-11.8.0 -y
pip install torch==2.0.0+cu118 torchvision==0.15.1+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
```
Once this has finished installing, you can run the `app.py` script by performing the following command:
```bash
# cmd

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
