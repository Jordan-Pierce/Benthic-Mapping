# App

The `app.py` is a python script with a `gradio` interface that will allow you to test the functionality of the Great 
Lakes Rock Detector.

## Installation

To install, ensure that [miniconda](https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe) is 
installed on your machine if it is not already (this can be done by the user). After it is installed, open a `Anaconda
terminal`, navigate to a directory of your choosing, and perform the following commands:
```bash
# cmd
git clone https://github.com/Jordan-Pierce/Benthic-Mapping.git

cd Benthic-Mapping
 
conda create --name benthic-mapping python=3.8 -y
conda activate benthic-mapping
python install.py
python Algorithms/app.py
```
In the terminal you should be provided with a local host address that if clicked on, will open your web-browser to the 
`gradio` interface.

## Parameters

- `Token`: [TATOR API Token](https://www.tator.io/docs/developer-guide/getting-started/get-an-api-token); This is required for authentication.
- `Remember Token`: Check this box to save your token for future use; Stored as user variable on local machine.
- `Project ID`: Enter the ID of the Tator project you're working on; defaults to `155`.
- `Frame Ranges`: Use commas to separate ranges, dashes for inclusive ranges, and single numbers for individual frames: 25-30, 45, 50
- `Media ID`: Enter the ID of the media file you want to process.
- `Confidence Threshold`: Set the confidence threshold for object detection; Higher values mean stricter detection.
- `IoU Threshold`: Set the Intersection over Union threshold for object detection; Higher values mean less overlap allowed between detections.
- `Model Type`: Select the architecture of the model corresponding to the model weights; either YOLO or RTDETR
- `Model Weights`: Upload the file containing the trained model weights;
- `Output`: The results and any messages from the algorithm will be displayed here.


### Model Weights

Download the latest version of the weights (.pth) by NOAA users:
- YOLO - [06/26/2024](https://drive.google.com/file/d/1vcsO9rQr0lScHuBLISBR72Xgr1kpYIec/view?usp=drive_link)
- RTDETR - [06/30/2024](https://drive.google.com/file/d/1qotY6xEF5Y3cOknseGROEqtpUa3AnVZ2/view?usp=drive_link)