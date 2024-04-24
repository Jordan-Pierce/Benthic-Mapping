import sys
import shutil
import platform
import subprocess

# ----------------------------------------------
# OS
# ----------------------------------------------
osused = platform.system()

if osused not in ['Windows', 'Linux']:
    raise Exception("This install script is only for Windows or Linux")

# ----------------------------------------------
# Conda
# ----------------------------------------------
# Need conda to install NVCC if it isn't already
console_output = subprocess.getstatusoutput('conda --version')

# Returned 1; conda not installed
if console_output[0]:
    raise Exception("This install script is only for machines with Conda already installed")

conda_exe = shutil.which('conda')

# ----------------------------------------------
# Python version
# ----------------------------------------------
python_v = f"{sys.version_info[0]}{sys.version_info[1]}"
python_sub_v = int(sys.version_info[1])

# check python version
if python_sub_v != 8:
    raise Exception(f"Only Python 3.{python_sub_v} is supported.")

# ---------------------------------------------
# MSVC for Windows
# ---------------------------------------------
if osused == 'Windows':

    try:
        print(f"NOTE: Installing msvc-runtime")
        subprocess.check_call([sys.executable, "-m", "pip", "install", 'msvc-runtime'])

    except Exception as e:
        print(f"There was an issue installing msvc-runtime\n{e}")
        sys.exit(1)

# ----------------------------------------------
# CUDA Toolkit version
# ----------------------------------------------
try:
    # Command for installing cuda nvcc
    conda_command = [conda_exe, "install", "-c", f"nvidia/label/cuda-11.8.0", "cuda-nvcc", "-y"]

    # Run the conda command
    print("NOTE: Installing CUDA NVCC 11.8")
    subprocess.run(conda_command, check=True)

    # Command for installing cuda nvcc
    conda_command = [conda_exe, "install", "-c", f"nvidia/label/cuda-11.8.0", "cuda-toolkit", "-y"]

    # Run the conda command
    print("NOTE: Installing CUDA Toolkit 11.8")
    subprocess.run(conda_command, check=True)

except Exception as e:
    print("ERROR: Could not install CUDA Toolkit")
    sys.exit(1)

# ----------------------------------------------
# Pytorch
# ----------------------------------------------
try:

    torch_package = 'torch==2.0.0+cu118'
    torchvision_package = 'torchvision==0.15.1+cu118'
    torch_extra_argument1 = '--extra-index-url'
    torch_extra_argument2 = 'https://download.pytorch.org/whl/cu118'

    # Setting Torch, Torchvision versions
    list_args = [sys.executable, "-m", "pip", "install", torch_package, torchvision_package]
    if torch_extra_argument1 != "":
        list_args.extend([torch_extra_argument1, torch_extra_argument2])

    # Installing Torch, Torchvision
    print("NOTE: Installing Torch 2.0.0")
    subprocess.check_call(list_args)

except Exception as e:
    print("ERROR: Could not install Pytorch")
    sys.exit(1)

# ----------------------------------------------
# Other dependencies
# ----------------------------------------------
install_requires = [
    'wheel',

    'numpy',
    'pandas',
    'scipy',
    'scikit_learn',
    'matplotlib',
    'Pillow',
    'opencv_python',
    "opencv-contrib-python",
    'scikit_image',

    'tator',

    # Don't mess with these
    'ultralytics',
    'sahi',
    'supervision',
    'autodistill',
    'autodistill-grounded-sam',
    'autodistill_grounding_dino',
    'autodistill-yolov8',
]

# Installing all the other packages
for package in install_requires:

    try:
        print(f"NOTE: Installing {package}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

    except Exception as e:
        print(f"There was an issue installing {package}\n{e}\n")
        print(f"If you're not already, please try using a conda environment with python 3.8")
        sys.exit(1)

print("Done.")