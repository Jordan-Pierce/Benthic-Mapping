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

### benthic_mapper




</details>