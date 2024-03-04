FROM nvidia/cuda:11.8.0-base-ubuntu22.04

# OS dependencies
RUN apt update && \
    apt upgrade -y && \
    apt install -y \
        python3 \
        python3-pip \
        wget \
        libglu1-mesa \
        libgl1-mesa-glx \
        libcurl4 \
        libsm6 \
        libxext6 \
        libglib2.0-0\
    && rm -rf /var/lib/apt/lists/*

# Set the workdir
WORKDIR /workdir

# Copy over the requirements
COPY requirements.txt /workdir/

# Install dependencies
RUN python3 -m pip install -U pip
RUN python3 -m pip install -r requirements.txt

# Copy over the script and model
COPY Algorithm/ /workdir/

# Create a data directory
RUN mkdir -p /workdir/Data

# Specify the default command to run when the container starts
CMD ["python3", "/workdir/infer.py"]