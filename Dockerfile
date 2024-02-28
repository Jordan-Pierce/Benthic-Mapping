FROM nvidia/cuda:11.8.0-base-ubuntu18.04

# OS dependencies
RUN apt update && \
    apt install -y \
        python3.8 \
        python3-pip \
        wget \
        libglu1-mesa \
        libgl1-mesa-glx \
        libcurl4 \
        ffmpeg \
        libsm6 \
        libxext6 -y \
    && rm -rf /var/lib/apt/lists/*

# set the workdir
WORKDIR /workdir

# Copy over the requirements
COPY requirements.txt /workdir/

# Install dependencies
RUN python3.8 -m pip install -U pip
RUN python3.8 -m pip install -r requirements.txt

# Store license key as environmental variable
ENV PROJECT_ID=PROJECT_ID
ENV MEDIA_IDS=MEDIA_IDS
ENV HOST=HOST
ENV TOKEN=TOKEN

# Copy over the script and model
COPY Algorithm/ /workdir/
# Create a data directory
RUN mkdir -p /workdir/Data

# Specify the default command to run when the container starts
CMD ["python3.8", "/workdir/infer.py"]