FROM nvidia/cuda:11.8.0-base-ubuntu18.04

# OS dependencies
RUN apt update && \
    apt install -y \
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

# Copy everything over
COPY Algorithm/* /workdir/
COPY requirements.txt /workdir/

# Create a data directory
RUN mkdir -p /workdir/Data

# Install dependencies
RUN python3 -m pip install --user -r requirements.txt

# Store license key as environmental variable
ENV PROJECT_ID=PROJECT_ID
ENV MEDIA_IDS=MEDIA_IDS
ENV HOST=HOST
ENV TOKEN=TOKEN

# Specify the default command to run when the container starts
CMD ["python3", "/workdir/infer.py"]

