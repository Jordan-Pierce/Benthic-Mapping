# Model weights
Need to place the model weights in a top folder `Data`

# Rock Algorithm
Run the rock algorithm inference script to create Rock Mask localization polygons for a given image or video frame. It's best to run this script in the docker container.

```
cd tator/scripts
python3 tator_rock_infer.py --host https://cloud.tator.io --token $CLOUD_TATOR_IO_TOKEN --media-id <> --frame <> --version-id <> --algorithm-config ../../Algorithms/Rocks/configs/rock_config_v1.1.0.yaml --tator-config ../../Algorithms/Rocks/configs/tator_config.yaml
```

# Model Server
Expected to be run using the model server framework here: `https://gitlab.com/bgwoodward/keras-model-server-fast-api`

## Docker file
docker build -t rock-masks-model-server -f ./tator/docker/glbm_rocks_model_server.docker .
