# Rock Algorithm
Run the rock algorithm inference script to create Rock Mask localization polygons for a given image or video frame. It's best to run this script in the docker container.

```
cd tator/scripts
python3 tator_rock_infer.py --host https://cloud.tator.io --token $CLOUD_TATOR_IO_TOKEN --media-id <> --frame <> --version-id <> --algorithm-config ../../Algorithms/Rocks/configs/rock_config_v1.1.0.yaml --tator-config ../../Algorithms/Rocks/configs/tator_config.yaml
```
