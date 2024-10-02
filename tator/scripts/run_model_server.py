import os
import logging
import redis
import json
import numpy as np
import sys
import base64
import yaml
from pathlib import Path

current_dir = Path.cwd()

algorithm_path = current_dir.joinpath("Algorithms")
algorithm_path = algorithm_path.joinpath("Rocks")
sys.path.insert(1, str(algorithm_path))

from rock_algorithm import RockAlgorithm

logging.basicConfig(
    handlers=[logging.StreamHandler()],
    format="%(asctime)s %(levelname)s:%(message)s",
    datefmt="%m/%d/%Y %I:%M:%S %p",
    level=logging.INFO,
)

logger = logging.getLogger(__name__)

db = redis.StrictRedis(
    host=os.getenv("REDIS_HOST"),
    port=os.getenv("REDIS_PORT"),
    db=os.getenv("REDIS_DB"))

def base64_decode_image(a, dtype, shape):
    a = bytes(a, encoding="utf-8")
    a = np.frombuffer(base64.decodestring(a), dtype=dtype)
    a = a.reshape(shape)
    return a

class DictToDotNotation:
    '''Useful class for getting dot notation access to dict'''
    def __init__(self, d):
        for k, v in d.items():
            setattr(self, k, v)

def process_input_info(input_info):
    """
    Gather relevant input information for processing

    :return:
    [0] str: ID associated with input message
    [1] np.ndarray: Image to process
    """
    input_info = json.loads(input_info.decode("utf-8"))
    img_width = input_info["width"]
    img_height = input_info["height"]
    img0 = base64_decode_image(input_info["image"],
        np.uint8,
        (1, img_height, img_width, 3))
    img0 = np.squeeze(img0)
    img = np.ascontiguousarray(img0)

    id = q["id"]

    return id, img

def get_input_info_blocking():
    """
    Get info of an image to process. Block until available.
    """
    q = db.blpop(os.getenv("IMAGE_QUEUE_ROCK_MASK"))
    return q[1]

def main():
    """ Main application thread

    Only a keyboard interrupt will stop the thread.

    """

    with open(os.getenv("ROCK_MASK_ALGO_CONFIG"), "r") as file_handle:
        algo_config = yaml.safe_load(file_handle)

    algo = RockAlgorithm(config=algo_config)
    algo.initialize()

    logger.info("Rock algorithm initialized, ready for input.")

    while True:
        try:
            input_info = get_input_info_blocking()
            id, image = process_input_info(input_info)
            points = algo.infer(image)
            db.set(id, json.dumps(points))

        except KeyboardInterrupt:
            logger.info("Stopped")

if __name__ == "__main__":
    main()