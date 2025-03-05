from setuptools import setup, find_packages

setup(
    name='tator_tools',
    version='0.0.1',
    packages=find_packages(),
    install_requires=[
        'wheel',
        'dill',
        'tqdm',
        'numpy',
        'opencv-python',
        'opencv-contrib-python',
        'opencv-python-headless',
        'tator',
        'ultralytics==8.3.0',
        'supervision==0.25.0',
        'gradio==5.17.0',
        'fiftyone',
        'umap-learn>=0.5',
        'yolo-tiling>=0.0.11',
        'ipykernel',
        'ipywidgets'
    ]
)
