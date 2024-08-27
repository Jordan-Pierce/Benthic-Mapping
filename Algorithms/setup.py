from setuptools import setup, find_packages

setup(
    name='benthic-mapping',
    version='1.0.0',
    packages=find_packages(),
    install_requires=[
        'msvc-runtime',
        'wheel',
        'dill',
        'tqdm',
        'numpy',
        'opencv-python',
        'opencv-contrib-python',
        'opencv-python-headless',
        'tator',
        'ultralytics',
        'supervision',
        'sahi',
        'gradio',
    ],
    entry_points={
        'console_scripts': [
            'benthic-mapping=app:launch_gui',
        ],
    },
)