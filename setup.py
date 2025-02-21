from setuptools import setup, find_packages

setup(
    name='benthic-mapping',
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
        'sahi',
        'gradio',
    ],
    entry_points={
        'console_scripts': [
            'benthic-mapping=app:launch_gui',
        ],
    },
)
