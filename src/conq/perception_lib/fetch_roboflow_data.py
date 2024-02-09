"""
Ways to download last semester's dataset

Terminal:
- curl -L "https://universe.roboflow.com/ds/fKKjj7WssA?key=yzs9c7LgqW" &gt; roboflow.zip; unzip roboflow.zip; rm roboflow.zip

Direct Download:
- link: https://universe.roboflow.com/ds/fKKjj7WssA?key=yzs9c7LgqW

Python:
- !pip install roboflow
- Run code below
"""

from roboflow import Roboflow

# Config stuff
rf = Roboflow(api_key="22iMfgZpDlP0VFyWIc5t")

project = rf.workspace("agrobots-9sm1u").project("garden-implements")
dataset = project.version(5).download("yolov8")