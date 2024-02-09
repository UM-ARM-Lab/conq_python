"""
Ways to download last semester's dataset

Terminal: (Add your roboflow key in the URL)
- curl -L "https://universe.roboflow.com/ds/fKKjj7WssA?key=##########" &gt; roboflow.zip; unzip roboflow.zip; rm roboflow.zip

Direct Download: (Add your roboflow key in the URL)
- link: https://universe.roboflow.com/ds/fKKjj7WssA?key=##########

Python:
- !pip install roboflow
- Run code below
"""

from roboflow import Roboflow

# Config stuff
rf = Roboflow(api_key="***ADD API KEY HERE***")

project = rf.workspace("agrobots-9sm1u").project("garden-implements")
dataset = project.version(5).download("yolov8")
