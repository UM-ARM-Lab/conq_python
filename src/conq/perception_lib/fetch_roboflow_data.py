"""
Ways to download last semester's dataset

Terminal:
- curl -L "***ADD URL HERE***" &gt; roboflow.zip; unzip roboflow.zip; rm roboflow.zip

Direct Download:
- link: ***ADD URL HERE***

Python:
- !pip install roboflow
- Run code below
"""

from roboflow import Roboflow

# TODO: Remove this API key; use bash src
rf = Roboflow(api_key="***ADD API KEY HERE***")

project = rf.workspace("agrobots-9sm1u").project("garden-implements")
dataset = project.version(5).download("yolov8")