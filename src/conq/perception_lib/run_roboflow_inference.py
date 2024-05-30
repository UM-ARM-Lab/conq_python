"""Running inference directly via Roboflow"""

from roboflow import Roboflow

from conq.api_keys import ROBOFLOW_API_KEY

# Config stuff
rf = Roboflow(api_key=ROBOFLOW_API_KEY.get())


def run_roboflow_model_inference():
    project = rf.workspace().project("garden-implements")
    model = project.version(2).model

    print(model.predict("your_image.jpg").json())
    # print(model.predict("URL_OF_YOUR_IMAGE").json())

    model.predict("your_image.jpg").save("prediction.jpg")
