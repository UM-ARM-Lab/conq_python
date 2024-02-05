from roboflow import Roboflow

rf = Roboflow(api_key="22iMfgZpDlP0VFyWIc5t")
project = rf.workspace().project("garden-implements")
model = project.version(2).model

print(model.predict("your_image.jpg").json())
# print(model.predict("URL_OF_YOUR_IMAGE").json())

model.predict("your_image.jpg").save("prediction.jpg")