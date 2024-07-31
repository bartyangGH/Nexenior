from roboflow import Roboflow

rf = Roboflow(api_key="F1zXAL3OyphOyshUSP6y")
project = rf.workspace().project("cocotext-mbfa4")
text_detection_model = project.version(22).model

# infer on a local image
results = model.predict("truck.jpg", confidence=40, overlap=30)
print(type(results))
for r in results:
    print(type(r))
# model.predict("truck.jpg", confidence=40, overlap=80).save("prediction.jpg")
