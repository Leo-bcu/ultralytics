from ultralytics import YOLO

model = YOLO(r"yolo11n.pt")  # load a pretrained YOLOv8n model

model.predict(source=r"/Users/leo/Desktop/deeplearing/ultralytics/ultralytics/assets",
              show=False,
              save=True,
              )  # predict on an image URL and display results