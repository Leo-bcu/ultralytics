from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO(r"yolo11n.pt")  # load a pretrained YOLOv8n model

    # Train the model
    model.train(data="african-wildlife.yaml", 
                epochs=10, 
                imgsz=640, 
                batch=2,
                cache=False,
                workers=0,
               )