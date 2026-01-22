from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("yolo11n.pt")  # load a pretrained YOLOv8n model

    # Train the model
    model.train(data="MUObj.yaml", 
                epochs=10, 
                imgsz=320, 
                batch=-1,
                cache="ram",
                workers=1,
               )