from ultralytics import YOLO

model = YOLO("yolov12n.pt")

model.train(data = "dataset.yaml", imgsz = 640, batch = 8, workers = 0, device = 0, epochs = 25,)