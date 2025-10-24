from ultralytics import YOLO

model = YOLO("yolov8n.pt")


print(model)
model.train(
    data="data.yaml",
    epochs=50,
    imgsz=640,
    batch=16,
    name="storm_train"
)