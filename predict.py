from ultralytics import YOLO
import os
import matplotlib.pyplot as plt

model = YOLO("yolov8n.pt")  # hoặc model train xong nếu muốn test weights

folder_path = "satpic"  # folder chứa 5 ảnh
for filename in os.listdir(folder_path):
    if filename.endswith(".png") or filename.endswith(".jpg"):
        img_path = os.path.join(folder_path, filename)
        results = model.predict(img_path, save=True)  # lưu ảnh có bbox
        # Hiển thị luôn ảnh với bbox
        img_bbox = results[0].plot()
        plt.imshow(img_bbox)
        plt.axis("on")
        plt.show()
