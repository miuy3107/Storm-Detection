from ultralytics import YOLO
import matplotlib.pyplot as plt
import os

# Load model đã train xong 
model = YOLO("runs/detect/storm_train4/weights/best.pt")  

folder_path = "satpic"
for filename in os.listdir(folder_path):
    if filename.endswith(".png") or filename.endswith(".jpg"):
        img_path = os.path.join(folder_path, filename)
        results = model.predict(img_path, save=True)  # lưu kết quả
        img_bbox = results[0].plot()
        plt.imshow(img_bbox)
        plt.axis("off")
        plt.show()