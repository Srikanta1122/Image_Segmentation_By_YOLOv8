from ultralytics import YOLO

model = YOLO('yolov8n-seg.pt')  # load a pretrained model (recommended for training)

model.train(data='config.yaml', epochs=10, imgsz=640)

# train the YOLOv8 model in google colab for perform the image segmentation task

