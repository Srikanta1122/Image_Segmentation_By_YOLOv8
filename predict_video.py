from ultralytics import YOLO
import shutil

bestModelPath='D:\Image_Segmentation_By_YOLOV8\\best.pt'
bestModel=YOLO(bestModelPath)
videoPath='D:\Image_Segmentation_By_YOLOV8\Pothole_Segmentation_YOLOv8_Dataset\sample_video.mp4'
bestModel.predict(source=videoPath, save=True)

