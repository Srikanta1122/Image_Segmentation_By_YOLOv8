from ultralytics import YOLO

import cv2


model_path = "D:\Image_Segmentation_By_YOLOV8\\best.pt"

image_path = "D:\Image_Segmentation_By_YOLOV8\Pothole_Segmentation_YOLOv8_Dataset\images\\train\pic-94-_jpg.rf.ba3925642e2ccb4869665efacd0c7649.jpg"

img = cv2.imread(image_path)
H, W, _ = img.shape

model = YOLO(model_path)

results = model(img)

for result in results:
    for j, mask in enumerate(result.masks.data):

        mask = mask.numpy() * 255

        mask = cv2.resize(mask, (W, H))

        cv2.imwrite('./output.png', mask)


