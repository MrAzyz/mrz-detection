import os
from datetime import datetime

import cv2
from ultralytics import YOLO

from service.ocr_service import perform_ocr

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

model = YOLO('Model/yolov8_custom.pt')

def process_image(image_path):
    with open("Model/classes/classes.txt", "r") as my_file:
        class_list = my_file.read().strip().split("\n")

    frame = cv2.imread(image_path)
    frame = cv2.resize(frame, (1020, 500))

    # Run YOLO model prediction on the frame
    results = model.predict(frame)

    extracted_texts = []
    margin = 40

    for i, (box, cls, conf) in enumerate(zip(results[0].boxes.xyxy.tolist(),
                                             results[0].boxes.cls.tolist(),
                                             results[0].boxes.conf.tolist())):
        x1, y1, x2, y2 = map(int, box)

        x1_margin = max(0, x1 - margin)
        y1_margin = max(0, y1 - margin)
        x2_margin = min(frame.shape[1], x2 + margin)
        y2_margin = min(frame.shape[0], y2 + margin)

        # Crop the region of interest from the frame
        object_image = frame[y1_margin:y2_margin, x1_margin:x2_margin]

        # Save the cropped image
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        crop_path = f"cropped mrz/crop_{timestamp}_{i}.jpg"
        cv2.imwrite(crop_path, object_image)

        # Perform OCR on the cropped image
        texts = perform_ocr(crop_path)
        extracted_texts.extend(texts)

    return extracted_texts


print(process_image('passport.jpg'))