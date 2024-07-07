# MRZ Detection Project

This project focuses on detecting and extracting text from the Machine-Readable Zone (MRZ) of documents such as passports using a custom-trained YOLOv8 model and Optical Character Recognition (OCR).

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/mrz-detection.git
cd mrz-detection
```

2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

3. Ensure you have the YOLOv8 model and the classes file in the Model directory.

## Usage

1. Place the image you want to process in the project directory.
2. Run the following script:

```bash
python main.py --image_path path/to/your/image.jpg
```

## Example

Here's an example of how to use the process_image function directly in a Python script:

```python
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

        object_image = frame[y1_margin:y2_margin, x1_margin:x2_margin]

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        crop_path = f"cropped_mrz/crop_{timestamp}_{i}.jpg"
        cv2.imwrite(crop_path, object_image)

        texts = perform_ocr(crop_path)
        extracted_texts.extend(texts)

    return extracted_texts

print(process_image('passport.jpg'))
```
