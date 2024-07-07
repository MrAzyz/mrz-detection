from paddleocr import PaddleOCR
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
ocr = PaddleOCR(use_angle_cls=True, lang='fr', use_gpu=False)


def perform_ocr(image_path):
    result = ocr.ocr(image_path, cls=True)
    texts = [item[1][0] for line in result for item in line]
    return texts
