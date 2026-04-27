import numpy as np
import cv2 
from pathlib import Path
import json
import subprocess
from collections import defaultdict

class PreProc:

    @staticmethod
    def load_img(image_path: Path):

        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Не удалось загрузить: {image_path}")
        return image

    @staticmethod
    def rotate_image(image: np.ndarray, path_for_rotated: Path):

        img = image.copy()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        _, thresh = cv2.threshold(img, 10, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            cv2.imwrite(path_for_rotated, img)
            return img

        largest = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(largest)

        angle = rect[2]
        angle = min([angle, 90 + angle], key=lambda x: abs(x))

        h, w = img.shape
        M = cv2.getRotationMatrix2D(rect[0], angle=angle, scale=1.0)
        rotated = cv2.warpAffine(img, M, (w, h))

        cv2.imwrite(path_for_rotated, rotated)
        print(f"Угол поворота для {path_for_rotated.name}: {angle:.4} Deg")

        return rotated
    
    @staticmethod
    def to_json(results: dict, json_dir: Path):
        
        with open(json_dir, 'w', encoding='UTF-8') as file:
            json.dump(results, file, indent=2)

    @staticmethod
    def make_image(results: dict, img: np.ndarray, image_path: Path):

        class_colors = {
            0: (255, 0, 0),    # Синий - zero
            1: (0, 255, 0),    # Зеленый - strip
            2: (0, 0, 255)     # Красный - line
        }

        img_with_preds = cv2.cvtColor(img.copy(), cv2.COLOR_GRAY2BGR)
        h, w = img.shape[:2]
        sx, sy = w / 1280, h / 1280

        for cls, bboxes in results.items():
            for (x1, y1, x2, y2, conf) in bboxes:
                x1, y1 = int(x1 * sx), int(y1 * sy) 
                x2, y2 = int(x2 * sx), int(y2 * sy)
                cv2.rectangle(img_with_preds, (x1, y1), (x2, y2), class_colors[cls], 1)

        cv2.imwrite(image_path, img_with_preds)

    @staticmethod
    def show_img(img_path: Path):

        subprocess.Popen(['feh', str(img_path)])