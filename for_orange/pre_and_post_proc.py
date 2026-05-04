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
    def rotate_and_crop_image(image: np.ndarray, path_for_rotated: Path):

        img = image.copy()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        _, thresh = cv2.threshold(img, 30, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 30))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            cv2.imwrite(str(path_for_rotated), img)
            return img

        largest = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(largest)

        angle = rect[2]
        angle = min([angle, 90 + angle], key=lambda x: abs(x))

        print(f"Угол поворота: {angle:.4f} Deg")

        h, w = img.shape
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle=angle, scale=1.0)
        rotated = cv2.warpAffine(img, M, (w, h))

        cv2.imwrite(str(path_for_rotated), rotated)

        _, thresh_rot = cv2.threshold(rotated, 30, 255, cv2.THRESH_BINARY)
        thresh_rot = cv2.morphologyEx(thresh_rot, cv2.MORPH_CLOSE, kernel)
        thresh_rot = cv2.morphologyEx(thresh_rot, cv2.MORPH_OPEN, kernel)
        contours_rot, _ = cv2.findContours(thresh_rot, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        largest_rot = max(contours_rot, key=cv2.contourArea)
        x, y, cw, ch = cv2.boundingRect(largest_rot)

        rx1 = int(x)
        ry1 = int(y)
        rx2 = int(x + cw)
        ry2 = int(y + ch)

        return rotated, (rx1, ry1, rx2, ry2)
    
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
    def make_cls_image(results: list, img: np.ndarray, image_path: Path):
        img_with_preds = cv2.cvtColor(img.copy(), cv2.COLOR_GRAY2BGR)
        h, w = img.shape[:2]
        sx = w / 1280 

        for res in results:
            lane_id     = res["lane_id"]
            classical_y = res["classical_y"] 

            cx = int((lane_id[0] + lane_id[2]) / 2 * sx)
            cy = int(classical_y) 

            cv2.circle(img_with_preds, (cx, cy), radius=3, color=(0, 0, 255), thickness=-1)

        cv2.imwrite(image_path, img_with_preds)


    @staticmethod
    def show_img(img_path: Path):

        subprocess.Popen(['feh', str(img_path)])


class PostProc:

    @staticmethod
    def check_in_gel(image: np.ndarray, res: dict, cords: tuple):

        new_lines = []

        h, w = image.shape[:2]
        sx, sy = w / 1280, h / 1280

        for lane in res[2]:

            lx1 = lane[0] * sx
            ly1 = lane[1] * sy
            lx2 = lane[2] * sx
            ly2 = lane[3] * sy

            x_left   = max(lx1, cords[0])
            y_top    = max(ly1, cords[1])
            x_right  = min(lx2, cords[2])
            y_bottom = min(ly2, cords[3])

            if x_right <= x_left or y_bottom <= y_top:
                continue

            s_intersection = (x_right - x_left) * (y_bottom - y_top)
            s_lane = (lx2 - lx1) * (ly2 - ly1)

            if s_lane > 0 and s_intersection >= s_lane * 0.7:
                new_lines.append(lane)
    
        res[2] = new_lines
        return res
    
    @staticmethod
    def nms(boxes, iou_threshold=0.5):

        if not boxes:
            return []
        
        boxes = np.array(boxes)
        
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        scores = boxes[:, 4]
        
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        
        order = scores.argsort()[::-1]
        
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            iou = inter / (areas[i] + areas[order[1:]] - inter)
            
            inds = np.where(iou <= iou_threshold)[0]
            order = order[inds + 1]
        
        return [boxes[i].tolist() for i in keep]
    
    @staticmethod
    def add_new_zeros_and_lines(image: np.ndarray, res: dict, cords: tuple):      
        # and blyat crop them

        h, w = image.shape[:2]
        sx, sy = w / 1280, h / 1280

        new_lines = []
        new_boxes = []

        y_max = 0

        prom, size_x, size_y = [], [], []

        for zero in res[0]:
            if y_max < zero[3]:
                y_max = zero[3]
            size_x.append(zero[2] - zero[0])
            size_y.append(zero[3] - zero[1])
            prom.append((zero[3] + zero[1]) // 2)

            new_boxes.append(zero)

        zero_mean = (sum(prom) // len(prom))
        zero_size_x = (sum(size_x) // len(size_x)) // 2
        zero_size_y = (sum(size_y) // len(size_y)) // 2

        for line in res[2]:
            x = (line[2] + line[0]) // 2  
            x1 = (x - zero_size_x) 
            x2 = (x + zero_size_x) 
            y1 = (zero_mean - zero_size_y) 
            y2 = (zero_mean + zero_size_y)
            new_boxes.append([x1, y1, x2, y2, 0.01])
            new_lines.append([line[0], y_max, line[2], line[3], line[4]])

        filtered_boxes = PostProc.nms(new_boxes, iou_threshold=0.01)

        res[0] = filtered_boxes
        res[2] = new_lines

        return res
    
    @staticmethod
    def make_final_image(result_yolo: dict, result_cls: dict, img: np.ndarray, image_path: Path):

        class_colors = {
            0: (255, 0, 0),    # Синий - zero
            1: (0, 0, 255),    # Красный - strip
            2: (0, 255, 0)     # Зеленый - line
        }

        img_with_preds = cv2.cvtColor(img.copy(), cv2.COLOR_GRAY2BGR)
        h, w = img.shape[:2]
        sx, sy = w / 1280, h / 1280

        for cls, bboxes in result_yolo.items():
            if cls in [0, 2]:
                for (x1, y1, x2, y2, conf) in bboxes:
                    x1, y1 = int(x1 * sx), int(y1 * sy) 
                    x2, y2 = int(x2 * sx), int(y2 * sy)
                    cv2.rectangle(img_with_preds, (x1, y1), (x2, y2), class_colors[cls], 1)

        for res in result_cls:
            lane_id     = res["lane_id"]
            classical_y = res["classical_y"] 

            cx = int((lane_id[0] + lane_id[2]) / 2 * sx)
            cy = int(classical_y) 

            cv2.circle(img_with_preds, (cx, cy), radius=3, color=class_colors[1], thickness=-1)

        cv2.imwrite(image_path, img_with_preds)







        
    