import cv2
import onnxruntime as ort
import numpy as np

from cfg import YoloCfg


class YoloDetect:

    def __init__(self, cfg=None):
        self.cfg = cfg or YoloCfg()
        self.session = ort.InferenceSession(
            self.cfg.ONNX_MODEL,
            providers=['CPUExecutionProvider']
        )
        self.input_name = self.session.get_inputs()[0].name

    def detect(self, image: np.ndarray):
        ori_img = image.copy()
        img = cv2.resize(ori_img, (self.cfg.INPUT_SIZE, self.cfg.INPUT_SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)[np.newaxis]      # (1, 3, 1280, 1280)

        outputs = self.session.run(None, {self.input_name: img})[0][0]  # (300, 6)
        results_after_nms = self._nms(outputs)

        results_in_list = [
            x for x in results_after_nms
            if x[-2] > self.cfg.YOLO_CONF and int(x[-1]) in self.cfg.CLS
        ]

        return self.to_dict(results_in_list)

    def release(self):
        pass

    def to_dict(self, boxes: list):
        dict_results = {}
        for bbox in boxes:
            cls = int(bbox[-1])
            if cls not in dict_results:
                dict_results[cls] = []
            dict_results[cls].append(bbox[:-1])   # [x1, y1, x2, y2, conf]
        return dict_results

    def _nms(self, boxes: np.ndarray):
        if len(boxes) == 0:
            return []

        classes = {}
        for box in boxes:
            cls = int(box[5])
            if cls not in classes:
                classes[cls] = []
            classes[cls].append(box)

        result = []

        for cls, cls_boxes in classes.items():
            boxes_np = np.array(cls_boxes, dtype=np.float32)

            x1, y1 = boxes_np[:, 0], boxes_np[:, 1]
            x2, y2 = boxes_np[:, 2], boxes_np[:, 3]
            scores  = boxes_np[:, 4]
            areas   = (x2 - x1) * (y2 - y1)
            order   = np.argsort(scores)[::-1]

            keep = []
            while len(order) > 0:
                i = order[0]
                keep.append(i)
                if len(order) == 1:
                    break

                xx1 = np.maximum(x1[i], x1[order[1:]])
                yy1 = np.maximum(y1[i], y1[order[1:]])
                xx2 = np.minimum(x2[i], x2[order[1:]])
                yy2 = np.minimum(y2[i], y2[order[1:]])

                inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
                iou   = inter / (areas[i] + areas[order[1:]] - inter)

                order = order[1:][iou <= self.cfg.YOLO_IOU]

            result.extend(boxes_np[keep].tolist())

        return result