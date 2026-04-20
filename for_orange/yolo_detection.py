import cv2
from rknnlite.api import RKNNLite
import numpy as np

from cfg import YoloCfg


class YoloDetect:

    def __init__(self, cfg=None):

        self.cfg = cfg or YoloCfg()
        self.RKNN_MODEL = 'best_noquant.rknn'
        self.rknn_lite = RKNNLite()

        ret = self.rknn_lite.load_rknn(self.cfg.RKNN_MODEL)
        if ret != 0:
            raise RuntimeError(f'Load RKNN model failed: {ret}')

        ret = self.rknn_lite.init_runtime(core_mask=RKNNLite.NPU_CORE_ALL)
        if ret != 0:
            raise RuntimeError(f'Init runtime failed: {ret}')

    def detect(self, image: np.ndarray):

        ori_img = image.copy()
        img = cv2.resize(ori_img, (self.cfg.INPUT_SIZE, self.cfg.INPUT_SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.expand_dims(img, 0)       

        outputs = self.rknn_lite.inference(inputs=[img], data_format='nhwc')[0][0]
        results_after_nms = self._nms(outputs)  

        results_in_list = [x for x in results_after_nms if x[-2] > self.cfg.YOLO_CONF and x[-1] in self.cfg.CLS]

        results = self.to_dict(results_in_list)

        return(results)
    
    def release(self):

        self.rknn_lite.release()

    def to_dict(self, boxes: list):

        dict_results = {}

        for bbox in boxes:

            cls = int(bbox[-1])
            if cls not in dict_results:
                dict_results[cls] = []
            dict_results[cls].append(bbox[:-1])
        
        return dict_results

    def _nms(self, boxes: list):

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

            x1 = boxes_np[:, 0]
            y1 = boxes_np[:, 1]
            x2 = boxes_np[:, 2]
            y2 = boxes_np[:, 3]
            scores = boxes_np[:, 4]
            
            areas = (x2 - x1) * (y2 - y1)           
            order = np.argsort(scores)[::-1]
            
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
                
                w = np.maximum(0.0, xx2 - xx1)
                h = np.maximum(0.0, yy2 - yy1)
                
                inter = w * h
                iou = inter / (areas[i] + areas[order[1:]] - inter)
                
                order = order[1:][iou <= self.cfg.YOLO_IOU]
            
            result.extend(boxes_np[keep].tolist())
        
        return result