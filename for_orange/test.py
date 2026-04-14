import cv2
from rknnlite.api import RKNNLite
import numpy as np

INPUT_SIZE = 1280
RKNN_MODEL = 'best_noquant.rknn'

rknn_lite = RKNNLite()

print('--> Load RKNN model')
ret = rknn_lite.load_rknn(RKNN_MODEL)
if ret != 0:
    print('Load RKNN model failed')
    exit(ret)
print('done')

ori_img = cv2.imread('img_gel_144_rotated.jpg')
img = cv2.resize(ori_img, (INPUT_SIZE, INPUT_SIZE))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = np.transpose(img, (2, 0, 1))
img = np.expand_dims(img, 0)       
img = img.astype(np.float32) / 255.0  

print(img.shape)

ret = rknn_lite.init_runtime(core_mask=RKNNLite.NPU_CORE_ALL)

outputs = rknn_lite.inference(inputs=[img], data_format='nchw')

rknn_lite.release()