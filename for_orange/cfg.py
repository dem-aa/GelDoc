from dataclasses import dataclass

@dataclass
class YoloCfg:

    # for YOLO
    CLS: tuple = (0, 1, 2)
    TRESHOLD: float = 0.3
    INPUT_SIZE: int = 1280
    YOLO_CONF: float = 0.3
    YOLO_IOU: float = 0.1
    RKNN_MODEL: str = 'best_noquant.rknn'
    ONNX_MODEL: str = 'GelDoc/for_orange/best.onnx'


@dataclass
class ClassicalCfg: 

    # for classical method
    CLAHE_CLIP: float = 4.0
    SG_WINDOW: int = 11
    SG_POLYORDER: int = 3
    RB_RADIUS: int = 50

    POSITION_TOL: int = 10
    SHIFT_TOL: int = 25
    SNR_MIN: float = 3.0

    ALPHA: float = 0.5