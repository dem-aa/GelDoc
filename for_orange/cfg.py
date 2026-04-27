from dataclasses import dataclass

@dataclass
class YoloCfg:

    # for YOLO
    CLS: tuple = (0, 1, 2)
    TRESHOLD: float = 0.3
    INPUT_SIZE: int = 1280
    YOLO_CONF: float = 0.2
    YOLO_IOU: float = 0.1
 
@dataclass
class ClassicalCfg:
 
        CLAHE_CLIP: float = 2.0
        SG_WINDOW: int = 11
        SG_POLYORDER: int = 3
        RB_RADIUS: int = 5
        SNR_MIN: float = 2.0  
        SHIFT_TOL: int = 5  
        ALPHA: float = 0.5   
        X_TOLERANCE: int = 5 