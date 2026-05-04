from pathlib import Path
import argparse

from cfg import YoloCfg, ClassicalCfg
# from yolo_detection import YoloDetect
from yolo_detection_PC import YoloDetect
from classical_detection import ClassicalDetect
from pre_and_post_proc import PreProc, PostProc
from paths import PathsWire

# <folder>/
# ├── src/
# │   └── {index}Full.png          ← ВХОД (обязателен)
# ├── rotated/
# │   └── {index}FullRot.png          ← ВХОД (обязателен)
# ├── res/
# │   └── {index}.json             ← ВЫХОД JSON
# └── img/
#     ├── {index}_mono.png          ← ВЫХОД ч/б overlay
#     └── {index}_color.png         ← ВЫХОД цветной overlay

class ImagePipeline:

    def __init__(self, folder: Path, index: int):

        dirs = PathsWire(folder)
        paths = dirs.get_paths(index)

        yolo_detector = YoloDetect(dirs.onnx_model)
        classical_detector = ClassicalDetect()

        img = PreProc.load_img(paths.src) 
        new_img, cords = PreProc.rotate_and_crop_image(img, paths.rotated)

        res = yolo_detector.detect(new_img)

        PreProc.to_json(res, paths.res)
        PreProc.make_image(res, new_img, paths.color)

        new_res = PostProc.check_in_gel(new_img, res, cords)
        new_res = PostProc.add_new_zeros_and_lines(new_img, res, cords)
        PreProc.make_image(new_res, new_img, paths.color_new)

        cls_res = classical_detector.detect(new_img, res, cords)   

        PreProc.to_json(cls_res, paths.classical)
        PreProc.make_cls_image(cls_res, new_img, paths.color_cls)

        PostProc.make_final_image(res, cls_res, new_img, paths.color_dec)
        

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', '-f', type=Path, required=True)
    parser.add_argument('--index', type=int, required=True)
    args = parser.parse_args()

    ImagePipeline(folder=args.folder, index=args.index)