from pathlib import Path
import argparse

from cfg import YoloCfg, ClassicalCfg
from yolo_detection import YoloDetect
from pre_and_post_proc import PreProc 
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
    

        yolo_cfg = YoloCfg()
        classical_cfg = ClassicalCfg()

        yolo_detector = YoloDetect()

        dirs = PathsWire(folder)
        paths = dirs.get_paths(index)

        img = PreProc.load_img(paths.src) 
        new_img = PreProc.rotate_image(img, paths.rotated)

        res = yolo_detector.detect(new_img)

        PreProc.yolo_out_to_json(res, paths.res)
        PreProc.make_image(res, new_img, paths.color)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', '-f', type=Path, required=True)
    parser.add_argument('--index', type=int, required=True)
    args = parser.parse_args()

    ImagePipeline(folder=args.folder, index=args.index)