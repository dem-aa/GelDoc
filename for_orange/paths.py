from pathlib import Path
from dataclasses import dataclass

@dataclass
class ImagePaths:

    src: Path          
    rotated: Path       
    res: Path          
    mono: Path      
    color: Path 
    classical: Path   


class PathsWire:

    def __init__(self, base_folder: Path):
        self.base = Path(base_folder)
        self._ensure_dirs()
    
    def _ensure_dirs(self):
        for subdir in ['src', 'rotated', 'res', 'img']:
            (self.base / subdir).mkdir(parents=True, exist_ok=True)
    
    def get_paths(self, index: int):

        return ImagePaths(
            src=self.base / 'src' / f'{index}Full.png',
            rotated=self.base / 'rotated' / f'{index}Full_rotated.png',
            res=self.base / 'res' / f'{index}.json',
            mono=self.base / 'img' / f'{index}_mono.png',
            color=self.base / 'img' / f'{index}_color.png',
            classical=self.base / 'res' / f'{index}_classical.json'
        )
    
    @property
    def onnx_model(self) -> Path:
        return self.base / 'best.onnx'

    @property
    def rknn_model(self) -> Path:
        return self.base / 'best_noquant.rknn'
    
    @property
    def src_dir(self) -> Path:
        return self.base / 'src'
    
    @property
    def rotated_dir(self) -> Path:
        return self.base / 'rotated'
    
    @property
    def res_dir(self) -> Path:
        return self.base / 'res'
    
    @property
    def img_dir(self) -> Path:
        return self.base / 'img'