from fusion_gv.config import FusionConfig
from fusion_gv.preprocess import preprocess

def __getattr__(name):
    if name == "FusionGV":
        from fusion_gv.model import FusionGV
        return FusionGV
    raise AttributeError(f"module 'fusion_gv' has no attribute {name!r}")

__all__ = ["FusionGV", "FusionConfig", "preprocess"]
