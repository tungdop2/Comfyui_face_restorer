import sys
import os

repo_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, repo_dir)
original_modules = sys.modules.copy()


from .FaceRestore import FaceRestorerLoader, FaceRestorer

NODE_CLASS_MAPPINGS = {
    "FaceRestorerLoader": FaceRestorerLoader,
    "FaceRestorer": FaceRestorer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FaceRestorerLoader": "🤩 Face Restorer Loader",
    "FaceRestorer": "🤩 Face Restorer",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
