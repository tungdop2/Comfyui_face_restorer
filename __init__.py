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
