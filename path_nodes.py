import cv2
import os
import random
import time

class MultiplePathsInput:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "inputcount": ("INT", {"default": 1, "min": 1, "max": 1000, "step": 1}),
                "path_1": ("PATH",),
            },
        }

    RETURN_TYPES = ("PATH",)
    RETURN_NAMES = ("paths",)
    FUNCTION = "combine"
    CATEGORY = "Comfyui_Qwen3-VL-Instruct"

    @staticmethod
    def convert_path_to_json(file_path):
        if not file_path or not os.path.exists(file_path):
            print(f"Warning: File does not exist at path: {file_path}")
            return None
        
        ext = os.path.splitext(file_path)[1].lower()

        image_exts = [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"]
        video_exts = [".mp4", ".mkv", ".mov", ".avi", ".flv", ".wmv", ".webm", ".m4v"]

        if ext in image_exts:
            return {"type": "image", "image": f"{file_path}"}
        elif ext in video_exts:
            try:
                vidObj = cv2.VideoCapture(file_path)
                if not vidObj.isOpened():
                    print(f"Error: Could not open video file: {file_path}")
                    return None
                
                return {
                    "type": "video",
                    "video": f"{file_path}",
                    "fps": 1.0, 
                }
            except Exception as e:
                print(f"Error processing video file {file_path}: {e}")
                return None
            finally:
                if 'vidObj' in locals() and vidObj.isOpened():
                    vidObj.release()
        else:
            return None

    def combine(self, inputcount, **kwargs):
        path_list = []
        for c in range(inputcount):
            path = kwargs.get(f"path_{c + 1}")
            if path:
                json_path = self.convert_path_to_json(path)
                if json_path:
                    path_list.append(json_path)
        return (path_list,)

class LoadMediaFromFolder:
    def __init__(self):
        self.last_folder_path = None
        self.next_file_index = 0

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "folder_path": ("STRING", {"default": "C:/path/to/your/media"}),
                "mode": (["sequential", "random", "first"],),
            },
        }

    RETURN_TYPES = ("PATH", "STRING",)
    RETURN_NAMES = ("path", "filename",)
    FUNCTION = "load_from_folder"
    CATEGORY = "Comfyui_Qwen3-VL-Instruct"

    @classmethod
    def IS_CHANGED(cls, folder_path, mode):
        if mode in ["sequential", "random"]:
            return time.time() 
        return folder_path

    convert_path_to_json = staticmethod(MultiplePathsInput.convert_path_to_json)

    def load_from_folder(self, folder_path, mode):
        if not os.path.isdir(folder_path):
            print(f"Error: Folder not found at path: {folder_path}")
            return ([], "",) 

        supported_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp", ".mp4", ".mkv", ".mov", ".avi", ".flv", ".wmv", ".webm", ".m4v"}
        
        try:
            media_files = sorted([f for f in os.listdir(folder_path) if os.path.splitext(f)[1].lower() in supported_extensions])
        except Exception as e:
            print(f"Error reading directory {folder_path}: {e}")
            return ([], "",)

        if not media_files:
            print(f"Warning: No supported media files found in folder: {folder_path}")
            return ([], "",)

        selected_file = None

        if mode == "first":
            selected_file = media_files[0]
        elif mode == "random":
            selected_file = random.choice(media_files)
        elif mode == "sequential":
            if self.last_folder_path != folder_path:
                self.last_folder_path = folder_path
                self.next_file_index = 0
            
            selected_file = media_files[self.next_file_index]
            self.next_file_index = (self.next_file_index + 1) % len(media_files)
        
        if selected_file:
            full_path = os.path.join(folder_path, selected_file)
            json_data = self.convert_path_to_json(full_path)
            if json_data:
                return ([json_data], selected_file)

        return ([], "",)

NODE_CLASS_MAPPINGS = {
    "MultiplePathsInput": MultiplePathsInput,
    "LoadMediaFromFolder": LoadMediaFromFolder
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MultiplePathsInput": "Multiple Paths Input",
    "LoadMediaFromFolder": "Load Media From Folder"
}
