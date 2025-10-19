import cv2
import os


class MultiplePathsInput:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "inputcount": ("INT", {"default": 1, "min": 1, "max": 1000, "step": 1}),
                "path_1": ("PATH",),
                "video_sample_strategy": (["auto", "uniform", "custom"], {"default": "auto"}),
                "max_frames": ("INT", {"default": 64, "min": 4, "max": 256, "step": 4}),
                "custom_fps": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 10.0, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("PATH",)
    RETURN_NAMES = ("paths",)
    FUNCTION = "combine"
    CATEGORY = "Comfyui_Qwen3-VL-Instruct"

    @staticmethod
    def calculate_optimal_fps(duration, max_frames, strategy="auto"):
        if duration <= 0:
            return 1.0
            
        if strategy == "uniform":
            optimal_fps = max_frames / duration
            print(f"Uniform sampling: {max_frames} frames / {duration:.1f}s = {optimal_fps:.4f} fps")
        elif strategy == "auto":
            if duration <= 30:
                optimal_fps = 1.0
            elif duration <= 60:
                optimal_fps = min(1.0, max_frames / duration)
            elif duration <= 180:
                optimal_fps = max_frames / duration
            else:
                optimal_fps = max_frames / duration
                estimated_frames = int(duration * optimal_fps)
                if estimated_frames < 30:
                    print(f"Warning: Only {estimated_frames} frames for {duration/60:.1f}min video")
        else:
            optimal_fps = None
        
        return optimal_fps

    @staticmethod
    def estimate_token_usage(estimated_frames, max_pixels):
        if max_pixels <= 256 * 28 * 28:
            tokens_per_frame = 800
        elif max_pixels <= 512 * 28 * 28:
            tokens_per_frame = 1000
        elif max_pixels <= 768 * 28 * 28:
            tokens_per_frame = 1200
        elif max_pixels <= 1280 * 28 * 28:
            tokens_per_frame = 1400
        else:
            tokens_per_frame = 1600
        return estimated_frames * tokens_per_frame

    @staticmethod
    def convert_path_to_json(file_path, strategy="auto", max_frames=64, custom_fps=1.0):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        ext = file_path.split('.')[-1].lower()

        if ext in ["jpg", "jpeg", "png", "bmp", "tiff", "webp"]:
            return {"type": "image", "image": file_path}
        
        elif ext in ["mp4", "mkv", "mov", "avi", "flv", "wmv", "webm", "m4v"]:
            print(f"\n{'='*70}")
            print(f"Processing video: {os.path.basename(file_path)}")
            
            vidObj = None
            try:
                vidObj = cv2.VideoCapture(file_path)
                if not vidObj.isOpened():
                    raise RuntimeError(f"Cannot open video file: {file_path}")
                
                total_frames = int(vidObj.get(cv2.CAP_PROP_FRAME_COUNT))
                original_fps = vidObj.get(cv2.CAP_PROP_FPS)
                width = int(vidObj.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(vidObj.get(cv2.CAP_PROP_FRAME_HEIGHT))
                duration = total_frames / original_fps if original_fps > 0 else 0
                
                print(f"Video Info: {duration:.2f}s, {total_frames} frames, {original_fps:.2f} fps, {width}x{height}")
                
                if strategy == "custom":
                    sample_fps = custom_fps
                else:
                    sample_fps = MultiplePathsInput.calculate_optimal_fps(duration, max_frames, strategy)
                
                estimated_frames = min(int(duration * sample_fps), max_frames)
                estimated_tokens = MultiplePathsInput.estimate_token_usage(estimated_frames, 768 * 28 * 28)
                context_usage = (estimated_tokens / 32000) * 100
                
                print(f"Sampling: {sample_fps:.4f} fps, ~{estimated_frames} frames")
                print(f"Estimated: ~{estimated_tokens:,} tokens ({context_usage:.1f}% of 32K)")
                
                if context_usage > 85:
                    print(f"CRITICAL: Token usage too high! Reduce max_frames or max_pixels")
                elif context_usage > 70:
                    print(f"WARNING: High token usage, consider reducing parameters")
                elif estimated_frames < 16 and duration > 60:
                    print(f"WARNING: Sparse sampling, may miss content")
                
                print(f"{'='*70}\n")
                
            except Exception as e:
                raise RuntimeError(f"Error reading video: {str(e)}")
            finally:
                if vidObj is not None:
                    vidObj.release()
            
            return {"type": "video", "video": file_path, "fps": sample_fps}
        else:
            raise ValueError(f"Unsupported file format: .{ext}")

    def combine(self, inputcount, video_sample_strategy="auto", max_frames=64, custom_fps=1.0, **kwargs):
        path_list = []
        
        for c in range(inputcount):
            path_key = f"path_{c + 1}"
            if path_key not in kwargs:
                raise ValueError(f"Missing required input: {path_key}")
            
            path = kwargs[path_key]
            try:
                path_json = self.convert_path_to_json(path, video_sample_strategy, max_frames, custom_fps)
                path_list.append(path_json)
            except Exception as e:
                print(f"Error processing {path_key}: {str(e)}")
                raise
        
        print(f"Successfully processed {len(path_list)} path(s)\n")
        return (path_list,)
