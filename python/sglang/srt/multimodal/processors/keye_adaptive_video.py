from __future__ import annotations

from typing import Tuple, Dict, Any, Union
import base64
import logging
import math
from functools import lru_cache
import time
import warnings
import itertools
import io as py_io
import os.path as osp
import cv2
import os
import random
import numpy as np
import copy
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import collections

from io import BytesIO
from packaging import version
from PIL import Image
from torchvision import io, transforms
from torchvision.transforms import InterpolationMode
from einops import rearrange
from typing_extensions import TypeAlias
from sglang.srt.utils import get_int_env_var


logger = logging.getLogger(__name__)

IMAGE_FACTOR = 28
# min tokens per image
MIN_TOKENS = 4
# max tokens per image
MAX_TOKENS = get_int_env_var("IMAGE_MAX_TOKENS", default=20480)
MIN_PIXELS = MIN_TOKENS * IMAGE_FACTOR * IMAGE_FACTOR  # 4 * 28 * 28 = 3,136
MIN_PIXELS = get_int_env_var("MIN_PIXELS", default=MIN_PIXELS)
MAX_PIXELS = MAX_TOKENS * IMAGE_FACTOR * IMAGE_FACTOR  # 20480 * 28 * 28 = 16,056,320
MAX_PIXELS = get_int_env_var("MAX_PIXELS", default=MAX_PIXELS)
MAX_RATIO = 200

# min tokens per video frame
VIDEO_MIN_TOKENS = get_int_env_var("VIDEO_MIN_TOKENS", default=64)
# max tokens per video frame
VIDEO_MAX_TOKENS = get_int_env_var("VIDEO_MAX_TOKENS", default=256)
# min pixels per video frame
VIDEO_MIN_PIXELS = (
    VIDEO_MIN_TOKENS * IMAGE_FACTOR * IMAGE_FACTOR
)  # 32 * 28 * 28 = 25,088
# max pixels per video frame
VIDEO_MAX_PIXELS = (
    VIDEO_MAX_TOKENS * IMAGE_FACTOR * IMAGE_FACTOR
)  # 768 * 28 * 28 = 602,112
VIDEO_TOTAL_MAX_TOKENS = get_int_env_var("VIDEO_TOTAL_MAX_TOKENS", default=184320)
# max total pixels per video
VIDEO_TOTAL_PIXELS = (
    VIDEO_TOTAL_MAX_TOKENS * IMAGE_FACTOR * IMAGE_FACTOR
)  # 65,536 * 28 * 28 = 51,380,224
VIDEO_TOTAL_PIXELS = get_int_env_var("VIDEO_TOTAL_PIXELS", default=VIDEO_TOTAL_PIXELS)
# default fps
FPS = 2.0

FAST_TOKEN_RATIO = 0.3

# Slow-Fast帧最小相似度，低于阈值需要重新建立Slow帧，降低该阈值会创建更多的Fast帧
MIN_FRAME_SIMILARITY = 0.9

# 视频读取 timeout 时间
VIDEO_READ_TIMEOUT = int(os.getenv("VIDEO_READ_TIMEOUT", 600))
ENABLE_ADAPTIVE_VIDEO_TOKEN = os.environ.get("ENABLE_ADAPTIVE_VIDEO_TOKEN", "False")
if ENABLE_ADAPTIVE_VIDEO_TOKEN in ["1", "true", "True"]:
    ENABLE_ADAPTIVE_VIDEO_TOKEN = True
else:
    ENABLE_ADAPTIVE_VIDEO_TOKEN = False


def get_assistant_mask(batch_input_ids: torch.Tensor,
                       start_pattern: Optional[List[int]],
                       end_pattern: Optional[List[int]],
                       replacement: Dict = None
                       ):
    batch_input_ids = batch_input_ids.clone()
    if replacement is not None:
        for k, v in replacement.items():
            batch_input_ids[batch_input_ids==k] = v

    if not start_pattern:
        start_pattern = [151644, 77091, 198]
    if not end_pattern:
        end_pattern = [151645, 198]

    masks = []
    for input_ids in batch_input_ids:
        mask = []
        assistant_start = []
        assistant_end = []
        to_mask = False
        for _id in input_ids:
            mask.append(int(to_mask))
            if not to_mask:
                if _id in start_pattern:
                    assistant_start.append(_id.item())
                else:
                    assistant_start = []
                if assistant_start[-len(start_pattern):] == start_pattern:
                    if len(start_pattern) + 1 < len(mask) and mask[-len(start_pattern)-1] == 1:
                        for i in range(0, len(start_pattern)):
                            mask[-1 -i] = 1

                    to_mask = True
                    assistant_start = []
            else:
                if _id in end_pattern:
                    assistant_end.append(_id.item())
                else:
                    assistant_end = []
                if assistant_end[-len(end_pattern):] == end_pattern:
                    to_mask = False
                    assistant_end = []
        masks.append(mask)
    return torch.tensor(masks)


def round_by_factor(number: int, factor: int) -> int:
    """Returns the closest integer to 'number' that is divisible by 'factor'."""
    return round(number / factor) * factor


def ceil_by_factor(number: int, factor: int) -> int:
    """Returns the smallest integer greater than or equal to 'number' that is divisible by 'factor'."""
    return math.ceil(number / factor) * factor


def floor_by_factor(number: int, factor: int) -> int:
    """Returns the largest integer less than or equal to 'number' that is divisible by 'factor'."""
    return math.floor(number / factor) * factor


def smart_resize(
        height: int, width: int,
        factor: int = IMAGE_FACTOR,
        min_pixels: int = MIN_PIXELS,
        max_pixels: int = MAX_PIXELS) -> Tuple[int, int]:
    """
    Rescales the image so that the following conditions are met:

    1. Both dimensions (height and width) are divisible by 'factor'.

    2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].

    3. The aspect ratio of the image is maintained as closely as possible.
    """
    # if int(height < factor//4) + int(width < factor//4):
    #     raise ValueError(f"height:{height} or width:{width} must be larger than factor:{factor//4}")

    if max(height, width) / min(height, width) > MAX_RATIO:
        raise ValueError(
            f"absolute aspect ratio must be smaller than {MAX_RATIO}, got {max(height, width) / min(height, width)}"
        )
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
    return max(h_bar, factor), max(w_bar, factor)


def fetch_image(ele: Dict[str, str | Image.Image],
                size_factor: int = IMAGE_FACTOR,
                is_video = False) -> Image.Image:
    if "image" in ele:
        image = ele["image"]
    else:
        image = ele["image_url"]
    image_obj = None
    if isinstance(image, Image.Image):
        image_obj = image
    elif image.startswith("http://") or image.startswith("https://"):
        image_obj = Image.open(requests.get(image, stream=True).raw)
    elif image.startswith("file://"):
        image_obj = Image.open(image[7:])
    elif image.startswith("data:image"):
        if "base64," in image:
            _, base64_data = image.split("base64,", 1)
            data = base64.b64decode(base64_data)
            image_obj = Image.open(BytesIO(data))
    else:
        image_obj = Image.open(image)
    if image_obj is None:
        raise ValueError(
            f"Unrecognized image input, support local path, http url, base64 and PIL.Image, got {image}")
    image = image_obj.convert("RGB")  ## resize
    if "resized_height" in ele and "resized_width" in ele:
        resized_height, resized_width = smart_resize(
            ele["resized_height"],
            ele["resized_width"],
            factor=size_factor,
        )
    else:
        width, height = image.size
        # 以image list形式传入的视频
        if is_video:
            min_pixels = ele.get("min_pixels", VIDEO_MIN_PIXELS)
            max_pixels = ele.get("max_pixels", VIDEO_MAX_PIXELS)
        else:
            min_pixels = ele.get("min_pixels", MIN_PIXELS)
            max_pixels = ele.get("max_pixels", MAX_PIXELS)
        resized_height, resized_width = smart_resize(
            height,
            width,
            factor=size_factor,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )

    image = image.resize((resized_width, resized_height))

    return image

def smart_nframes(
        ele: dict,
        total_frames: int,
        video_fps: int | float) -> int:
    """calculate the number of frames for video used for model inputs.

    Args:
        ele (dict): a dict contains the configuration of video.
            support either `fps` or `nframes`:
                - nframes: the number of frames to extract for model inputs.
                - fps: the fps to extract frames for model inputs.
                    - min_frames: the minimum number of frames of the video, only used when fps is provided.
                    - max_frames: the maximum number of frames of the video, only used when fps is provided.
        total_frames (int): the original total number of frames of the video.
        video_fps (int | float): the original fps of the video.

    Raises:
        ValueError: nframes should in interval [FRAME_FACTOR, total_frames].

    Returns:
        int: the number of frames for video used for model inputs.
    """
    # TODO: 兼容image list形式
    fps = ele.get("fps", FPS) # 应该是走的默认FPS，按照每秒抽两帧来算
    fps = min(fps, video_fps) # 注意，这里的video_fps是真实的后验FPS
    min_pixels = int(VIDEO_MIN_PIXELS)
    max_pixels = int(VIDEO_MAX_PIXELS)
    total_pixels = int(ele.get("video_total_pixels", VIDEO_TOTAL_PIXELS))
    min_frames_by_pixels = 1
    max_frames_by_pixels = total_frames
    if min_pixels > 0 and max_pixels > 0 and total_pixels > 0:
        min_frames_by_pixels = max(1, math.ceil(total_pixels / max_pixels))
        max_frames_by_pixels = max(1, math.floor(total_pixels / min_pixels))
    max_frames = min(ele.get("max_frames", max_frames_by_pixels), max_frames_by_pixels, total_frames)
    min_frames = ele.get("min_frames", 2)
    fps_nframes = int(total_frames / video_fps * fps) # 换算为秒数，之后计算希望抽多少帧
    nframes = min(max(fps_nframes, min_frames), max_frames)
    return nframes


def get_video_total_pixels_by_duration(duration: float, num_videos: int = 1, video_total_pixels: int = None) -> int:
    """Return video_total_pixels based on video duration in seconds.
    When num_videos > 1, the budget is divided equally among videos."""
    total_pixels = video_total_pixels if video_total_pixels is not None else VIDEO_TOTAL_PIXELS
    base = total_pixels / max(num_videos, 1)
    if num_videos == 1 and ENABLE_ADAPTIVE_VIDEO_TOKEN:
        if duration <= 259:
            return 0.09 * base
        if duration <= 518:
            return 0.18 * base
        if duration <= 1036:
            return 0.36 * base
        if duration <= 2016:
            return 0.7 * base
    return base


def set_video_total_pixels_by_duration(ele: dict, total_frames: int, video_fps: float) -> None:
    """Set ele["video_total_pixels"] by duration derived from total_frames and video_fps."""
    if not video_fps or video_fps <= 0:
        return
    duration = total_frames / video_fps
    if isinstance(ele, dict):
        num_videos = ele.get("_num_videos", 1)
        user_total_pixels = ele.get("video_total_pixels", None)
        ele["video_total_pixels"] = get_video_total_pixels_by_duration(duration, num_videos=num_videos, video_total_pixels=user_total_pixels)


def _read_video_torchvision(ele: Dict) -> Tuple[torch.Tensor, float]:
    """read video using torchvision.io.read_video

    Args:
        ele (dict): a dict contains the configuration of video.
        support keys:
            - video: the path of video. support "file://", "http://", "https://" and local path.
            - video_start: the start time of video.
            - video_end: the end time of video.
    Returns:
        torch.Tensor: the video tensor with shape (T, C, H, W).
    """
    # process video url
    st = time.time()
    if isinstance(ele["video"], str):
        video_path = ele["video"]
        if version.parse(torchvision.__version__) < version.parse("0.19.0"):
            if "http://" in video_path or "https://" in video_path:
                warnings.warn("torchvision < 0.19.0 does not support http/https video path, please upgrade to 0.19.0.")
            if "file://" in video_path:
                video_path = video_path[7:]
        video, audio, info = io.read_video(
            video_path,
            start_pts=ele.get("video_start", 0.0),
            end_pts=ele.get("video_end", None),
            pts_unit="sec",
            output_format="TCHW",
        )
        total_frames, video_fps = video.size(0), info["video_fps"]
        logger.info(f"torchvision:  {video_path=}, {total_frames=}, {video_fps=}, time={time.time() - st:.3f}s")

    elif isinstance(ele["video"], bytes):
        video_reader = torchvision.io.VideoReader(ele["video"], "video")
        video_meta = video_reader.get_metadata()["video"]

        start_ptr = ele.get("video_start", 0.0)
        end_pts = ele.get("video_end", video_meta["duration"][-1])
        video = []
        for frame in itertools.takewhile(lambda x: x['pts'] <= end_pts, video_reader.seek(start_ptr)):
            video.append(frame['data'])
        video = torch.stack(video)
        total_frames, video_fps = video.size(0), video_meta["fps"][-1]
        logger.info(f"torchvision:  {total_frames=}, {video_fps=}, time={time.time() - st:.3f}s")

    set_video_total_pixels_by_duration(ele, total_frames=total_frames, video_fps=video_fps)
    nframes = smart_nframes(ele, total_frames=total_frames, video_fps=video_fps)
    indices = torch.linspace(0, total_frames - 1, nframes).round().long()
    frames = video[indices]
    timestamps = torch.FloatTensor([(1 / video_fps) * i for i in range(nframes)])
    timestamps = timestamps[indices]

    frame_types = torch.zeros(size=(frames.size(0), ), dtype=torch.int32)

    video_meta = {
        "total_frames": int(total_frames),
        "video_fps": float(video_fps),
    }
    return frames, timestamps.tolist(), frame_types, video_meta

def is_torchcodec_available() -> bool:
    import importlib.util

    return importlib.util.find_spec("torchcodec") is not None

def _read_video_torchcodec(ele: Dict) -> Tuple[torch.Tensor, float]:
    """read video using torchvision.io.read_video

    Args:
        ele (dict): a dict contains the configuration of video.
        support keys:
            - video: the path of video. support "file://", "http://", "https://" and local path.
            - video_start: the start time of video.
            - video_end: the end time of video.
    Returns:
        torch.Tensor: the video tensor with shape (T, C, H, W).
    """
    # process video url
    from torchcodec.decoders import VideoDecoder
    st = time.time()
    if isinstance(ele["video"], str):
        video_path = ele["video"]
        TORCHCODEC_NUM_THREADS = int(os.environ.get('TORCHCODEC_NUM_THREADS', 8))
        decoder = VideoDecoder(video_path, num_ffmpeg_threads=TORCHCODEC_NUM_THREADS)

        total_frames = decoder.metadata.num_frames
        video_fps = decoder.metadata.average_fps

        logger.info(f"torchvision:  {video_path=}, {total_frames=}, {video_fps=}, time={time.time() - st:.3f}s")

    elif isinstance(ele["video"], bytes):
        video_reader = torchvision.io.VideoReader(ele["video"], "video")
        video_meta = video_reader.get_metadata()["video"]

        start_ptr = ele.get("video_start", 0.0)
        end_pts = ele.get("video_end", video_meta["duration"][-1])
        video = []
        for frame in itertools.takewhile(lambda x: x['pts'] <= end_pts, video_reader.seek(start_ptr)):
            video.append(frame['data'])
        video = torch.stack(video)
        total_frames, video_fps = video.size(0), video_meta["fps"][-1]
        logger.info(f"torchvision:  {total_frames=}, {video_fps=}, time={time.time() - st:.3f}s")

    set_video_total_pixels_by_duration(ele, total_frames=total_frames, video_fps=video_fps)
    nframes = smart_nframes(ele, total_frames=total_frames, video_fps=video_fps)
    indices = torch.linspace(0, max(total_frames - 1, 0), nframes).round().long()
    indices = torch.clamp(indices, 0, max(total_frames - 1, 0))
    if isinstance(ele["video"], str):
        frames = decoder.get_frames_at(indices=indices.tolist()).data
    else:
        frames = video[indices]
    timestamps = indices.float() / video_fps

    frame_types = torch.zeros(size=(frames.size(0), ), dtype=torch.int32)

    video_meta = {
        "total_frames": int(total_frames),
        "video_fps": float(video_fps),
    }
    return frames, timestamps.tolist(), frame_types, video_meta

def _read_video_ffmpeg(ele: Dict) -> Tuple[torch.Tensor, float]:
    st = time.time()
    import ffmpeg
    import gc

    video_path = ele["video"]

    # 使用 try-finally 确保资源清理
    try:
        probe = ffmpeg.probe(video_path)
        video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
        width = int(video_info['width'])
        height = int(video_info['height'])
        total_frames = int(video_info['nb_frames'])
        
        # 更安全的帧率解析，避免使用 eval()
        frame_rate_str = video_info["avg_frame_rate"]
        if '/' in frame_rate_str:
            num, den = map(int, frame_rate_str.split('/'))
            video_fps = num / den if den != 0 else 30.0
        else:
            video_fps = float(frame_rate_str)

        # 创建 ffmpeg 进程并显式清理
        process = (
            ffmpeg
            .input(video_path)
            .output('pipe:', format='rawvideo', pix_fmt='rgb24')
            .run_async(pipe_stdout=True, pipe_stderr=True)
        )
        
        # 读取输出并确保进程正确关闭
        out, err = process.communicate()
        
        # 显式关闭进程
        if process.returncode != 0:
            raise RuntimeError(f"FFmpeg 失败: {err.decode()}")

        #print(height, width)

        # 创建 numpy 数组并立即复制数据以断开缓冲区引用
        video_data = np.frombuffer(out, np.uint8).copy()  # .copy() 断开缓冲区引用
        video = video_data.reshape([-1, 3, height, width])

        # 显式删除大型变量
        del out, video_data
        gc.collect()  # 强制垃圾回收

        #print(f"ffmpeg: {video_path=}, {video_fps=}, time={time.time() - st:.3f}s")

        return torch.from_numpy(video), [0] * len(video), [0] * len(video)
        
    except Exception as e:
        # 错误时确保清理
        gc.collect()
        raise e

def is_decord_available() -> bool:
    import importlib.util

    return importlib.util.find_spec("decord") is not None

def get_frame_sim(frame1, frame2,
                  patch_size: int=28,
                  threshold: float = 0.7,
                  epsilon: float=1e-8):
    assert frame1.dim() == 3 and frame2.dim() == 3, "输入必须是3D张量 [C, H, W]"
    
    # 将PyTorch张量转换为OpenCV格式的numpy数组
    def to_numpy_cvt(tensor):
        # 确保张量在CPU上并转换为HWC格式
        tensor = tensor.cpu().permute(1, 2, 0).numpy()
        if tensor.dtype == np.float32 or tensor.dtype == np.float64:
            tensor = (tensor).astype(np.uint8)
        # 转换为HSV颜色空间
        return cv2.cvtColor(tensor, cv2.COLOR_RGB2HSV)

    # 转换颜色空间
    frame1_hsv = to_numpy_cvt(frame1)
    frame2_hsv = to_numpy_cvt(frame2)

    # 将HSV图像转回PyTorch张量
    frame1_tensor = torch.from_numpy(frame1_hsv).permute(2, 0, 1).to(frame1.device).float()
    frame2_tensor = torch.from_numpy(frame2_hsv).permute(2, 0, 1).to(frame2.device).float()

    # 分块处理
    patch1 = rearrange(
        frame1_tensor, "c (h p1) (w p2) -> h w (c p1 p2)", p1=patch_size, p2=patch_size).float()
    patch2 = rearrange(
        frame2_tensor, "c (h p1) (w p2) -> h w (c p1 p2)", p1=patch_size, p2=patch_size).float()

    norm1 = torch.norm(patch1, p=2, dim=-1, keepdim=True) + epsilon
    norm2 = torch.norm(patch2, p=2, dim=-1, keepdim=True) + epsilon
    
    normalized1 = patch1 / norm1
    normalized2 = patch2 / norm2
    cos_sim = (normalized1 * normalized2).sum(dim=-1)
    
    zero_vector_mask = (norm1.squeeze() < 0.01) & (norm2.squeeze() < 0.01)  # 全黑图
    
    similar = torch.ones_like(cos_sim)  # 默认全部相似
    
    non_zero_mask = ~zero_vector_mask
    similar[non_zero_mask] = (cos_sim[non_zero_mask] > threshold).float()

    return similar[non_zero_mask].float().mean().item()

def _read_video_decord(
        ele: dict,
) -> torch.Tensor:
    """read video using decord.VideoReader

    Args:
        ele (dict): a dict contains the configuration of video.
        support keys:
            - video: the path of video. support "file://", "http://", "https://" and local path.
            - video_start: the start time of video.
            - video_end: the end time of video.
    Returns:
        torch.Tensor: the video tensor with shape (T, C, H, W).
    """
    import decord
    st = time.time()
    if isinstance(ele["video"], bytes):
        video_path = ""
        fp = py_io.BytesIO(ele["video"])
        vr = decord.VideoReader(fp)
    else:
        video_path = ele["video"]
        vr = decord.VideoReader(video_path)
    # TODO: support start_pts and end_pts

    if 'video_start' in ele or 'video_end' in ele:
        raise NotImplementedError("not support start_pts and end_pts in decord for now.")
    nframes, video_fps = len(vr), vr.get_avg_fps()
    # timestamp start from 0.0
    timestamps = torch.FloatTensor([(1 / video_fps) * i for i in range(nframes)])

    set_video_total_pixels_by_duration(ele, total_frames=nframes, video_fps=video_fps)
    final_nframes = smart_nframes(ele, total_frames=nframes, video_fps=video_fps)

    indices = torch.linspace(0, nframes - 1, final_nframes).round().long()
    frames = vr.get_batch(indices.tolist()).asnumpy().copy()
    frames = torch.tensor(frames).permute(0, 3, 1, 2)
    logger.debug(f"Decord: {video_path=}, {nframes=}, {video_fps=}, time={time.time() - st:.3f}s")
    timestamps = timestamps[indices]

    frame_types = torch.zeros(size=(frames.size(0), ), dtype=torch.int32)
    logger.debug(f"Read video:  {video_path=}, {nframes=}, {video_fps=}, time={time.time() - st:.3f}s")

    video_meta = {
        "total_frames": int(nframes),
        "video_fps": float(video_fps),
    }
    return frames, timestamps, frame_types, video_meta


VIDEO_READER_BACKENDS = {
    "decord": _read_video_decord,
    "torchvision": _read_video_torchvision,
    "slowfast_torchvision": _read_video_torchvision,
    "slowfast_decord": _read_video_decord,
    "torchcodec": _read_video_torchcodec
}


FORCE_KEYE_VIDEO_READER = os.getenv("FORCE_KEYE_VIDEO_READER", None)


@lru_cache(maxsize=1)
def get_video_reader_backend() -> str:
    if FORCE_KEYE_VIDEO_READER is not None:
        video_reader_backend = FORCE_KEYE_VIDEO_READER
    elif is_torchcodec_available():
        video_reader_backend = "torchcodec"
    elif is_decord_available():
        video_reader_backend = "decord"
    else:
        video_reader_backend = "torchvision"
    # return video_reader_backend
    # Hack
    return video_reader_backend


import multiprocessing, signal
class SafeExecutor:
    def __init__(self):
        self.parent_conn, self.child_conn = multiprocessing.Pipe()
        self.process = None
        self._start_worker()

    def _start_worker(self):
        """Starts (or restarts) the worker process."""
        if self.process and self.process.is_alive():
            self._kill_process(self.process)

        # Clean up old pipes to prevent reading leftover garbage
        if self.parent_conn:
            self.parent_conn.close()
        if self.child_conn:
            self.child_conn.close()
        self.parent_conn, self.child_conn = multiprocessing.Pipe()

        # We run a loop inside the process so we don't pay startup costs every time
        self.process = multiprocessing.Process(target=self._worker_loop, args=(self.child_conn,), daemon=True)
        self.process.start()

    def _kill_process(self, process):
        try:
            os.kill(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except AttributeError:
            process.terminate()

        process.join()

    @staticmethod
    def _worker_loop(conn):
        """This runs inside the separate process forever."""
        while True:
            try:
                # Wait for a function and args
                func, args, kwargs = conn.recv()
                result = func(*args, **kwargs)
                conn.send(("OK", result))
            except Exception as e:
                conn.send(("ERROR", e))

    def run(self, timeout, func, *args, **kwargs):
        """The main entry point."""
        # 1. Send task to worker
        self.parent_conn.send((func, args, kwargs))

        # 2. Wait for result with timeout (poll is efficient)
        if self.parent_conn.poll(timeout):
            try:
                status, payload = self.parent_conn.recv()
            except EOFError:
                self._start_worker()
                raise RuntimeError("Worker process died unexpectedly")
            if status == "ERROR":
                raise payload
            return payload
        else:
            # 3. TIMEOUT OCCURRED
            # We must kill the worker because it's stuck in C code
            self._start_worker() # Kill old, start new
            raise TimeoutError("Function timed out (Worker restarted)")

    def close(self):
        if self.process:
            self._kill_process(self.process)


_VIDEO_PROCESSING_EXECUTOR = None
def _get_video_processing_executor():
    global _VIDEO_PROCESSING_EXECUTOR
    if _VIDEO_PROCESSING_EXECUTOR is None:
        _VIDEO_PROCESSING_EXECUTOR = SafeExecutor()
    return _VIDEO_PROCESSING_EXECUTOR


def fetch_video(ele: Dict, image_factor: int = IMAGE_FACTOR, num_videos: int = 1) -> Dict[str, Any]:
    if num_videos > 1:
        ele["_num_videos"] = num_videos
    if isinstance(ele["video"], str) or isinstance(ele["video"], bytes):
        video_reader_backend = get_video_reader_backend()
        fallback_backends = [b for b in ["decord", "torchvision"] if b != video_reader_backend
                             and b in VIDEO_READER_BACKENDS]
        try:
            result = _get_video_processing_executor().run(VIDEO_READ_TIMEOUT, VIDEO_READER_BACKENDS[video_reader_backend], ele)
        except TimeoutError:
            raise TimeoutError(f"video {ele} reading timed out after {VIDEO_READ_TIMEOUT} seconds")
        except Exception as e:
            logger.warning(f"video_reader_backend {video_reader_backend} error: {e}")
            result = None
            for fb in fallback_backends:
                try:
                    logger.warning(f"fallback to {fb}")
                    result = _get_video_processing_executor().run(VIDEO_READ_TIMEOUT, VIDEO_READER_BACKENDS[fb], ele)
                    break
                except Exception as fb_e:
                    logger.warning(f"fallback {fb} also failed: {fb_e}")
            if result is None:
                raise RuntimeError(f"All video backends failed for {ele.get('video', '')}")
        video_meta = None
        if isinstance(result, (list, tuple)) and len(result) == 4:
            frames, timestamps, frame_types, video_meta = result
        else:
            frames, timestamps, frame_types = result
    else:
        # TODO: image list没有走smart_nframes，所以可能会超过video_total_pixels
        assert isinstance(ele["video"], (list, tuple))
        process_info = ele.copy()
        process_info.pop("type", None)
        process_info.pop("video", None)
        images = []
        for video_element in ele["video"]:
            # preprocess images
            if isinstance(video_element, dict):
                images.append(
                    fetch_image(video_element, size_factor=image_factor, is_video = True))
            else:
                images.append(
                    fetch_image(
                        {"image": video_element, **process_info},
                        size_factor=image_factor, is_video = True)
                )

        images_tensor = [torch.from_numpy(np.array(image)).permute(2, 0, 1) for image in images]
        frames = torch.stack(images_tensor, dim=0)
        nframes = len(images)
        video_fps = ele.get("fps", None)
        timestamps = None
        # 如果用户主动提供了fps，按照fps来估算timestames
        # 如果没有提供，不会按默认的fps去算
        if video_fps:
            assert isinstance(video_fps, Union[float, int]) and video_fps > 0, \
                "Invalid fps, should be float or int"
            timestamps = torch.FloatTensor([(1 / video_fps) * i for i in range(nframes)])
        if num_videos > 1 and "video_total_pixels" not in ele:
            ele["video_total_pixels"] = int(VIDEO_TOTAL_PIXELS / num_videos)
        final_nframes = smart_nframes(ele, total_frames=nframes, video_fps=ele.get("fps", FPS))
        indices = torch.linspace(0, nframes - 1, final_nframes).round().long()
        frames = frames[indices]
        if timestamps is not None:
            timestamps = timestamps[indices]
        frame_types = torch.zeros(size=(frames.size(0), ), dtype=torch.int32)

    nframes, _, height, width = frames.shape
    if isinstance(ele["video"], str) or isinstance(ele["video"], bytes):
        total_frames = None
        video_fps = None
        if isinstance(video_meta, dict):
            total_frames = video_meta.get("total_frames")
            video_fps = video_meta.get("video_fps")
        if isinstance(total_frames, int) and isinstance(video_fps, (int, float)) and video_fps > 0:
            set_video_total_pixels_by_duration(ele, total_frames=total_frames, video_fps=video_fps)

    min_pixels = max(int(ele.get("min_pixels", VIDEO_MIN_PIXELS)), VIDEO_MIN_PIXELS)
    max_pixels = int(ele.get("max_pixels", VIDEO_MAX_PIXELS))
    total_pixels = int(ele.get("video_total_pixels", VIDEO_TOTAL_PIXELS / max(num_videos, 1)))
    left = int(min_pixels / IMAGE_FACTOR / IMAGE_FACTOR)
    right = int(max_pixels / IMAGE_FACTOR / IMAGE_FACTOR)
    if left < 1:
        left = 1
    if right < left:
        right = left

    def estimate_total_pixels(tokens_per_frame):
        return nframes * tokens_per_frame * IMAGE_FACTOR * IMAGE_FACTOR

    while left < right:
        mid = int(left + right) // 2
        if estimate_total_pixels(mid) > total_pixels:
            right = mid
        else:
            left = mid + 1

    max_pixels = left * IMAGE_FACTOR * IMAGE_FACTOR
    resized_height, resized_width = smart_resize(
        height,
        width,
        factor=IMAGE_FACTOR,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )

    processor_kwargs = {
        "height": resized_height,
        "width": resized_width,
        "fast_height": resized_height,
        "fast_width": resized_width,
    }
    if timestamps is not None:
        processor_kwargs["timestamps"] = timestamps
    if frame_types is not None:
        processor_kwargs["frame_types"] = frame_types
    return frames, processor_kwargs


def extract_vision_info(conversations: list[dict] | list[list[dict]]) -> list[dict]:
    vision_infos = []
    if isinstance(conversations[0], dict):
        conversations = [conversations]
    for conversation in conversations:
        for message in conversation:
            if isinstance(message["content"], list):
                for ele in message["content"]:
                    if (
                            "image" in ele
                            or "image_url" in ele
                            or "video" in ele
                            or ele["type"] in ("image", "image_url", "video")
                    ):
                        vision_infos.append(ele)
    return vision_infos


def process_vision_info(
        conversations: list[dict] | list[list[dict]] = None, vision_infos: list[dict] = None,
        image_factor: int = IMAGE_FACTOR
) -> tuple[list[Image.Image] | None, list[torch.Tensor | list[Image.Image]] | None]:
    assert conversations is not None or vision_infos is not None
    torch.set_num_threads(1)

    if vision_infos is None:
        vision_infos = extract_vision_info(conversations)
    ## Read images or videos
    image_inputs = []
    video_inputs = []
    processor_kwargs = collections.defaultdict(list)
    num_videos = sum(1 for v in vision_infos if "video" in v)
    for vision_info in vision_infos:
        if "image" in vision_info or "image_url" in vision_info:
            image_inputs.append(fetch_image(vision_info, image_factor))
        elif "video" in vision_info:
            # TODO: 这是为了处理站内预抽帧的视频；对外开源的版本，不需要下面这段逻辑
            if isinstance(vision_info["video"], str) and "480p_60s_4fps_v2" in vision_info["video"]:
                path = vision_info["video"]
                pid_str = osp.basename(osp.splitext(path)[0])
                if not osp.exists(path):
                    post = str(int(pid_str[-4:]))
                    path = path.replace("480p_60s_4fps_v2", "480p_60s_4fps_0215_0316/{}".format(post))
                vision_info["video"] = path
            _video_inputs, _processor_kwargs = fetch_video(vision_info, image_factor, num_videos=num_videos)
            video_inputs.append(_video_inputs)
            for k, v in _processor_kwargs.items():
                processor_kwargs[k].append(v)
        else:
            raise ValueError("image, image_url or video should in content.")
    if len(image_inputs) == 0:
        image_inputs = None
    if len(video_inputs) == 0:
        video_inputs = None
    return image_inputs, video_inputs, processor_kwargs


def get_rope_index_slowfast(
    input_ids: torch.LongTensor,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    fast_video_grid_thw: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    spatial_merge_size: Optional[int] = None,
    image_token_id: Optional[int] = None,
    video_token_id: Optional[int] = None,
    vision_start_token_id: Optional[int] = None,
    fast_video_token_id: Optional[int] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Calculate the 3D rope index based on image and video's temporal, height and width in LLM.

    Explanation:
        Each embedding sequence contains vision embedding and text embedding or just contains text embedding.

        For pure text embedding sequence, the rotary position embedding has no difference with modern LLMs.
        Examples:
            input_ids: [T T T T T], here T is for text.
            temporal position_ids: [0, 1, 2, 3, 4]
            height position_ids: [0, 1, 2, 3, 4]
            width position_ids: [0, 1, 2, 3, 4]

        For vision and text embedding sequence, we calculate 3D rotary position embedding for vision part
        and 1D rotary position embedding for text part.
        Examples:
            Temporal (Time): 3 patches, representing different segments of the video in time.
            Height: 2 patches, dividing each frame vertically.
            Width: 2 patches, dividing each frame horizontally.
            We also have some important parameters:
            fps (Frames Per Second): The video's frame rate, set to 1. This means one frame is processed each second.
            tokens_per_second: This is a crucial parameter. It dictates how many "time-steps" or "temporal tokens" are conceptually packed into a one-second interval of the video. In this case, we have 25 tokens per second. So each second of the video will be represented with 25 separate time points. It essentially defines the temporal granularity.
            temporal_patch_size: The number of frames that compose one temporal patch. Here, it's 2 frames.
            interval: The step size for the temporal position IDs, calculated as tokens_per_second * temporal_patch_size / fps. In this case, 25 * 2 / 1 = 50. This means that each temporal patch will be have a difference of 50 in the temporal position IDs.
            input_ids: [V V V V V V V V V V V V T T T T T], here V is for vision.
            vision temporal position_ids: [0, 0, 0, 0, 50, 50, 50, 50, 100, 100, 100, 100]
            vision height position_ids: [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1]
            vision width position_ids: [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
            text temporal position_ids: [101, 102, 103, 104, 105]
            text height position_ids: [101, 102, 103, 104, 105]
            text width position_ids: [101, 102, 103, 104, 105]
            Here we calculate the text start position_ids as the max vision position_ids plus 1.

    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height and width of feature shape of each image in LLM.
        video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
            The temporal, height and width of feature shape of each video in LLM.
        second_per_grid_ts (`torch.Tensor` of shape `(num_videos)`, *optional*):
            The time interval (in seconds) for each grid along the temporal dimension in the 3D position IDs.
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

    Returns:
        position_ids (`torch.LongTensor` of shape `(3, batch_size, sequence_length)`)
        mrope_position_deltas (`torch.Tensor` of shape `(batch_size)`)
    """
    if input_ids is not None and (image_grid_thw is not None or video_grid_thw is not None):
        total_input_ids = input_ids
        if attention_mask is None:
            attention_mask = torch.ones_like(total_input_ids)
        position_ids = torch.ones(
            3,
            input_ids.shape[0],
            input_ids.shape[1],
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        image_index, video_index, fast_video_index = 0, 0, 0
        attention_mask = attention_mask.to(total_input_ids.device)
        for i, input_ids in enumerate(total_input_ids):
            input_ids = input_ids[attention_mask[i] == 1]

            if image_grid_thw is not None:
                image_nums = image_grid_thw.size(0) # 这里实际上是图片的数量
            else:
                image_nums = 0

            if video_grid_thw is not None:
                video_nums = video_grid_thw.size(0) # 这里实际上是slow_frame的数量
            else:
                video_nums = 0

            if fast_video_grid_thw is not None:
                fast_video_nums = fast_video_grid_thw.size(0) # 这里实际上是fast_frame的数量
            else:
                fast_video_nums = 0

            input_tokens = input_ids.tolist()
            llm_pos_ids_list: list = []
            st = 0
            remain_images, remain_videos_frames, remain_fast_videos_frames = image_nums, video_nums, fast_video_nums
            # remain_images, remain_videos = image_nums, video_grid_thw.size(0)//2
            for _ in range(image_nums + video_nums + fast_video_nums):

                if image_token_id in input_tokens and remain_images > 0:
                    ed_image = input_tokens.index(image_token_id, st)
                else:
                    ed_image = len(input_tokens) + 1

                if video_token_id in input_tokens and remain_videos_frames > 0:
                    ed_video = input_tokens.index(video_token_id, st)
                else:
                    ed_video = len(input_tokens) + 1

                if fast_video_token_id in input_tokens and remain_fast_videos_frames > 0:
                    ed_fast_video = input_tokens.index(fast_video_token_id, st)
                else:
                    ed_fast_video = len(input_tokens) + 1

                if ed_image < min(ed_video, ed_fast_video):
                    t, h, w = (
                        image_grid_thw[image_index][0],
                        image_grid_thw[image_index][1],
                        image_grid_thw[image_index][2],
                    )
                    image_index += 1
                    remain_images -= 1
                    ed = ed_image

                elif ed_video < min(ed_image, ed_fast_video):
                    t, h, w = (
                        video_grid_thw[video_index][0],
                        video_grid_thw[video_index][1],
                        video_grid_thw[video_index][2],
                    )
                    video_index += 1
                    remain_videos_frames -= 1
                    ed = ed_video

                elif ed_fast_video < min(ed_image, ed_video):
                    t, h, w = (
                        fast_video_grid_thw[fast_video_index][0],
                        fast_video_grid_thw[fast_video_index][1],
                        fast_video_grid_thw[fast_video_index][2],
                    )
                    fast_video_index += 1
                    remain_fast_videos_frames -= 1
                    ed = ed_fast_video


                llm_grid_t, llm_grid_h, llm_grid_w = (
                    t.item(),
                    h.item() // spatial_merge_size,
                    w.item() // spatial_merge_size,
                )
                text_len = ed - st

                st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                range_tensor = torch.arange(llm_grid_t).view(-1, 1)
                expanded_range = range_tensor.expand(-1, llm_grid_h * llm_grid_w)

                t_index = expanded_range.flatten()
                h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
                w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()
                llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index]) + text_len + st_idx)
                st = ed + llm_grid_t * llm_grid_h * llm_grid_w

            if st < len(input_tokens):
                st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                text_len = len(input_tokens) - st
                llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

            llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
            position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(position_ids.device)
        return position_ids
    else:
        if attention_mask is not None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(attention_mask.device)
        else:
            position_ids = (
                torch.arange(input_ids.shape[1], device=input_ids.device)
                .view(1, 1, -1)
                .expand(3, input_ids.shape[0], -1)
            )

        return position_ids, mrope_position_deltas


def get_rope_index(
        input_ids: torch.LongTensor,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        spatial_merge_size: Optional[int] = None,
        image_token_id: Optional[int] = None,
        video_token_id: Optional[int] = None,
        vision_start_token_id: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Calculate the 3D rope index based on image and video's temporal, height and width in LLM.

    Explanation:
        Each embedding sequence contains vision embedding and text embedding or just contains text embedding.

        For pure text embedding sequence, the rotary position embedding has no difference with mordern LLMs.
        Examples:
            input_ids: [T T T T T], here T is for text.
            temporal position_ids: [0, 1, 2, 3, 4]
            height position_ids: [0, 1, 2, 3, 4]
            width position_ids: [0, 1, 2, 3, 4]

        For vision and text embedding sequence, we calculate 3D rotary position embedding for vision part
        and 1D rotary position embeddin for text part.
        Examples:
            Assume we have a video input with 3 temporal patches, 2 height patches and 2 width patches.
            input_ids: [V V V V V V V V V V V V T T T T T], here V is for vision.
            vision temporal position_ids: [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]
            vision height position_ids: [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1]
            vision width position_ids: [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
            text temporal position_ids: [3, 4, 5, 6, 7]
            text height position_ids: [3, 4, 5, 6, 7]
            text width position_ids: [3, 4, 5, 6, 7]
            Here we calculate the text start position_ids as the max vision position_ids plus 1.

    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height and width of feature shape of each image in LLM.
        video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
            The temporal, height and width of feature shape of each video in LLM.
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

    Returns:
        position_ids (`torch.LongTensor` of shape `(3, batch_size, sequence_length)`)
        mrope_position_deltas (`torch.Tensor` of shape `(batch_size)`)
    """
    # spatial_merge_size = self.config.vision_config.spatial_merge_size
    # image_token_id = self.config.image_token_id
    # video_token_id = self.config.video_token_id
    # vision_start_token_id = self.config.vision_start_token_id
    mrope_position_deltas = []
    if input_ids is not None and (
            image_grid_thw is not None or video_grid_thw is not None):
        total_input_ids = input_ids
        if attention_mask is None:
            attention_mask = torch.ones_like(total_input_ids)
        position_ids = torch.ones(
            3,
            input_ids.shape[0],
            input_ids.shape[1],
            dtype=input_ids.dtype,
            device=input_ids.device)
        image_index, video_index = 0, 0
        for i, input_ids in enumerate(total_input_ids):
            input_ids = input_ids[attention_mask[i] == 1]
            image_nums, video_nums = 0, 0
            vision_start_indices = torch.argwhere(
                input_ids == vision_start_token_id).squeeze(1)
            vision_tokens = input_ids[vision_start_indices + 1]
            image_nums = (vision_tokens == image_token_id).sum()
            video_nums = (vision_tokens == video_token_id).sum()
            input_tokens = input_ids.tolist()
            llm_pos_ids_list: list = []
            st = 0
            remain_images, remain_videos = image_nums, video_nums
            for _ in range(image_nums + video_nums):
                if image_token_id in input_tokens and remain_images > 0:
                    ed_image = input_tokens.index(image_token_id, st)
                else:
                    ed_image = len(input_tokens) + 1
                if video_token_id in input_tokens and remain_videos > 0:
                    ed_video = input_tokens.index(video_token_id, st)
                else:
                    ed_video = len(input_tokens) + 1
                if ed_image < ed_video:
                    t, h, w = (
                        image_grid_thw[image_index][0],
                        image_grid_thw[image_index][1],
                        image_grid_thw[image_index][2],
                    )
                    image_index += 1
                    remain_images -= 1
                    ed = ed_image
                else:
                    t, h, w = (
                        video_grid_thw[video_index][0],
                        video_grid_thw[video_index][1],
                        video_grid_thw[video_index][2],
                    )
                    video_index += 1
                    remain_videos -= 1
                    ed = ed_video
                llm_grid_t, llm_grid_h, llm_grid_w = (
                    t.item(),
                    h.item() // spatial_merge_size,
                    w.item() // spatial_merge_size,
                )
                text_len = ed - st

                st_idx = llm_pos_ids_list[-1].max() + \
                    1 if len(llm_pos_ids_list) > 0 else 0
                llm_pos_ids_list.append(torch.arange(
                    text_len).view(1, -1).expand(3, -1) + st_idx)

                t_index = torch.arange(llm_grid_t).view(-1,
                                                        1).expand(-1, llm_grid_h * llm_grid_w).flatten()
                h_index = torch.arange(llm_grid_h).view(
                    1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
                w_index = torch.arange(llm_grid_w).view(
                    1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()
                llm_pos_ids_list.append(torch.stack(
                    [t_index, h_index, w_index]) + text_len + st_idx)
                st = ed + llm_grid_t * llm_grid_h * llm_grid_w

            if st < len(input_tokens):
                st_idx = llm_pos_ids_list[-1].max() + \
                    1 if len(llm_pos_ids_list) > 0 else 0
                text_len = len(input_tokens) - st
                llm_pos_ids_list.append(torch.arange(
                    text_len).view(1, -1).expand(3, -1) + st_idx)

            llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
            position_ids[..., i, attention_mask[i] ==
                         1] = llm_positions.to(position_ids.device)
            mrope_position_deltas.append(
                llm_positions.max() + 1 - len(total_input_ids[i]))
        mrope_position_deltas = torch.tensor(
            mrope_position_deltas,
            device=input_ids.device).unsqueeze(1)
        return position_ids, mrope_position_deltas
    else:
        if attention_mask is not None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            position_ids = position_ids.unsqueeze(
                0).expand(3, -1, -1).to(input_ids.device)
            max_position_ids = position_ids.max(0, keepdim=False)[
                0].max(-1, keepdim=True)[0]
            mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
        else:
            position_ids = (
                torch.arange(input_ids.shape[1], device=input_ids.device)
                .view(1, 1, -1)
                .expand(3, input_ids.shape[0], -1)
            )
            mrope_position_deltas = torch.zeros(
                [input_ids.shape[0], 1],
                device=input_ids.device,
                dtype=input_ids.dtype,
            )

        return position_ids, mrope_position_deltas


class Reader():
    @staticmethod
    def torchcodec(video_file: Union[str, bytes], use_gpu: bool = True):
        from torchcodec.decoders import VideoDecoder
        import torchvision

        if isinstance(video_file, str):
            return VideoDecoder(video_file, num_ffmpeg_threads=int(os.environ.get('TORCHCODEC_NUM_THREADS', 8)))
        elif isinstance(video_file, bytes):
            return torchvision.io.VideoReader(video_file, "video")

    @staticmethod
    def torchvision(video_file: Union[str, bytes], use_gpu: bool = True):
        from packaging import version
        import torchvision
        if isinstance(video_file, str):
            if version.parse(torchvision.__version__) < version.parse("0.19.0"):
                if "http://" in video_file or "https://" in video_file:
                    warnings.warn("torchvision < 0.19.0 does not support http/https video path, please upgrade to 0.19.0.")
                if "file://" in video_file:
                    video_file = video_file[7:]
            video, audio, info = torchvision.io.read_video(
                video_file,
                start_pts=0.0,
                end_pts=None,
                pts_unit="sec",
                output_format="TCHW",
            )
            return video
        elif isinstance(video_file, bytes):
            return torchvision.io.VideoReader(video_file, "video")

    @staticmethod
    def decord(video_file: Union[str, bytes], use_gpu: bool = True):
        import decord
        if isinstance(video_file, bytes):
            video_path = ""
            fp = py_io.BytesIO(video_file)
            return decord.VideoReader(fp)
        else:
            video_path = video_file
            return decord.VideoReader(video_path)

    @staticmethod
    def is_torchcodec_available() -> bool:
        import importlib.util

        return importlib.util.find_spec("torchcodec") is not None

    @staticmethod
    def is_decord_available() -> bool:
        import importlib.util

        return importlib.util.find_spec("decord") is not None

    @staticmethod
    def get_video_reader_backend():
        FORCE_KEYE_VIDEO_READER = os.getenv("FORCE_KEYE_VIDEO_READER", None)
        if FORCE_KEYE_VIDEO_READER is not None:
            video_reader_backend = FORCE_KEYE_VIDEO_READER
        elif Reader.is_torchcodec_available():
            video_reader_backend = "torchcodec"
        elif Reader.is_decord_available():
            video_reader_backend = "decord"
        else:
            video_reader_backend = "torchvision"
        return video_reader_backend

    @staticmethod
    def idle_func(video_file, use_gpu: bool = True):
        from sglang.srt.utils.common import VideoData
        if isinstance(video_file, VideoData):
            # VideoData carries URL + optional preprocess_kwargs
            # Extract the URL so fetch_video can use the video reader backend
            result = {"video": video_file.url}
            if video_file.preprocess_kwargs:
                result.update(video_file.preprocess_kwargs)
            return result
        return {
            "video": video_file
        }

    @staticmethod
    def get_video_reader_backend_func():
        return Reader.idle_func


class PreProcessor():
    @staticmethod
    def get_video_preprocessor_backend_func():
        async def preprocessor_func(vr, ele: dict):
            if isinstance(vr, dict) and ("frames" in vr and "processor_kwargs" in vr):
                return vr["frames"], vr["processor_kwargs"]
            if isinstance(vr, dict):
                ele.update(vr)
                return fetch_video(ele)
            # Fallback: set vr as video value and let fetch_video handle it
            ele["video"] = vr
            return fetch_video(ele)
        return preprocessor_func

class Rope():
    @staticmethod
    def get_rope_backend_func():
        return Rope.get_rope_index

    @staticmethod
    def get_rope_index(**kwargs):
        kwargs.pop("model_type")
        return get_rope_index(**kwargs)
