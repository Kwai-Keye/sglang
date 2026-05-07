import asyncio
import collections
import math
import re
import time
from typing import Dict, List, Tuple, Union

import cv2
import numpy as np
import copy
import torch
from einops import rearrange
from PIL import Image

from sglang.srt.layers.rotary_embedding import MRotaryEmbedding
from sglang.srt.models.keye_qwen3 import KeyeVL1_5ForConditionalGeneration
from sglang.srt.models.keye_topk_mask import KeyeVL2ForConditionalGeneration
from sglang.srt.models.keye_topk_mask_moe import KeyeVL2MoeForConditionalGeneration
from sglang.srt.multimodal.processors.base_processor import (
    BaseMultimodalProcessor as SGLangBaseProcessor,
)
from sglang.srt.multimodal.processors.base_processor import MultimodalSpecialTokens
from sglang.srt.utils import get_int_env_var
from sglang.srt.utils.common import ImageData, VideoData
from sglang.utils import logger

# NOTE(wanghao44): same as preprocessor_config.json in keye-slowfast
PATCH_SIZE = 14
MERGE_SIZE = 2
IMAGE_FACTOR = get_int_env_var("KEYE_IMAGE_FACTOR", default=28)


# TODO: need to align with keye-vl-utils
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
VIDEO_MIN_TOKENS = get_int_env_var("VIDEO_MIN_TOKENS", default=48)
# max tokens per video frame
VIDEO_MAX_TOKENS = get_int_env_var("VIDEO_MAX_TOKENS", default=768)
# min pixels per video frame
VIDEO_MIN_PIXELS = (
    VIDEO_MIN_TOKENS * IMAGE_FACTOR * IMAGE_FACTOR
)  # 32 * 28 * 28 = 25,088
# max pixels per video frame
VIDEO_MAX_PIXELS = (
    VIDEO_MAX_TOKENS * IMAGE_FACTOR * IMAGE_FACTOR
)  # 768 * 28 * 28 = 602,112
VIDEO_TOTAL_MAX_TOKENS = get_int_env_var("VIDEO_TOTAL_MAX_TOKENS", default=65536)
# max total pixels per video
VIDEO_TOTAL_PIXELS = (
    VIDEO_TOTAL_MAX_TOKENS * IMAGE_FACTOR * IMAGE_FACTOR
)  # 65,536 * 28 * 28 = 51,380,224
VIDEO_TOTAL_PIXELS = get_int_env_var("VIDEO_TOTAL_PIXELS", default=VIDEO_TOTAL_PIXELS)
# default fps
FPS = 2.0

FAST_TOKEN_RATIO = 0.3

MIN_FRAME_SIMILARITY = 0.9


def round_by_factor(number: int, factor: int) -> int:
    """Returns the closest integer to 'number' that is divisible by 'factor'."""
    return round(number / factor) * factor


def ceil_by_factor(number: int, factor: int) -> int:
    """Returns the smallest integer greater than or equal to 'number' that is divisible by 'factor'."""
    return math.ceil(number / factor) * factor


def floor_by_factor(number: int, factor: int) -> int:
    """Returns the largest integer less than or equal to 'number' that is divisible by 'factor'."""
    return math.floor(number / factor) * factor


# NOTE(wanghao44): copied from keye-vl-utils/vision_process.py
def smart_resize(
    height: int,
    width: int,
    factor: int = IMAGE_FACTOR,
    min_pixels: int = MIN_PIXELS,
    max_pixels: int = MAX_PIXELS,
) -> Tuple[int, int]:
    """
    Rescales the image so that the following conditions are met:

    1. Both dimensions (height and width) are divisible by 'factor'.

    2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].

    3. The aspect ratio of the image is maintained as closely as possible.
    """

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


def resize_image(
    image,
    size_factor: int = IMAGE_FACTOR,
    min_pixels: int = MIN_PIXELS,
    max_pixels: int = MAX_PIXELS,
) -> Image.Image:
    width, height = image.size
    resized_height, resized_width = smart_resize(
        height,
        width,
        factor=size_factor,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )
    image = image.resize((resized_width, resized_height))
    return image


async def resize_image_async(
    image,
    size_factor: int = IMAGE_FACTOR,
    min_pixels: int = MIN_PIXELS,
    max_pixels: int = MAX_PIXELS,
):
    return resize_image(
        image,
        size_factor=size_factor,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )


def smart_nframes(ele: dict, total_frames: int, video_fps: int | float) -> int:
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
    fps = ele.get("fps", FPS)
    fps = min(fps, video_fps)
    max_frames = int(
        ele.get("video_total_pixels", VIDEO_TOTAL_PIXELS)
        / ele.get("min_pixels", VIDEO_MIN_PIXELS)
    )
    max_frames = min(ele.get("max_frames", max_frames), max_frames)
    fps_nframes = int(total_frames / video_fps * fps)
    nframes = min(fps_nframes, max_frames)
    return nframes


def get_frame_sim(
    frame1,
    frame2,
    patch_size: int = 28,
    threshold: float = 0.7,
    epsilon: float = 1e-8,
):
    assert (
        frame1.dim() == 3 and frame2.dim() == 3
    ), "frame1 and frame2 must be a 3D tensor [C, H, W]"

    def to_numpy_cvt(tensor):
        tensor = tensor.cpu().permute(1, 2, 0).numpy()
        if tensor.dtype == np.float32 or tensor.dtype == np.float64:
            tensor = (tensor).astype(np.uint8)
        return cv2.cvtColor(tensor, cv2.COLOR_RGB2HSV)

    frame1_hsv = to_numpy_cvt(frame1)
    frame2_hsv = to_numpy_cvt(frame2)

    frame1_tensor = (
        torch.from_numpy(frame1_hsv).permute(2, 0, 1).to(frame1.device).float()
    )
    frame2_tensor = (
        torch.from_numpy(frame2_hsv).permute(2, 0, 1).to(frame2.device).float()
    )

    patch1 = rearrange(
        frame1_tensor, "c (h p1) (w p2) -> h w (c p1 p2)", p1=patch_size, p2=patch_size
    ).float()
    patch2 = rearrange(
        frame2_tensor, "c (h p1) (w p2) -> h w (c p1 p2)", p1=patch_size, p2=patch_size
    ).float()

    norm1 = torch.norm(patch1, p=2, dim=-1, keepdim=True) + epsilon
    norm2 = torch.norm(patch2, p=2, dim=-1, keepdim=True) + epsilon

    normalized1 = patch1 / norm1
    normalized2 = patch2 / norm2
    cos_sim = (normalized1 * normalized2).sum(dim=-1)

    zero_vector_mask = (norm1.squeeze() < 0.01) & (norm2.squeeze() < 0.01)

    similar = torch.ones_like(cos_sim)

    non_zero_mask = ~zero_vector_mask
    similar[non_zero_mask] = (cos_sim[non_zero_mask] > threshold).float()

    return similar[non_zero_mask].float().mean().item()


def extract_slow_fast_frames(
    frames,
    threshold=MIN_FRAME_SIMILARITY,
):
    def _extract_slow_indices(frames):
        assert frames.dim() == 4, "frames must be a 4D tensor [N, C, H, W]"

        slow_indices = [0]
        last_key_frame = frames[0]
        for i in range(1, frames.size(0)):
            current_frame = frames[i]
            sim = get_frame_sim(last_key_frame, current_frame)

            if sim < threshold:
                slow_indices.append(i)
                last_key_frame = current_frame

        return slow_indices

    _, _, height, width = frames.shape
    resized_height, resized_width = smart_resize(
        height,
        width,
        factor=IMAGE_FACTOR,
        min_pixels=VIDEO_MIN_PIXELS,
        max_pixels=256 * IMAGE_FACTOR * IMAGE_FACTOR,
    )

    resized_frames = torch.nn.functional.interpolate(
        frames,
        [resized_height, resized_width],
        mode="bilinear",
        antialias=True,
    ).float()

    slow_indices = _extract_slow_indices(resized_frames)
    frame_types = torch.ones(size=(frames.size(0),), dtype=torch.int32)
    frame_types[slow_indices] = 0

    return frame_types

def extract_all_slow_frames(
    frames,
    threshold=MIN_FRAME_SIMILARITY,
):
    return torch.zeros(size=(frames.size(0), ), dtype=torch.int32)


def calculate_video_frame_range(
    ele: dict,
    total_frames: int,
    video_fps: float,
) -> tuple[int, int, int]:
    """
    Calculate the start and end frame indices based on the given time range.

    Args:
        ele (dict): A dictionary containing optional 'video_start' and 'video_end' keys (in seconds).
        total_frames (int): Total number of frames in the video.
        video_fps (float): Frames per second of the video.

    Returns:
        tuple: A tuple containing (start_frame, end_frame, frame_count).

    Raises:
        ValueError: If input parameters are invalid or the time range is inconsistent.
    """
    # Validate essential parameters
    if video_fps <= 0:
        raise ValueError("video_fps must be a positive number")
    if total_frames <= 0:
        raise ValueError("total_frames must be a positive integer")

    # Get start and end time in seconds
    video_start = ele.get("video_start", None)
    video_end = ele.get("video_end", None)
    if video_start is None and video_end is None:
        return 0, total_frames - 1, total_frames

    max_duration = total_frames / video_fps
    # Process start frame
    if video_start is not None:
        video_start_clamped = max(0.0, min(video_start, max_duration))
        start_frame = math.ceil(video_start_clamped * video_fps)
    else:
        start_frame = 0
    # Process end frame
    if video_end is not None:
        video_end_clamped = max(0.0, min(video_end, max_duration))
        end_frame = math.floor(video_end_clamped * video_fps)
        end_frame = min(end_frame, total_frames - 1)
    else:
        end_frame = total_frames - 1

    # Validate frame order
    if start_frame >= end_frame:
        raise ValueError(
            f"Invalid time range: Start frame {start_frame} (at {video_start_clamped if video_start is not None else 0}s) "
            f"exceeds end frame {end_frame} (at {video_end_clamped if video_end is not None else max_duration}s). "
            f"Video duration: {max_duration:.2f}s ({total_frames} frames @ {video_fps}fps)"
        )

    logger.info(
        f"calculate video frame range: {start_frame=}, {end_frame=}, {total_frames=} from {video_start=}, {video_end=}, {video_fps=:.3f}"
    )
    return start_frame, end_frame, end_frame - start_frame + 1


def _read_video_decord(
    vr,
    ele: dict,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """read video using decord.VideoReader

    Args:
        vr: decord.VideoReader
        ele (dict): a dict contains the configuration of video.
        support keys:
            - video_start: the start time of video.
            - video_end: the end time of video.
    Returns:
        torch.Tensor: the video tensor with shape (T, C, H, W).
    """
    # TODO: support start_pts and end_pts
    if "video_start" in ele or "video_end" in ele:
        raise NotImplementedError(
            "not support start_pts and end_pts in decord for now."
        )
    nframes, video_fps = len(vr), vr.get_avg_fps()
    # timestamp start from 0.0
    timestamps = torch.arange(nframes, dtype=torch.float32) / float(video_fps)

    final_nframes = smart_nframes(ele, total_frames=nframes, video_fps=video_fps)

    indices = torch.linspace(0, nframes - 1, final_nframes).round().long()
    frames_hwc = vr.get_batch(indices.tolist())

    # Ensure torch tensor and convert to CHW
    if not isinstance(frames_hwc, torch.Tensor):
        if hasattr(frames_hwc, "to_dlpack"):
            frames_hwc = torch.utils.dlpack.from_dlpack(frames_hwc.to_dlpack())
        elif hasattr(frames_hwc, "asnumpy"):
            frames_hwc = torch.from_numpy(frames_hwc.asnumpy())
        else:
            frames_hwc = torch.from_numpy(np.array(frames_hwc))
    frames = frames_hwc.permute(0, 3, 1, 2)
    logger.debug(f"Decord: {nframes=}, {video_fps=}")
    timestamps = timestamps[indices]

    ##### extract key frames start ######
    threshold = ele.get("min_frame_similarity", MIN_FRAME_SIMILARITY)

    if ele.get("keye_all_slow_frames", False):
        frame_types = torch.zeros(size=(frames.size(0), ), dtype=torch.int32)
    else:
        frame_types = extract_slow_fast_frames(frames, threshold)

    ##### extract key frames end ######

    return frames, timestamps, frame_types


async def preprocess_video(vr, ele: dict) -> Tuple[torch.Tensor, dict]:
    if isinstance(vr, dict):
        assert "frames" in vr and "processor_kwargs" in vr, "vr must contain 'frames' and 'processor_kwargs'"
        return vr["frames"], vr["processor_kwargs"]

    frames, timestamps, frame_types = _read_video_decord(vr, ele)
    img_factor = ele.get("factor", IMAGE_FACTOR)
    nframes = len(frame_types)
    fast_nframes = int(sum(frame_types))
    slow_nframes = nframes - fast_nframes

    min_pixels = max(int(ele.get("min_pixels", VIDEO_MIN_PIXELS)), VIDEO_MIN_PIXELS)
    min_tokens = int(min_pixels / img_factor / img_factor)
    left = min_pixels / img_factor / img_factor
    right = ele.get("max_pixels", VIDEO_MAX_PIXELS) / img_factor / img_factor

    def _estimate_total_pixels(tokens_per_frame):
        return (
            slow_nframes * tokens_per_frame * img_factor * img_factor
            + fast_nframes
            * max(int(FAST_TOKEN_RATIO * tokens_per_frame), min_tokens)
            * img_factor
            * img_factor
        )

    while left < right:
        mid = int(left + right) // 2
        if _estimate_total_pixels(mid) > ele.get(
            "video_total_pixels", VIDEO_TOTAL_PIXELS
        ):
            right = mid
        else:
            left = mid + 1
    slow_max_pixels = left * img_factor * img_factor

    _, _, height, width = frames.shape

    resized_height, resized_width = smart_resize(
        height,
        width,
        factor=img_factor,
        min_pixels=min_pixels,
        max_pixels=slow_max_pixels,
    )
    slow_num_tokens = resized_height * resized_width / img_factor / img_factor
    fast_max_pixels = max(
        int(slow_num_tokens * FAST_TOKEN_RATIO) * img_factor * img_factor,
        VIDEO_MIN_PIXELS,
    )

    fast_resized_height, fast_resized_width = smart_resize(
        height,
        width,
        factor=img_factor,
        min_pixels=min_pixels,
        max_pixels=fast_max_pixels,
    )
    fast_num_tokens = fast_resized_height * fast_resized_width / img_factor / img_factor
    logger.debug(
        f"fetch_video: {nframes=}, {slow_nframes=}, {fast_nframes=}, {slow_num_tokens=}, "
        f"{fast_num_tokens=}, {min_pixels=}, {resized_height=}, {resized_width=}, "
        f"{fast_resized_height=}, {fast_resized_width}"
    )
    processor_kwargs = {
        "height": resized_height,
        "width": resized_width,
        "fast_height": fast_resized_height,
        "fast_width": fast_resized_width,
    }
    if timestamps is not None:
        processor_kwargs["timestamps"] = timestamps
    if frame_types is not None:
        processor_kwargs["frame_types"] = frame_types
    return frames, processor_kwargs


class KeyeImageProcessor(SGLangBaseProcessor):
    models = [
        KeyeVL1_5ForConditionalGeneration,
    ]
    gpu_image_decode = False

    def __init__(self, hf_config, server_args, _processor, *args, **kwargs):
        super().__init__(hf_config, server_args, _processor, *args, **kwargs)
        # The single, pre-expanded image token.
        self.IMAGE_TOKEN = "<|vision_start|><|image_pad|><|vision_end|>"
        # The regex that matches expanded image tokens.
        self.IMAGE_TOKEN_REGEX = re.compile(
            r"<\|vision_start\|>(?:<\|image_pad\|>)+<\|vision_end\|>"
        )
        self.IM_START_TOKEN_ID = hf_config.vision_start_token_id
        self.IM_END_TOKEN_ID = hf_config.vision_end_token_id
        self.IM_TOKEN_ID = hf_config.image_token_id
        self.VIDEO_TOKEN_ID = hf_config.video_token_id
        self.FAST_VIDEO_TOKEN_ID = hf_config.fast_video_token_id
        self.vision_start_token_id = hf_config.vision_start_token_id
        self.vision_end_token_id = hf_config.vision_end_token_id

        self.audio_token_id = getattr(hf_config, "audio_token_id", None)

        self.mm_tokens = MultimodalSpecialTokens(
            image_token="<|vision_start|><|image_pad|><|vision_end|>",
            image_token_id=hf_config.image_token_id,
            image_token_regex=re.compile(
                r"<\|vision_start\|>(?:<\|image_pad\|>)+<\|vision_end\|>"
            ),
            video_token_id=hf_config.video_token_id,
            audio_token_id=self.audio_token_id,
        ).build(_processor)

    async def process_mm_data_async(
        self,
        image_data: List[Union[str, bytes, Dict]],
        input_text,
        request_obj,
        *args,
        **kwargs,
    ):
        base_output = self.load_mm_data(
            prompt=input_text,
            image_data=image_data,
            video_data=request_obj.video_data,
            audio_data=request_obj.audio_data,
            multimodal_tokens=self.mm_tokens,
        )

        # keye-specific: resize images if they are raw Image objects
        if base_output.images and isinstance(base_output.images[0], Image.Image):
            # Build image config from per-image preprocess_kwargs in ImageData
            # Note: image_data contains ImageData objects with preprocess_kwargs,
            # while base_output.images contains loaded Image.Image objects (metadata lost)
            image_cfg = {}
            for img_item in image_data:
                if isinstance(img_item, ImageData) and img_item.preprocess_kwargs:
                    image_cfg = img_item.preprocess_kwargs
                    break
            
            size_factor = image_cfg.get("factor", IMAGE_FACTOR)
            min_pixels_cfg = image_cfg.get("min_pixels", MIN_PIXELS)
            max_pixels_cfg = image_cfg.get("max_pixels", MAX_PIXELS)

            resize_tasks = [
                resize_image_async(
                    image,
                    size_factor=size_factor,
                    min_pixels=min_pixels_cfg,
                    max_pixels=max_pixels_cfg,
                )
                for image in base_output.images
            ]
            base_output.images = await asyncio.gather(*resize_tasks)

        if base_output.videos:
            # Build video config from per-video preprocess_kwargs in VideoData
            # Note: request_obj.video_data contains VideoData objects with preprocess_kwargs,
            # while base_output.videos contains loaded video data (metadata lost)
            video_cfg = {"keye_all_slow_frames": kwargs.get("keye_all_slow_frames", False)}
            
            # Merge per-video preprocess_kwargs from request_obj.video_data (VideoData objects)
            for video_item in (request_obj.video_data or []):
                if isinstance(video_item, VideoData) and video_item.preprocess_kwargs:
                    video_cfg.update(video_item.preprocess_kwargs)
            
            processed = [
                await preprocess_video(video, copy.deepcopy(video_cfg)) for video in base_output.videos
            ]
            frames_list, kwargs_list = zip(*processed) if processed else ([], [])
            base_output.videos = list(frames_list)
            videos_kwargs = collections.defaultdict(list)
            for kw in kwargs_list:
                for k, v in kw.items():
                    if v is None:
                        continue
                    videos_kwargs[k].append(v)

        if base_output.videos and videos_kwargs:
            mm_items, input_ids, ret = self.process_and_combine_mm_data(
                base_output,
                self.mm_tokens,
                videos_kwargs=videos_kwargs,
            )
        else:
            mm_items, input_ids, ret = self.process_and_combine_mm_data(
                base_output,
                self.mm_tokens,
            )

        input_ids = input_ids.flatten()
        get_rope_index_func = self._rope_func if hasattr(self, "_rope_func") else MRotaryEmbedding.get_rope_index_keye
        mrope_positions, mrope_position_delta = get_rope_index_func(
            spatial_merge_size=self.hf_config.vision_config.spatial_merge_size,
            image_token_id=self.IM_TOKEN_ID,
            video_token_id=self.VIDEO_TOKEN_ID,
            vision_start_token_id=self.vision_start_token_id,
            model_type=self.hf_config.model_type,
            input_ids=input_ids.unsqueeze(0),
            image_grid_thw=getattr(ret, "image_grid_thw", None),
            video_grid_thw=getattr(ret, "video_grid_thw", None),
            attention_mask=getattr(ret, "attention_mask", None),
        )
        mrope_positions = mrope_positions.squeeze(1)

        from sglang.srt.managers.schedule_batch import MultimodalProcessorOutput
        return MultimodalProcessorOutput.from_dict({
            "input_ids": input_ids.tolist(),
            "mm_items": mm_items,
            "im_start_id": self.IM_START_TOKEN_ID,
            "im_end_id": self.IM_END_TOKEN_ID,
            "im_token_id": self.IM_TOKEN_ID,
            "video_token_id": self.VIDEO_TOKEN_ID,
            "audio_token_id": self.mm_tokens.audio_token_id,
            "mrope_positions": mrope_positions,
            "mrope_position_delta": mrope_position_delta,
        })


class KeyeImageAllSlowProcessor(KeyeImageProcessor):
    models = [
        KeyeVL2ForConditionalGeneration,
        KeyeVL2MoeForConditionalGeneration,
    ]
    gpu_image_decode = False

    def __init__(self, hf_config, server_args, _processor, *args, **kwargs):
        super().__init__(hf_config, server_args, _processor, *args, **kwargs)

        if self.hf_config.model_type == "KeyeVL2":
            from .keye_adaptive_video import PreProcessor, Reader, Rope

            import sglang.srt.multimodal.processors.base_processor as base_processor
            base_processor.load_video = Reader.get_video_reader_backend_func()

            globals()["preprocess_video"] = PreProcessor.get_video_preprocessor_backend_func()

            logger.info(f"✅ Patched load_video and preprocess_video for {hf_config.model_type}")

    async def process_mm_data_async(
        self,
        image_data: List[Union[str, bytes, Dict]],
        input_text,
        request_obj,
        *args,
        **kwargs,
    ):
        kwargs["keye_all_slow_frames"] = True
        return await super().process_mm_data_async(image_data, input_text, request_obj, *args, **kwargs)
