import asyncio
import multiprocessing as mp
import os
import signal
import threading
import traceback
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

from sglang.srt.environ import envs

# Ensure spawn mode to prevent daemon process issues with child processes
# This must be set before creating any Process objects
try:
    mp.set_start_method('spawn', force=False)
except RuntimeError:
    # Start method already set, check if it's spawn
    if mp.get_start_method() != 'spawn':
        import logging
        logging.warning(
            f"Multiprocessing start method is '{mp.get_start_method()}', "
            "which may cause issues with nested processes. Consider using 'spawn'."
        )


@dataclass
class MMProcessRequest:
    request_id: str
    image_data: Optional[List[Union[str, bytes]]] = None
    audio_data: Optional[List[Union[str, bytes]]] = None
    input_text: Optional[str] = None
    max_req_input_len: Optional[int] = None
    kwargs: Optional[Dict[str, Any]] = None


@dataclass
class MMProcessResponse:
    request_id: str
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


def worker_loop(config: Dict[str, Any], task_q: mp.Queue, result_q: mp.Queue):
    """Worker process that monitors parent process and exits if parent dies."""
    parent_pid = os.getppid()

    def check_parent_alive():
        """Check if parent process is still alive."""
        try:
            # Send signal 0 to check if process exists (doesn't actually send signal)
            os.kill(parent_pid, 0)
            return True
        except (OSError, ProcessLookupError):
            return False

    from sglang.srt.managers.multimodal_processor import (
        get_mm_processor,
        import_processors,
    )
    from sglang.srt.server_args import set_global_server_args_for_scheduler

    # Worker is a separate process (mp.Process) — the parent's global server_args
    # singleton is not inherited. base_processor.process_mm_data calls
    # get_global_server_args() (added in upstream 0.5.11), so we must set it here.
    set_global_server_args_for_scheduler(config["server_args"])

    import_processors("sglang.srt.multimodal.processors")
    if mm_process_pkg := envs.SGLANG_EXTERNAL_MM_PROCESSOR_PACKAGE.get():
        import_processors(mm_process_pkg, overwrite=True)
    mm_processor = get_mm_processor(
        config["hf_config"],
        config["server_args"],
        config["processor"],
        config["transport_mode"],
    )

    while True:
        # Check if parent is still alive before processing
        if not check_parent_alive():
            # Parent process died, exit immediately
            break

        try:
            # Use timeout to periodically check parent process
            req: MMProcessRequest = task_q.get(timeout=1.0)
        except:
            # Timeout or queue closed, check parent and continue
            continue

        if req is None:
            break

        try:
            result = asyncio.run(
                mm_processor.process_mm_data_async(
                    image_data=req.image_data,
                    audio_data=req.audio_data,
                    input_text=req.input_text,
                    max_req_input_len=req.max_req_input_len,
                    **(req.kwargs or {}),
                )
            )
            resp = MMProcessResponse(request_id=req.request_id, result=result)
        except Exception as e:
            resp = MMProcessResponse(
                request_id=req.request_id,
                error=f"Processing failed: {e}\n{traceback.format_exc()}",
            )

        # Check parent before putting result
        if check_parent_alive():
            try:
                result_q.put(resp)
            except:
                pass  # Queue might be closed


class MultimodalProcessorPool:
    def __init__(self, hf_config, server_args, processor, transport_mode):
        self.config = dict(
            hf_config=hf_config,
            server_args=server_args,
            processor=processor,
            transport_mode=transport_mode,
        )
        self.num_workers = server_args.mm_processor_workers

        self.task_q = mp.Queue()
        self.result_q = mp.Queue()

        # active[request_id] = (future, loop)
        self.active: Dict[str, Tuple[asyncio.Future, asyncio.AbstractEventLoop]] = {}
        self.req_counter = 0

        # Hide all GPUs before starting workers so each spawned child inherits
        # CUDA_VISIBLE_DEVICES="" from the OS environment before any Python
        # code (including module-level imports) runs.  Setting it inside
        # worker_loop is too late — CUDA may already be initialized by the
        # spawn bootstrap.  All GPU work (ViT forward) happens in the
        # scheduler/model-runner processes, not here.
        _orig_cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        self.workers = [
            mp.Process(
                target=worker_loop,
                args=(self.config, self.task_q, self.result_q),
                daemon=False,
            )
            for _ in range(self.num_workers)
        ]
        for w in self.workers:
            w.start()
        if _orig_cvd is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = _orig_cvd

        self._listener_thread = threading.Thread(
            target=self._result_listener, daemon=True
        )
        self._listener_thread.start()

    async def process_mm_data_async(
        self,
        image_data=None,
        audio_data=None,
        input_text=None,
        max_req_input_len=None,
        **kwargs,
    ) -> Dict[str, Any]:
        loop = asyncio.get_running_loop()
        req_id = str(self.req_counter)
        self.req_counter += 1

        fut = loop.create_future()
        self.active[req_id] = (fut, loop)

        try:
            self.task_q.put_nowait(
                MMProcessRequest(
                    request_id=req_id,
                    image_data=image_data,
                    audio_data=audio_data,
                    input_text=input_text,
                    max_req_input_len=max_req_input_len,
                    kwargs=kwargs,
                )
            )
        except Exception as e:
            self.active.pop(req_id, None)
            fut.set_exception(RuntimeError(f"Task queue put failed: {e}"))
            return await fut

        return await fut

    def _result_listener(self):
        while True:
            resp = self.result_q.get()
            if resp is None:
                break

            pair = self.active.pop(resp.request_id, None)
            if not pair:
                continue

            fut, loop = pair
            if fut.done():
                continue

            if resp.error:
                loop.call_soon_threadsafe(fut.set_exception, Exception(resp.error))
            else:
                loop.call_soon_threadsafe(fut.set_result, resp.result)

    def shutdown(self):
        for _ in range(self.num_workers):
            self.task_q.put(None)
        for w in self.workers:
            w.join(timeout=5)
            if w.is_alive():
                w.terminate()

        self.result_q.put(None)
        if self._listener_thread.is_alive():
            self._listener_thread.join(timeout=2)

        for _, (fut, loop) in list(self.active.items()):
            if not fut.done():
                loop.call_soon_threadsafe(
                    fut.set_exception, RuntimeError("Pool shutdown")
                )
        self.active.clear()
