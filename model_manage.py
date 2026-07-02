import gc
import logging
import threading
from collections import OrderedDict
from typing import Optional

import torch

logger = logging.getLogger(__name__)

if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

torch_device = torch.device(device)

MAX_GPU_LOADED_MODELS = 2  # lets us do current model + one prefetched model

lock = threading.RLock()
cache = {} # name -> model object
loaders = {} # name -> function that builds and returns model on CPU or device
on_gpu: "OrderedDict[str, None]" = OrderedDict() # models on the GPU, oldest first
prefetch_threads: dict[str, threading.Thread] = {}


def register_loader(name: str, loader_fn):
    """Register how to build a model. Loader should return the model already in eval mode"""
    loaders[name] = loader_fn


def preload_all():
    """
    Build all models on CPU to start with, called at app startup after register_loader() calls
    """
    for name, loader_fn in loaders.items():
        if name in cache:
            continue
        logger.info(f"Preloading '{name}' on cpu")
        model = loader_fn()
        if hasattr(model, "to"):
            model.to("cpu")
        cache[name] = model
    logger.info(f"Preloaded {len(cache)} models on cpu: {list(cache)}")


def ensure_model_built(name: str):
    if name not in cache:
        logger.info(f"Loading '{name}' model")
        model = loaders[name]()
        if hasattr(model, "to"):
            model.to("cpu")
        cache[name] = model


def mark_used(name: str):
    """Mark as most-recently-used."""
    on_gpu[name] = None
    on_gpu.move_to_end(name)


def make_room_for(incoming: str, protect: set[str]):
    """Evict model on GPU that hasn't been used for longest time until there's room for the incoming one"""
    if incoming in on_gpu:
        return
    while len(on_gpu) >= MAX_GPU_LOADED_MODELS:
        for candidate in list(on_gpu.keys()):
            if candidate in protect:
                continue
            on_gpu.pop(candidate, None)
            model = cache.get(candidate)
            if model is not None and hasattr(model, "to"):
                model.to("cpu")
            logger.info(f"Had to evict {candidate} to cpu to make room for {incoming}")
            break
        else:
            break  # everything resident is protected, can't evict further
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def get_model(name: str, next_name: Optional[str] = None):
    """
    Returns the model with given name index ready to use. If a next_name is give, background thread is started that
    moves model onto device too, so ideally it is ready to be used by the time the caller finishes with the first model.
    """
    if name not in loaders:
        raise KeyError(f"No loader registered for '{name}'")

    # If a prior prefetch is already bringing this model up, wait for it
    thread = prefetch_threads.get(name)
    if thread is not None and thread.is_alive():
        thread.join()

    with lock:
        ensure_model_built(name)
        protect = {name} | ({next_name} if next_name else set())
        make_room_for(name, protect=protect)
        model = cache[name]
        if hasattr(model, "to") and next(model.parameters()).device != torch_device:
            logger.info(f"Moving '{name}' to {device}")
            model.to(torch_device)
        mark_used(name)

    if next_name and next_name in loaders and next_name != name:
        prefetch(next_name, protect={name, next_name})

    return cache[name]


def prefetch(name: str, protect: set[str]):
    existing = prefetch_threads.get(name)
    if existing is not None and existing.is_alive():
        return  # already being prefetched

    def _worker():
        with lock:
            ensure_model_built(name)
            make_room_for(name, protect=protect)
            model = cache[name]
            if hasattr(model, "to") and next(model.parameters()).device != torch_device:
                logger.info(f"Prefetching '{name}' onto {device}")
                model.to(torch_device)
            mark_used(name)

    thread = threading.Thread(target=_worker, daemon=True, name=f"prefetch-{name}")
    prefetch_threads[name] = thread
    thread.start()
