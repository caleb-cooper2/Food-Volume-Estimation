import gc
import logging

import torch

logger = logging.getLogger(__name__)

if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

torch_device = torch.device(device)

cache = {} # name -> model object
loaders = {} # name -> function that builds and returns model on CPU or device

def register_loader(name: str, loader_fn):
    """Register how to build a model. Loader should return the model already in eval mode"""
    loaders[name] = loader_fn

def get_model(name: str):
    """Returns model, loading it and moving to GPU if needed"""
    if name not in loaders:
        raise KeyError(f"No loader found registered for {name}")

    if name not in cache:
        logger.info(f"Loading '{name}' model")
        cache[name] = loaders[name]()

    model = cache[name]
    if hasattr(model, "to") and next(model.parameters()).device != torch_device:
        logger.info(f"Moving {name} to {device}")
        model.to(torch_device)

    return model

def release_model(name: str):
    """Move a loaded model off GPU to free up VRAM"""
    if name in cache:
        model = cache[name]
        if hasattr(model, "to"):
            model.to("cpu")
        logger.info(f"Released {name} to cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
