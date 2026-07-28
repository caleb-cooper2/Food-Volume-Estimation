import torch
import torch.nn as nn
from PIL import Image

CLIP_MODEL_NAME = "openai/clip-vit-base-patch16"
CLIP_FEATURE_DIMENSIONS = 512 # image-projection dimensions of ViT-B/16


class SizePriorHead(nn.Module):
    """
    Small MLP mapping a frozen CLIP image embedding to the food's real-world size, as log equivalent footprint diameter (log cm).
    The CLIP trunk stays frozen so a few thousand overhead dishes are enough to fit just this head in theory
    """
    def __init__(self, feature_dim: int = CLIP_FEATURE_DIMENSIONS):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )

    def forward(self, clip_features: torch.Tensor) -> torch.Tensor:
        return self.head(clip_features).squeeze(1) # log cm


@torch.no_grad()
def extract_clip_features(pillow_image: Image.Image, clip_model, clip_processor, device: torch.device) -> torch.Tensor:
    """Unit-normalised CLIP image embedding (1, feature_dim) -> the frozen input the size-prior head reads"""
    inputs = clip_processor(images=pillow_image, return_tensors="pt").to(device)
    image_features = clip_model.get_image_features(**inputs)
    image_features = image_features.pooler_output
    features = image_features / image_features.norm(dim=-1, keepdim=True) # CLIP embeddings are meant to be used unit-norm
    return features.float()
