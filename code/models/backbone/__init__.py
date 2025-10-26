from . import feature
from . import vision

BACKBONES = {
    "resnet18": vision.ResNet18, # ~11.7M parameters
    "conv4": vision.Conv4, # ~0.1M parameters
    # Added new backbones:
    "vit2": vision.ViT2,
    "tiny_vit": vision.tiny_vit
}
