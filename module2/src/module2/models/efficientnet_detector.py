"""EfficientNet-B3 backbone and multi-output Tamper model."""

from typing import Dict, List, Optional, Union
from pathlib import Path

import torch
import torch.nn as nn
from torchvision.models import EfficientNet_B3_Weights, efficientnet_b3

from module2.utils.device import get_device
from module2.utils.logger import get_logger

logger = get_logger("module2.models.efficientnet_detector")

CLASS_NAMES = ["BONA_FIDE", "FORGED"]

MANIPULATION_CLASSES = [
    "GENERAL_MANIPULATION",
    "NAME_FIELD_MANIPULATION",
    "DOB_FIELD_MANIPULATION",
    "DOCUMENT_NUMBER_FIELD_MANIPULATION",
    "PHOTO_REPLACEMENT",
    "COPY_PASTE_SPLICING",
    "ERASURE_REWRITING",
    "DOCUMENT_RECONSTRUCTION",
]


class EfficientNetB3TamperModel(nn.Module):
    """One EfficientNet-B3 model with a binary head and manipulation-reason head.

    The model still represents ONE Tamper branch. The second head only makes
    the failure reason explicit when the training data supports that label.
    """

    def __init__(
        self,
        num_classes: int = 2,
        reason_classes: int = len(MANIPULATION_CLASSES),
        pretrained: bool = False,
        dropout: float = 0.3,
        input_channels: int = 5,
    ):
        super().__init__()
        weights = EfficientNet_B3_Weights.DEFAULT if pretrained else None
        base = efficientnet_b3(weights=weights)

        old_conv = base.features[0][0]
        if input_channels != 3:
            new_conv = nn.Conv2d(
                input_channels,
                old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                dilation=old_conv.dilation,
                groups=old_conv.groups,
                bias=old_conv.bias is not None,
            )
            with torch.no_grad():
                new_conv.weight[:, :3] = old_conv.weight
                extra = old_conv.weight.mean(dim=1, keepdim=True)
                new_conv.weight[:, 3:] = extra.repeat(1, input_channels - 3, 1, 1)
                if old_conv.bias is not None:
                    new_conv.bias.copy_(old_conv.bias)
            base.features[0][0] = new_conv

        self.features = base.features
        self.avgpool = base.avgpool
        in_features = base.classifier[1].in_features
        self.dropout = nn.Dropout(p=dropout)
        self.forgery_head = nn.Linear(in_features, num_classes)
        self.reason_head = nn.Linear(in_features, reason_classes)
        self.input_channels = input_channels
        self.reason_classes = reason_classes

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return {
            "forgery_logits": self.forgery_head(x),
            "reason_logits": self.reason_head(x),
        }


def build_efficientnet_b3(
    num_classes: int = 2,
    pretrained: bool = False,
    dropout: float = 0.3,
    input_channels: int = 5,
    reason_classes: int = len(MANIPULATION_CLASSES),
) -> nn.Module:
    return EfficientNetB3TamperModel(
        num_classes=num_classes,
        reason_classes=reason_classes,
        pretrained=pretrained,
        dropout=dropout,
        input_channels=input_channels,
    )


class EfficientNetB3TamperDetector:
    """Production wrapper for the single Tamper model."""

    def __init__(
        self,
        checkpoint_path: Optional[Union[str, Path]] = None,
        device: Optional[Union[str, torch.device]] = None,
        num_classes: int = 2,
        class_names: Optional[List[str]] = None,
        reason_classes: Optional[List[str]] = None,
        reason_threshold: float = 0.50,
    ):
        self.device = get_device(device) if not isinstance(device, torch.device) else device
        self.class_names = class_names or CLASS_NAMES
        self.reason_classes = reason_classes or MANIPULATION_CLASSES
        self.reason_threshold = reason_threshold
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        self.model = build_efficientnet_b3(
            num_classes=num_classes,
            pretrained=False,
            input_channels=5,
            reason_classes=len(self.reason_classes),
        )
        self.checkpoint_metadata = {}
        self._load_checkpoint()
        self.model.to(self.device)
        self.model.eval()

    def _load_checkpoint(self) -> None:
        if not self.checkpoint_path or not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Tamper checkpoint not found: {self.checkpoint_path}")

        checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        state = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
        state = {(k[7:] if k.startswith("module.") else k): v for k, v in state.items()}
        self.model.load_state_dict(state, strict=True)
        self.checkpoint_metadata = checkpoint if isinstance(checkpoint, dict) else {}

    @torch.inference_mode()
    def forward(self, tensor: torch.Tensor) -> Dict[str, object]:
        tensor = tensor.to(self.device)
        outputs = self.model(tensor)
        forgery_probs = torch.softmax(outputs["forgery_logits"], dim=1)
        reason_probs = torch.sigmoid(outputs["reason_logits"])
        return {
            "bona_fide_probability": float(forgery_probs[0, 0].item()),
            "forged_probability": float(forgery_probs[0, 1].item()),
            "reason_probabilities": {
                name: float(reason_probs[0, i].item())
                for i, name in enumerate(self.reason_classes)
            },
        }


# Backward-compatible import name.
SIDTDEfficientNetDetector = EfficientNetB3TamperDetector
