"""EfficientNet-B3 model implementation for SIDTD document forgery detection.

Binary classifier: BONA_FIDE vs FORGED.
Note: This model is for Module 2.2 tamper/forgery detection, NOT AI-generation detection.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
from torchvision.models import EfficientNet_B3_Weights, efficientnet_b3

from module2.utils.device import get_device
from module2.utils.logger import get_logger

logger = get_logger("module2.models.efficientnet_detector")

CLASS_NAMES = ["BONA_FIDE", "FORGED"]


def build_efficientnet_b3(
    num_classes: int = 2,
    pretrained: bool = False,
    dropout: float = 0.3,
) -> nn.Module:
    """Construct EfficientNet-B3 with custom classification head.

    Args:
        num_classes: Number of target classes (2 for BONA_FIDE vs FORGED).
        pretrained: If True, loads ImageNet pretrained backbone weights.
        dropout: Dropout probability before the linear projection.

    Returns:
        torch.nn.Module instance.
    """
    weights = EfficientNet_B3_Weights.DEFAULT if pretrained else None
    model = efficientnet_b3(weights=weights)

    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout, inplace=True),
        nn.Linear(in_features, num_classes),
    )
    return model


class SIDTDEfficientNetDetector:
    """Manages the lifecycle, loading, and inference of the SIDTD EfficientNet-B3 model.

    Ensures single checkpoint loading across multiple documents and cases.
    """

    def __init__(
        self,
        checkpoint_path: Optional[Union[str, Path]] = None,
        device: Optional[Union[str, torch.device]] = None,
        num_classes: int = 2,
        class_names: Optional[List[str]] = None,
    ):
        """Initialize and load the model into memory.

        Args:
            checkpoint_path: Path to .pth checkpoint file.
            device: Target torch device or 'auto'.
            num_classes: Number of classes (default 2).
            class_names: Names of classes, default ['BONA_FIDE', 'FORGED'].
        """
        self.device = get_device(device) if not isinstance(device, torch.device) else device
        self.num_classes = num_classes
        self.class_names = class_names or CLASS_NAMES
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None

        self.model = build_efficientnet_b3(num_classes=self.num_classes, pretrained=False)
        self._load_checkpoint()
        self.model.to(self.device)
        self.model.eval()

        logger.info(
            f"SIDTD EfficientNet-B3 initialized on {self.device} (classes: {self.class_names})"
        )

    def _load_checkpoint(self) -> None:
        """Load weights from checkpoint path if available."""
        if self.checkpoint_path and self.checkpoint_path.exists():
            try:
                checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
                if isinstance(checkpoint, dict):
                    if "state_dict" in checkpoint:
                        state_dict = checkpoint["state_dict"]
                    elif "model_state_dict" in checkpoint:
                        state_dict = checkpoint["model_state_dict"]
                    else:
                        state_dict = checkpoint
                else:
                    state_dict = checkpoint

                # Remove module. prefix if trained with DataParallel
                cleaned_state = {
                    (k[7:] if k.startswith("module.") else k): v
                    for k, v in state_dict.items()
                }
                self.model.load_state_dict(cleaned_state, strict=False)
                logger.info(f"Loaded SIDTD checkpoint from {self.checkpoint_path}")
            except Exception as e:
                logger.warning(
                    f"Could not load weights from {self.checkpoint_path}: {e}. "
                    "Using initialized weights for inference/testing."
                )
        else:
            if self.checkpoint_path:
                logger.warning(
                    f"Checkpoint not found at {self.checkpoint_path}. "
                    "Operating with initialized architecture."
                )

    @torch.inference_mode()
    def forward_probs(self, tensor: torch.Tensor) -> torch.Tensor:
        """Execute forward pass and return softmax probabilities.

        Args:
            tensor: Input tensor of shape (B, 3, H, W) on self.device.

        Returns:
            Probabilities tensor of shape (B, num_classes).
        """
        if tensor.device != self.device:
            tensor = tensor.to(self.device)

        logits = self.model(tensor)
        probs = torch.softmax(logits, dim=-1)
        return probs
