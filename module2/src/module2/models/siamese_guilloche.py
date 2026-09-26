"""Siamese-style ResNet-18 model for Guilloché and security background pattern forensics.

Extracts normalized embedding vectors for security pattern crops and calculates
cosine similarity against authentic reference patterns.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet18_Weights, resnet18

from module2.utils.device import get_device
from module2.utils.logger import get_logger

logger = get_logger("module2.models.siamese_guilloche")


class SiameseResNet18(nn.Module):
    """Siamese-style ResNet-18 feature encoder for security patterns.

    Projects input pattern crops into a compact, normalized metric space.
    """

    def __init__(
        self,
        embedding_dim: int = 128,
        pretrained: bool = False,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim

        # Load ResNet-18 backbone
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        base_model = resnet18(weights=weights)

        # Feature extractor up to adaptive pooling
        self.backbone = nn.Sequential(
            base_model.conv1,
            base_model.bn1,
            base_model.relu,
            base_model.maxpool,
            base_model.layer1,
            base_model.layer2,
            base_model.layer3,
            base_model.layer4,
            base_model.avgpool,
        )

        in_features = base_model.fc.in_features  # 512

        # Projection MLP head
        self.projector = nn.Sequential(
            nn.Flatten(1),
            nn.Dropout(p=dropout),
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, embedding_dim),
        )

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through backbone and projection head.

        Args:
            x: Input tensor of shape (B, 3, H, W).

        Returns:
            L2-normalized embedding tensor of shape (B, embedding_dim).
        """
        features = self.backbone(x)
        embeddings = self.projector(features)
        return F.normalize(embeddings, p=2, dim=-1)

    def forward(
        self,
        x1: torch.Tensor,
        x2: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass.

        If x2 is provided, returns pair of embeddings (emb1, emb2).
        Otherwise, returns single embedding tensor.
        """
        emb1 = self.extract_features(x1)
        if x2 is not None:
            emb2 = self.extract_features(x2)
            return emb1, emb2
        return emb1


class GuillocheSiameseDetector:
    """Inference manager for Siamese ResNet-18 Guilloché Pattern Verifier.

    Loads and persists checkpoint across inference requests.
    """

    def __init__(
        self,
        checkpoint_path: Optional[Union[str, Path]] = None,
        embedding_dim: int = 128,
        device: Optional[Union[str, torch.device]] = None,
    ):
        self.device = get_device(device) if not isinstance(device, torch.device) else device
        self.embedding_dim = embedding_dim
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None

        self.model = SiameseResNet18(embedding_dim=embedding_dim, pretrained=False)
        self._load_checkpoint()
        self.model.to(self.device)
        self.model.eval()

        logger.info(
            f"Guilloché Siamese ResNet-18 initialized on {self.device} (embedding_dim={embedding_dim})"
        )

    def _load_checkpoint(self) -> None:
        """Load weights from checkpoint path if available."""
        if self.checkpoint_path and self.checkpoint_path.exists():
            try:
                try:
                    checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
                except TypeError:
                    checkpoint = torch.load(self.checkpoint_path, map_location=self.device)

                if isinstance(checkpoint, dict):
                    if "model_state_dict" in checkpoint:
                        state_dict = checkpoint["model_state_dict"]
                    elif "state_dict" in checkpoint:
                        state_dict = checkpoint["state_dict"]
                    else:
                        state_dict = checkpoint
                else:
                    state_dict = checkpoint

                self.model.load_state_dict(state_dict, strict=False)
                logger.info(f"Loaded Guilloché Siamese checkpoint from {self.checkpoint_path}")
            except Exception as e:
                logger.warning(
                    f"Failed loading checkpoint from {self.checkpoint_path}: {e}. "
                    "Using initialized architecture."
                )
        else:
            if self.checkpoint_path:
                logger.warning(
                    f"Guilloché checkpoint not found at {self.checkpoint_path}. "
                    "Operating with initialized Siamese ResNet-18."
                )

    @torch.inference_mode()
    def embed(self, tensor: torch.Tensor) -> torch.Tensor:
        """Compute L2-normalized embedding for input tensor (B, 3, H, W)."""
        tensor = tensor.to(self.device)
        return self.model.extract_features(tensor)

    @torch.inference_mode()
    def compute_similarity(self, tensor1: torch.Tensor, tensor2: torch.Tensor) -> float:
        """Compute cosine similarity between two pattern tensors."""
        emb1 = self.embed(tensor1)
        emb2 = self.embed(tensor2)
        sim = F.cosine_similarity(emb1, emb2).item()
        return float(sim)
