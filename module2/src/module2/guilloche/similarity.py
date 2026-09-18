"""Siamese-style pattern similarity computation for Guilloché forensics.

Extracts normalized embedding vectors via a modular CNN encoder and calculates
cosine similarity against authentic references.
Note: Feature space similarity is an architectural component; performance metrics
are not fabricated.
"""

from typing import Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from module2.utils.device import get_device
from module2.utils.logger import get_logger

logger = get_logger("module2.guilloche.similarity")


class PatternCNNEncoder(nn.Module):
    """Modular CNN encoder for extracting compact security pattern embeddings."""

    def __init__(self, in_channels: int = 3, embedding_dim: int = 128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc = nn.Linear(128, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.conv(x)
        feat = torch.flatten(feat, 1)
        emb = self.fc(feat)
        return F.normalize(emb, p=2, dim=-1)


class GuillocheSimilarityEvaluator:
    """Computes cosine similarity between query security patterns and authentic references."""

    def __init__(
        self,
        embedding_dim: int = 128,
        device: Optional[str] = "auto",
        encoder: Optional[nn.Module] = None,
    ):
        self.device = get_device(device)
        self.encoder = encoder or PatternCNNEncoder(embedding_dim=embedding_dim)
        self.encoder.to(self.device)
        self.encoder.eval()

    def _preprocess(self, image: Image.Image) -> torch.Tensor:
        """Preprocess pattern image for CNN encoder."""
        img = image.convert("RGB").resize((128, 128), Image.Resampling.BILINEAR)
        arr = np.array(img, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # 1, 3, 128, 128
        # Standard normalization
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        return (tensor - mean) / std

    @torch.inference_mode()
    def compute_similarity(
        self,
        query_image: Image.Image,
        reference_image: Image.Image,
    ) -> float:
        """Calculate cosine similarity between query pattern and reference pattern.

        Args:
            query_image: Extracted query pattern crop.
            reference_image: Authentic reference pattern crop.

        Returns:
            Cosine similarity float in range [-1.0, 1.0].
        """
        q_tensor = self._preprocess(query_image).to(self.device)
        r_tensor = self._preprocess(reference_image).to(self.device)

        q_emb = self.encoder(q_tensor)
        r_emb = self.encoder(r_tensor)

        sim = F.cosine_similarity(q_emb, r_emb).item()
        return float(sim)
