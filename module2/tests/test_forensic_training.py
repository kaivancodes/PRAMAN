"""Regression tests for the training contracts."""
import sys
from pathlib import Path
import numpy as np
from PIL import Image
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from module2.models.efficientnet_detector import build_efficientnet_b3, MANIPULATION_CLASSES
from module2.preprocessing.ela import ErrorLevelAnalysis
from module2.preprocessing.dct import DiscreteCosineTransform


def test_efficientnet_uses_five_channels():
    model = build_efficientnet_b3(pretrained=False, input_channels=5)
    assert model.features[0][0].in_channels == 5


def test_efficientnet_has_forgery_and_reason_heads():
    model = build_efficientnet_b3(pretrained=False, input_channels=5)
    model.eval()
    x = torch.zeros(2, 5, 300, 300)
    with torch.inference_mode():
        out = model(x)
    assert out["forgery_logits"].shape == (2, 2)
    assert out["reason_logits"].shape == (2, len(MANIPULATION_CLASSES))
    assert torch.isfinite(out["forgery_logits"]).all()
    assert torch.isfinite(out["reason_logits"]).all()


def test_ela_and_dct_are_finite():
    image = Image.fromarray(np.random.randint(0, 256, (128, 192, 3), dtype=np.uint8))
    ela = ErrorLevelAnalysis().compute_ela(image)
    dct = DiscreteCosineTransform().compute_dct_map(image)
    assert np.isfinite(np.asarray(ela)).all()
    assert np.isfinite(dct).all()


def test_five_channel_representation_shape():
    image = Image.fromarray(np.random.randint(0, 256, (128, 192, 3), dtype=np.uint8))
    image = image.resize((300, 300))
    ela = ErrorLevelAnalysis().compute_ela(image).convert("L")
    dct = DiscreteCosineTransform().compute_dct_map(image)
    dct = np.log1p(np.abs(dct))
    dct = (dct-dct.min()) / max(float(dct.max()-dct.min()), 1e-6)

    rgb = torch.from_numpy(np.asarray(image, dtype=np.float32)/255).permute(2,0,1)
    ela_t = torch.from_numpy(np.asarray(ela, dtype=np.float32)/255).unsqueeze(0)
    dct_t = torch.from_numpy(dct.astype(np.float32)).unsqueeze(0)

    x = torch.cat([rgb, ela_t, dct_t], dim=0)
    assert x.shape == (5,300,300)
    assert torch.isfinite(x).all()
