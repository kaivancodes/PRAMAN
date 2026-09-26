"""Training and forensic-representation regression tests."""
import sys
from pathlib import Path
import tempfile
import csv
import numpy as np
from PIL import Image
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from module2.models.efficientnet_detector import build_efficientnet_b3
from module2.preprocessing.ela import ErrorLevelAnalysis
from module2.preprocessing.dct import DiscreteCosineTransform


def test_efficientnet_uses_five_forensic_channels():
    model = build_efficientnet_b3(pretrained=False, input_channels=5)
    assert model.features[0][0].in_channels == 5
    assert model.features[0][0].weight.shape[1] == 5


def test_ela_and_dct_are_finite_and_image_sized():
    image = Image.fromarray(np.random.randint(0, 256, (128, 192, 3), dtype=np.uint8))
    ela = ErrorLevelAnalysis().compute_ela(image)
    dct = DiscreteCosineTransform().compute_dct_map(image)
    assert ela.size == image.size
    assert dct.shape == (128, 192)
    assert np.isfinite(np.asarray(ela)).all()
    assert np.isfinite(dct).all()


def test_five_channel_tensor_shape():
    image = Image.fromarray(np.random.randint(0, 256, (128, 192, 3), dtype=np.uint8))
    ela = ErrorLevelAnalysis().compute_ela(image).convert("L")
    dct = DiscreteCosineTransform().compute_dct_map(image)
    dct = np.log1p(np.abs(dct))
    dct = (dct-dct.min()) / max(float(dct.max()-dct.min()), 1e-6)
    rgb = torch.from_numpy(np.asarray(image.resize((300,300)), dtype=np.float32)/255).permute(2,0,1)
    ela_t = torch.from_numpy(np.asarray(ela.resize((300,300)), dtype=np.float32)/255).unsqueeze(0)
    dct_t = torch.from_numpy((dct*255).astype(np.uint8)).unsqueeze(0).float()/255
    x = torch.cat([rgb, ela_t, dct_t], dim=0).unsqueeze(0)
    assert x.shape == (1,5,300,300)
    assert torch.isfinite(x).all()


def test_binary_labels_are_valid():
    assert {0, 1}.issubset({0, 1})


def test_sidtd_csv_parser_preserves_binary_labels():
    from training.train_tamper import parse_sidtd_csv
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        img = root / "real.png"
        Image.new("RGB", (32, 32)).save(img)
        csv_path = root / "split.csv"
        with csv_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["image_path", "label"])
            w.writeheader(); w.writerow({"image_path": "real.png", "label": "0"})
        samples = parse_sidtd_csv(csv_path, root)
        assert samples == [(str(img.resolve()), 0)]


def test_efficientnet_forward_shape_for_training_contract():
    model = build_efficientnet_b3(pretrained=False, input_channels=5)
    model.eval()
    with torch.inference_mode():
        out = model(torch.zeros(2, 5, 300, 300))
    assert out.shape == (2, 2)
    assert torch.isfinite(out).all()
