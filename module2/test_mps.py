import sys
print("Python executable:", sys.executable)
import torch
print("PyTorch version:", torch.__version__)
print("MPS available:", torch.backends.mps.is_available())