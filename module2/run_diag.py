import sys, traceback
try:
    print("Testing torch...")
    import torch
    import torchvision.models as models
    print("Device mps is available:", torch.backends.mps.is_available())
    device = torch.device("mps")
    m = models.efficientnet_b3(weights=None)
    m.classifier[1] = torch.nn.Linear(m.classifier[1].in_features, 2)
    m.to(device)
    print("Model moved to MPS successfully.")
    x = torch.randn(4, 3, 300, 300, device=device)
    y = m(x)
    print("Forward passed. Output shape:", y.shape)
    loss = y.sum()
    loss.backward()
    print("Backward passed successfully.")
except Exception as e:
    print("Exception occurred:", e)
    traceback.print_exc()