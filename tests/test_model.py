import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from src.model import AnomalyDetector

def test_model_forward():
    input_dim = 3
    model = AnomalyDetector(input_dim)
    x = torch.randn(10, input_dim)
    output = model(x)
    assert output.shape == x.shape, "Output shape mismatch"
    print("Test passed: Model forward pass works")

if __name__ == "__main__":
    test_model_forward()