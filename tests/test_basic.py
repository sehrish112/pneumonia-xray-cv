import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import torch
from model import build_model


def test_model_output_shape():
    m = build_model(pretrained=False).eval()
    assert m(torch.randn(2, 3, 64, 64)).shape == (2, 1)


def test_probability_range():
    m = build_model(pretrained=False).eval()
    p = torch.sigmoid(m(torch.randn(4, 3, 64, 64)))
    assert ((p >= 0) & (p <= 1)).all()
