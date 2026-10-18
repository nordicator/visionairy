"""Export the official Depth Anything V2 Metric Indoor (Small) model to ONNX for depth.py.

Only needed once per machine, and only if you want to rebuild the models in models/:
  pip install torch --index-url https://download.pytorch.org/whl/cpu
  pip install transformers onnx onnxscript
  python export_depth.py                 # writes models/depth_indoor_s_<H>x<W>.onnx for each size

Sizes are multiples of the model's 14 px patch, in the two picture shapes in use, so frames aren't
squashed (a squashed picture skews distances): 5:3 for the drone camera, 4:3 for webcams; a fast
and a fine one of each. depth.py picks the one matching the camera.
"""
import os
import sys

import torch
from transformers import AutoModelForDepthEstimation

REPO = "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"
SIZES = [(252, 420), (378, 630),     # 5:3 - the drone camera (640x384)
         (294, 392), (378, 504)]     # 4:3 - webcams (640x480)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")


class Wrapped(torch.nn.Module):
    """RGB 0-255 image in, metres out - normalisation baked in so depth.py stays simple."""

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1) * 255)
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1) * 255)

    def forward(self, rgb):
        return self.model(pixel_values=(rgb - self.mean) / self.std).predicted_depth


def main():
    model = Wrapped(AutoModelForDepthEstimation.from_pretrained(REPO).eval())
    os.makedirs(OUT, exist_ok=True)
    for h, w in SIZES:
        path = os.path.join(OUT, f"depth_indoor_s_{h}x{w}.onnx")
        x = torch.rand(1, 3, h, w) * 255
        with torch.no_grad():
            ref = model(x)
            torch.onnx.export(model, (x,), path, input_names=["rgb"], output_names=["depth"],
                              opset_version=17, dynamo=False)
        print(f"{path}: output {tuple(ref.shape)}, {os.path.getsize(path) / 1e6:.0f} MB")


if __name__ == "__main__":
    sys.exit(main())
