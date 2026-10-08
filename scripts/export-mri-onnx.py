"""Export the source brain MRI ResNet-18 checkpoint to mri-api/model.onnx."""

from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from torch import nn
from torchvision import models

ROOT = Path(__file__).resolve().parents[1]
checkpoint_path = hf_hub_download(
    repo_id="ThisenEkanayake/brain-tumor-detection",
    filename="multiclass-classification/multi_class_resnet.pth",
)
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 4)
checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
state = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
model.load_state_dict({key.removeprefix("module."): value for key, value in state.items()})
model.eval()
torch.onnx.export(
    model,
    torch.zeros(1, 3, 224, 224),
    ROOT / "mri-api" / "model.onnx",
    input_names=["image"],
    output_names=["logits"],
    dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
    opset_version=17,
)
