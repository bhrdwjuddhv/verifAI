"""NPR — Neighboring Pixel Relationships (Tan et al., CVPR 2024).

Ported from the official repo, networks/resnet.py:
    https://github.com/chuangchuangtan/NPR-DeepfakeDetection

Why this model exists here: the face classifier only sees face-swap artifacts, so a fully
generated face (StyleGAN, "this person does not exist") sails past it. NPR ignores content
entirely and looks at the up-sampling artifact every generator's decoder leaves behind, which
is why it transfers to generators it never saw.

Faithful to the official code in the parts that define the method:
  * NPR = x - upsample(downsample(x)), nearest, factor 2 — then scaled by 2/3 into conv1.
  * conv1 is 3x3 stride 2 (NOT torchvision's 7x7), and the trunk stops after layer2.
  * One logit out; sigmoid gives P(fake). Official label convention: 0_real / 1_fake.

Everything else is torchvision's ResNet, which the official file is itself a copy of. Module
names match theirs exactly, so the repo's released checkpoint loads with strict=True.
"""

import torch
import torch.nn as nn
from torch.nn import functional as F

# Official options/base_options.py: loadSize 256, cropSize 224, ImageNet normalization.
IMG_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def npr_residual(x):
    """x minus its own 2x down-then-up nearest reconstruction.

    Zero wherever the image already *is* a nearest 2x up-sample; non-zero, and generator-
    specific, wherever real pixel detail lives. `interpolate(img, 0.5)` in the official code
    downsamples then upsamples — this is that, written so it also exports to ONNX.
    """
    down = x[:, :, ::2, ::2]
    up = down.repeat_interleave(2, dim=2).repeat_interleave(2, dim=3)
    return x[:, :, : up.shape[2], : up.shape[3]] - up


def official_npr_residual(x):
    """The official formulation verbatim. Kept only so selfcheck can prove the two agree."""
    half = F.interpolate(x, scale_factor=0.5, mode="nearest", recompute_scale_factor=True)
    return x - F.interpolate(half, scale_factor=2.0, mode="nearest", recompute_scale_factor=True)


class NPRDetector(nn.Module):
    """ResNet-50 trunk truncated after layer2, fed the NPR residual instead of pixels."""

    def __init__(self):
        super().__init__()
        from torchvision.models import resnet50

        # ponytail: torchvision builds layer3/layer4 we never keep — a moment of wasted init,
        # against ~200 lines of copied block definitions. Reuse wins.
        r = resnet50()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1, self.relu, self.maxpool = r.bn1, r.relu, r.maxpool
        self.layer1, self.layer2 = r.layer1, r.layer2
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(512, 1)  # layer2 out = 128 * Bottleneck.expansion(4)

    def forward(self, x):
        x = self.conv1(npr_residual(x) * 2.0 / 3.0)
        x = self.maxpool(self.relu(self.bn1(x)))
        x = self.layer2(self.layer1(x))
        return self.fc1(self.avgpool(x).flatten(1))


def load_npr(checkpoint_path, device="cpu"):
    """Loads our checkpoint or the official model_epoch_last_3090.pth — same key names."""
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    state = state.get("model", state.get("state_dict", state))
    model = NPRDetector()
    model.load_state_dict(state)  # strict: a silent partial load is a silently wrong model
    return model.to(device).eval()


def selfcheck():
    torch.manual_seed(0)
    x = torch.rand(2, 3, 64, 64)

    # The two residual formulations must agree — this is what licenses the ONNX-safe one.
    assert torch.allclose(npr_residual(x), official_npr_residual(x), atol=1e-6)

    # Defining property: an image that IS a 2x nearest up-sample has no NPR residual.
    upsampled = torch.rand(2, 3, 32, 32).repeat_interleave(2, 2).repeat_interleave(2, 3)
    assert npr_residual(upsampled).abs().max() < 1e-6, "up-sampled input must residual to zero"
    assert npr_residual(x).abs().max() > 1e-3, "natural detail must survive"

    model = NPRDetector()
    out = model(torch.rand(2, 3, IMG_SIZE, IMG_SIZE))
    assert out.shape == (2, 1), out.shape

    keys = set(model.state_dict())
    assert "fc1.weight" in keys and not any(k.startswith("layer3") for k in keys)
    assert model.state_dict()["fc1.weight"].shape == (1, 512)
    assert model.conv1.weight.shape == (64, 3, 3, 3), "official conv1 is 3x3, not torchvision 7x7"
    print("npr_model selfcheck passed")


if __name__ == "__main__":
    selfcheck()
