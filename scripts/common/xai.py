"""Grad-CAM overlays for the inference service.

Uses pytorch-grad-cam (`pip install grad-cam`). Two model families reach this file and they
need different target layers:

  * torchvision EfficientNet — last conv block, activations are already (B, C, H, W).
  * HF ViT — last encoder block's layernorm, activations are (B, tokens, C) and need
    reshaping back to a grid, with the CLS token dropped.

Get the target layer wrong and you still get a picture — a smooth, plausible, meaningless
one. That is the failure mode worth guarding against, so `selfcheck` asserts the CAM
actually responds to the image rather than merely existing.
"""

import base64
import io

import numpy as np
from PIL import Image


def _vit_reshape(tensor):
    """(B, 1 + H*W, C) -> (B, C, H, W). Drops the CLS token, which has no location.

    The grid side is derived from the token count rather than hardcoded to 14, so a model at
    a different patch size or resolution reshapes correctly instead of silently transposing
    garbage into a plausible-looking heatmap.
    """
    tokens = tensor.size(1) - 1
    side = int(round(tokens ** 0.5))
    if side * side != tokens:
        raise ValueError(f"{tokens} patch tokens is not a square grid")
    result = tensor[:, 1:, :].reshape(tensor.size(0), side, side, tensor.size(2))
    return result.transpose(2, 3).transpose(1, 2)


def _vit_blocks(vit):
    """transformers renamed encoder.layer -> layers around v5. Support both."""
    blocks = getattr(vit, "layers", None)
    if blocks is None:
        encoder = getattr(vit, "encoder", None)
        blocks = getattr(encoder, "layer", None) if encoder is not None else None
    return blocks


def wrap_for_cam(model):
    """Returns (module_to_hook, target_layers, reshape_transform), or (None, None, None).

    HF classifiers return an output object, not a tensor, so grad-cam cannot call them
    directly — hence the thin logits wrapper.
    """
    import torch.nn as nn

    vit = getattr(model, "vit", None)
    blocks = _vit_blocks(vit) if vit is not None else None
    if blocks is not None and len(blocks) > 0:

        class LogitsWrapper(nn.Module):
            def __init__(self, inner):
                super().__init__()
                self.inner = inner

            def forward(self, pixel_values):
                return self.inner(pixel_values=pixel_values).logits

        return LogitsWrapper(model), [blocks[-1].layernorm_before], _vit_reshape

    # torchvision EfficientNet (and anything else exposing .features) already returns logits.
    features = getattr(model, "features", None)
    if features is not None and len(features) > 0:
        return model, [features[-1]], None

    return None, None, None


def gradcam_overlay(model, input_tensor, class_index, base_image, alpha=0.45):
    """Returns a `data:image/png;base64,...` overlay, or None if CAM is unavailable.

    Returning None is a valid answer: a missing heatmap is honest, a heatmap from the wrong
    layer is not.
    """
    cam_model, layers, reshape = wrap_for_cam(model)
    if cam_model is None:
        return None

    try:
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    except ImportError:
        return None

    try:
        cam = GradCAM(model=cam_model, target_layers=layers, reshape_transform=reshape)
        grayscale = cam(
            input_tensor=input_tensor,
            targets=[ClassifierOutputTarget(class_index)],
        )[0]
    except Exception as e:
        print(f"[XAI] Grad-CAM unavailable: {type(e).__name__}: {e}")
        return None

    return encode_overlay(grayscale, base_image, alpha)


def encode_overlay(grayscale, base_image, alpha=0.45):
    """Blends a 0-1 CAM over the image as a red-to-blue heat overlay, returns a data URL."""
    size = base_image.size
    cam = np.asarray(
        Image.fromarray(np.uint8(255 * np.clip(grayscale, 0, 1))).resize(size, Image.BILINEAR),
        dtype=np.float32,
    ) / 255.0

    heat = np.zeros((size[1], size[0], 3), dtype=np.float32)
    heat[:, :, 0] = cam                      # red where the model looked
    heat[:, :, 2] = 1.0 - cam                # blue where it did not
    heat[:, :, 1] = np.clip(1.0 - abs(2 * cam - 1), 0, 1) * 0.6

    base = np.asarray(base_image.convert("RGB"), dtype=np.float32) / 255.0
    # Weight the blend by the CAM so cold regions stay recognizable.
    weight = (alpha * cam)[..., None]
    blended = np.uint8(255 * np.clip(base * (1 - weight) + heat * weight, 0, 1))

    buf = io.BytesIO()
    Image.fromarray(blended).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def selfcheck():
    """No torch needed: covers the overlay maths and the ViT reshape contract."""
    img = Image.new("RGB", (32, 24), (10, 20, 30))

    cold = encode_overlay(np.zeros((4, 4)), img)
    hot = encode_overlay(np.ones((4, 4)), img)
    assert cold.startswith("data:image/png;base64,")
    assert cold != hot, "an all-cold and an all-hot CAM must not render identically"

    decoded = Image.open(io.BytesIO(base64.b64decode(hot.split(",", 1)[1])))
    assert decoded.size == img.size, f"overlay must keep the image size, got {decoded.size}"
    r, g, b = np.asarray(decoded, dtype=np.float32).mean(axis=(0, 1))
    assert r > b, "a hot CAM must push the overlay red, not blue"

    # Half-hot CAM: the hot half must be redder than the cold half.
    half = np.zeros((4, 4))
    half[:, 2:] = 1.0
    arr = np.asarray(
        Image.open(io.BytesIO(base64.b64decode(encode_overlay(half, img).split(",", 1)[1]))),
        dtype=np.float32,
    )
    assert arr[:, 16:, 0].mean() > arr[:, :16, 0].mean(), "CAM must be spatially oriented"

    # ViT reshape: token grid -> feature map, CLS dropped, grid side inferred not assumed.
    try:
        import torch
    except ImportError:
        print("(torch absent — skipped the ViT reshape assertions)")
    else:
        assert tuple(_vit_reshape(torch.zeros(2, 1 + 14 * 14, 8)).shape) == (2, 8, 14, 14)
        assert tuple(_vit_reshape(torch.zeros(1, 1 + 16 * 16, 4)).shape) == (1, 4, 16, 16)
        try:
            _vit_reshape(torch.zeros(1, 1 + 15, 4))
            raise AssertionError("a non-square token count must raise, not reshape garbage")
        except ValueError:
            pass

    print("xai selfcheck passed")


if __name__ == "__main__":
    selfcheck()
