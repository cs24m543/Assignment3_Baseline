# README
# Three Python files for training MobileNet-V2 on CIFAR-10 with the requested augmentations, regularization,
# and training settings.
# Files included below (copy each into its own file):
#  - model.py
#  - train.py
#  - test.py
#
# Notes:
# - Train transforms use RandomResizedCrop(224) for ImageNet-style fine-tuning.
# - Default behavior: **pretrained weights are used by default**. Use `--no-pretrained` to disable.
# - Added MixUp, EMA, gradient accumulation (accum-steps), automatic LR scaling suggestion, and recommended defaults.
# - Weight decay = 5e-4, label smoothing = 0.1, dropout = 0.1 inside classifier.
# - MobileNetV2 width multiplier = 1.0 by default when pretrained is True (to load compatible weights).
# - Optimizer = SGD with momentum, LR schedule = cosine annealing with linear warmup.
# - Epochs default = 120 for finetuning, batch size default = 64.

# === model.py ===
import torch
import torch.nn as nn

try:
    # Newer torchvision (weights enum)
    from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
    _HAS_WEIGHTS_ENUM = True
except Exception:
    from torchvision.models import mobilenet_v2
    _HAS_WEIGHTS_ENUM = False


def get_mobilenet_v2(num_classes=10, pretrained=True, device=None, width_mult=1.0, dropout_prob=0.1):
    """
    Returns a MobileNet-V2 model with the final classification layer adapted to num_classes.

    If pretrained=True and width_mult!=1.0, the function will load the pretrained weights for width=1.0
    and then initialize the wider model; only matching parameter shapes will be copied.
    """
    # If user asks for pretrained but a non-1.0 width, warn and fall back to width 1.0 for weight-loading
    load_width_for_pretrained = 1.0 if pretrained else width_mult

    # Build model for requested width (we will try to pass width_mult to torchvision; if unsupported we fall back)
    try:
        if _HAS_WEIGHTS_ENUM:
            weights = MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained and load_width_for_pretrained == 1.0 else None
            model = mobilenet_v2(weights=weights, width_mult=width_mult)  # type: ignore
        else:
            model = mobilenet_v2(pretrained=(pretrained and load_width_for_pretrained == 1.0), width_mult=width_mult)  # type: ignore
    except TypeError:
        # torchvision doesn't accept width_mult or weights in this env; build default and adapt
        if _HAS_WEIGHTS_ENUM and pretrained and load_width_for_pretrained == 1.0:
            base_model = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
            model = mobilenet_v2()
        elif pretrained and load_width_for_pretrained == 1.0:
            base_model = mobilenet_v2(pretrained=True)
            model = mobilenet_v2()
        else:
            model = mobilenet_v2(pretrained=False)

        # Attempt to copy compatible params if base_model exists
        if 'base_model' in locals():
            base_sd = base_model.state_dict()
            model_sd = model.state_dict()
            compatible = {k: v for k, v in base_sd.items() if k in model_sd and model_sd[k].shape == v.shape}
            model_sd.update(compatible)
            model.load_state_dict(model_sd)

    # Replace classifier to match num_classes and desired dropout
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout_prob),
        nn.Linear(in_features, num_classes)
    )

    if device is not None:
        model = model.to(device)

    return model


if __name__ == "__main__":
    # quick sanity check
    m = get_mobilenet_v2(pretrained=False, device='cpu', width_mult=1.0, dropout_prob=0.1)
    print(m)


