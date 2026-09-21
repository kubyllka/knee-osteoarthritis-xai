import torch.nn as nn
from torchvision import models


def create_vit_model(num_classes=5, freeze_backbone=True):
    """
    Creates the same ViT architecture used during training.
    The head must match the checkpoint from Lab 3.
    """
    model = models.vit_b_16(weights=None)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    num_features = model.heads.head.in_features
    model.heads.head = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_features, 512),
        nn.GELU(),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes),
    )

    return model


def create_resnet50_model(num_classes=5, freeze_backbone=True):
    """
    Creates the ResNet-50 architecture used for KL grade classification.
    The classification head mirrors the user's training code.
    """
    model = models.resnet50(weights=None)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_features, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes),
    )

    return model


def create_resnet50_ordinal_model(num_classes=5, freeze_backbone=True):
    """ResNet-50 architecture used by the final ordinal-BCE checkpoint."""
    return create_resnet50_model(num_classes=num_classes - 1, freeze_backbone=freeze_backbone)


def create_densenet121_model(num_classes=5, freeze_backbone=True):
    """Create a standard DenseNet-121 with a KL-grade classification head."""
    model = models.densenet121(weights=None)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    num_features = model.classifier.in_features
    # The local DenseNet checkpoint stores a 512-unit fine-tuning head.
    # Dropout is inactive for inference and XAI because the model is in eval mode.
    model.classifier = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_features, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes),
    )
    return model


def create_densenet121_ordinal_model(num_classes=5, freeze_backbone=True):
    """DenseNet-121 architecture used by the final ordinal-BCE checkpoint."""
    model = models.densenet121(weights=None)
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
    num_features = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_features, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes - 1),
    )
    return model


def vit_reshape_transform(tensor, height=14, width=14):
    result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))
    return result.permute(0, 3, 1, 2)


def identity_reshape_transform(tensor):
    return tensor
