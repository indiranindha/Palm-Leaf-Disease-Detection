import timm
import torch.nn as nn

def build_convnext(num_classes, pretrained=True, freeze_backbone=True):
    model = timm.create_model(
        "hf_hub:timm/convnext_tiny.in12k_ft_in1k",
        pretrained=pretrained
    )

    # in_features = model.get_classifier().in_features

    # model.head = nn.Sequential(
    #     nn.Dropout(p=0.3),
    #     nn.Linear(in_features, num_classes)
    # )

    # if freeze_backbone:
    #     for name, param in model.named_parameters():
    #         if "head" not in name:
    #             param.requires_grad = False

    # return model

    model.reset_classifier(num_classes)

    if freeze_backbone:
        for name, param in model.named_parameters():
            if "head" not in name and "classifier" not in name:
                param.requires_grad = False

    return model
