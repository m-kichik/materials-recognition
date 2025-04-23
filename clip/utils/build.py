import json

import torch

from engine.criterion import vanilla_clip_loss, CLIPLoss, ReCLIPLoss, SigLIPLoss
from modelling import CLIP, LFCLIP, MLPCLIP


def build_experiment(
    config,
    device: str = "cpu",
):
    model_type = config.MODEL.TYPE
    clip_model_name = config.MODEL.CLIP_BACKBONE

    if model_type in ["vanilla_clip", "clip", "siglip"]:
        if config.TRAIN.PRETRAINED:
            model = CLIP(clip_model_name, device=device)
            preprocess = model.preprocess
        else:
            import open_clip

            model, _, preprocess = open_clip.create_model_and_transforms(
                clip_model_name, device=device, pretrained=None
            )

        model = model.to(torch.float32)

    elif model_type == "late_fusion_clip":
        model = LFCLIP(
            clip_model_name=clip_model_name,
            freeze_clip=config.MODEL.FREEZE_CLIP,
            clip_ckpt=config.MODEL.CLIP_CKPT,
            num_heads=config.MODEL.FUSION_HEADS,
            mode="train",
            device=device,
        )

        preprocess = model.preprocess

    elif model_type == "mlp_clip":
        with open(config.TRAIN.MATERIALS_PATH, "r") as file:
            materials = json.load(file)

        model = MLPCLIP(
            clip_model_name=clip_model_name,
            num_classes=len(materials["names"]),
            device=device,
        )

        preprocess = model.preprocess
    else:
        raise NotImplementedError(f"Model {model_type} is not implemented.")

    loss_type = config.TRAIN.CRITERION
    if loss_type == "vanilla":
        criterion = vanilla_clip_loss
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=config.TRAIN.LR, betas=(0.9, 0.98)
        )
    elif loss_type == "CLIP":
        t = config.TRAIN.TEMPERATURE
        criterion = CLIPLoss(t, log_wandb=config.TRAIN.WANDB)
        if model_type == "late_fusion_clip":
            if config.MODEL.FREEZE_CLIP:
                    optimizer = torch.optim.AdamW(
                    [
                        {"params": model.context_proj.parameters(), "lr": config.TRAIN.FUSION_LR},
                        {"params": model.fusion_attn.parameters(), "lr": config.TRAIN.FUSION_LR},
                        {"params": model.fusion_token, "lr": config.TRAIN.FUSION_LR},
                        {"params": model.fusion_token_proj.parameters(), "lr": config.TRAIN.FUSION_LR},
                        {"params": model.proj.parameters(), "lr": config.TRAIN.FUSION_LR},
                        {"params": criterion.logit_scale, "lr": config.TRAIN.TEMP_LR},
                    ],
                    lr=config.TRAIN.FUSION_LR,
                    weight_decay=0.1,
                    betas=(0.9, 0.98),
                )
            else:
                optimizer = torch.optim.AdamW(
                    [
                        {"params": model.clip.parameters()},
                        {"params": model.context_proj.parameters(), "lr": config.TRAIN.FUSION_LR},
                        {"params": model.fusion_attn.parameters(), "lr": config.TRAIN.FUSION_LR},
                        {"params": model.fusion_token, "lr": config.TRAIN.FUSION_LR},
                        {"params": model.fusion_token_proj.parameters(), "lr": config.TRAIN.FUSION_LR},
                        {"params": model.proj.parameters(), "lr": config.TRAIN.FUSION_LR},
                        {"params": criterion.logit_scale, "lr": config.TRAIN.TEMP_LR},
                    ],
                    lr=config.TRAIN.LR,
                    weight_decay=0.1,
                    betas=(0.9, 0.98),
                )
        else:
            optimizer = torch.optim.AdamW(
                [
                    {"params": model.parameters()},
                    {"params": criterion.logit_scale, "lr": config.TRAIN.TEMP_LR},
                ],
                lr=config.TRAIN.LR,
                weight_decay=0.1,
                betas=(0.9, 0.98),
            )
    elif loss_type == "ReCLIP":
        with open(config.TRAIN.MATERIALS_PATH, "r") as file:
            materials = json.load(file)
            class_weights = torch.tensor(materials["weights"]).to(device)
        lambda_ce = config.TRAIN.LAMBDA_CE
        t = config.TRAIN.TEMPERATURE
        criterion = ReCLIPLoss(class_weights, lambda_ce, t, log_wandb=config.TRAIN.WANDB)

        optimizer = torch.optim.AdamW(
            [
                {"params": model.clip.parameters()},
                {"params": model.mlp.parameters(), "lr": config.TRAIN.MLP_LR},
                {"params": criterion.logit_scale, "lr": config.TRAIN.TEMP_LR},
            ],
            lr=config.TRAIN.LR,
            weight_decay=0.1,
            betas=(0.9, 0.98),
        )

    elif loss_type == "SigLIP":
        t = config.TRAIN.TEMPERATURE
        b = config.TRAIN.BIAS
        criterion = SigLIPLoss(t, b, log_wandb=config.TRAIN.WANDB)
        optimizer = torch.optim.AdamW(
            [
                {"params": model.parameters()},
                {
                    "params": [criterion.t_prime, criterion.bias],
                    "lr": config.TRAIN.CRIT_LR,
                },
            ],
            lr=config.TRAIN.LR,
            weight_decay=0.1,
            betas=(0.9, 0.98),
        )
    else:
        raise NotImplementedError(f"Loss {loss_type} is not implemented.")

    return model, preprocess, criterion, optimizer
