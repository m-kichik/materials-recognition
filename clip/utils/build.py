import json
import os
from typing import Callable, Dict, List, Tuple

import torch

from .config import Config
from datasets.materials_dataset import MaterialsDataset
from engine.criterion import (
    vanilla_clip_loss,
    CLIPLoss,
    ReCLIPLoss,
    SigLIPLoss,
    CLIPMatSIM,
    EmbeddingsLoss,
    CombinedLoss
)
from modelling import CLIP, LFCLIP, MLPCLIP, TextEncoder


def build_experiment(
    config,
    device: str = "cpu",
):
    model, preprocess = build_model(config, device=device)

    S_cat, categories_dict = None, None
    if config.TRAIN.BUILD_CATEGORIES:
        S_cat, categories_dict = build_categories(
            [config.TRAIN.CAPTIONS_PATH, config.EVAL.CAPTIONS_PATH],
            key_="category",
            device=device,
        )
    S_mat, materials_dict = None, None
    if config.TRAIN.BUILD_MATERIALS:
        S_mat, materials_dict = build_categories(
            [config.TRAIN.CAPTIONS_PATH, config.EVAL.CAPTIONS_PATH],
            key_="material",
            device=device,
        )
    criterion = build_criterion(
        config,
        S={
            "S_cat": (S_cat.to(device) if S_cat is not None else None),
            "S_mat": (S_mat.to(device) if S_cat is not None else None),
        },
        device=device,
    )
    optimizer = build_optimizer(model, criterion, config)

    train_dataset = build_dataset(
        config.TRAIN,
        preprocess=preprocess,
        categories_dict=categories_dict,
        materials_dict=materials_dict,
    )
    val_dataset = build_dataset(
        config.EVAL,
        preprocess=preprocess,
        categories_dict=categories_dict,
        materials_dict=materials_dict,
    )

    return model, criterion, optimizer, train_dataset, val_dataset


def build_model(config: Config, device: str = "cpu"):
    model_type = config.MODEL.TYPE
    clip_model_name = config.MODEL.CLIP_BACKBONE

    pretrained = (
        config.TRAIN.PRETRAINED
        if (config.TRAIN is not None and config.TRAIN.PRETRAINED is not None)
        else config.PRETRAINED
    )
    if pretrained is None:
        pretrained = False

    if model_type in ["vanilla_clip", "vanilla_clip_text", "vanilla_clip_imgs", "clip", "siglip"]:
        model = CLIP(clip_model_name, device=device)
        preprocess = model.preprocess
    elif model_type == "late_fusion_clip":
        model = LFCLIP(
            clip_model_name=clip_model_name,
            freeze_clip=config.MODEL.FREEZE_CLIP,
            clip_ckpt=config.MODEL.CLIP_CKPT,
            fusion_type=config.MODEL.FUSION_TYPE,
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
    elif model_type == "text":
        preprocess = None
        model = TextEncoder(
            model_name=config.MODEL.BACKBONE,
            device=device
        )
    else:
        raise NotImplementedError(f"Model {model_type} is not implemented.")

    if config.MODEL.PRETRAINED_CKPT is not None:
        model.load_state_dict(
            torch.load(config.MODEL.PRETRAINED_CKPT, weights_only=True),
            strict=False
        )

    return model, preprocess


def build_criterion(config: Config, S: Dict = None, device: str = "cpu"):
    loss_type = config.TRAIN.CRITERION

    if loss_type == "vanilla":
        criterion = vanilla_clip_loss

    elif loss_type == "CLIP":
        t = config.TRAIN.TEMPERATURE
        criterion = CLIPLoss(t, log_wandb=config.TRAIN.WANDB)

    elif loss_type == "ReCLIP":
        class_weights = None
        if config.TRAIN.WEIGHT_CE:
            with open(config.TRAIN.MATERIALS_PATH, "r") as file:
                materials = json.load(file)
                class_weights = torch.tensor(materials["weights"]).to(device)
        lambda_ce = config.TRAIN.LAMBDA_CE
        t = config.TRAIN.TEMPERATURE
        criterion = ReCLIPLoss(
            class_weights, lambda_ce, t, log_wandb=config.TRAIN.WANDB
        )

    elif loss_type == "SigLIP":
        t = config.TRAIN.TEMPERATURE
        b = config.TRAIN.BIAS
        criterion = SigLIPLoss(t, b, log_wandb=config.TRAIN.WANDB)

    elif loss_type == "CLIPMatSIM":
        t = config.TRAIN.TEMPERATURE
        clip_loss = CLIPLoss(t, log_wandb=config.TRAIN.WANDB)
        criterion = CLIPMatSIM(
            clip_loss,
            S["S_mat"],
            lambda_=config.TRAIN.LAMBDA,
            log_wandb=config.TRAIN.WANDB,
        )

    elif loss_type == "Embeddings":
        criterion = EmbeddingsLoss(
            S=S,
            alpha=config.TRAIN.ALPHA,
            beta=config.TRAIN.BETA,
            tau=config.TRAIN.TAU,
            tau_cat=config.TRAIN.TAU_CAT,
            tau_mat=config.TRAIN.TAU_MAT,
            gamma=config.TRAIN.GAMMA,
            log_wandb=config.TRAIN.WANDB,
        )
    
    elif loss_type == "Combined":
        t = config.TRAIN.TEMPERATURE
        clip_loss = CLIPLoss(t, log_wandb=config.TRAIN.WANDB)
        tune_img = False
        tune_text = True
        if tune_img:
            image_embeds_loss = EmbeddingsLoss(
                S=S,
                alpha=config.TRAIN.ALPHA,
                beta=config.TRAIN.BETA,
                tau=config.TRAIN.TAU,
                tau_cat=config.TRAIN.TAU_CAT,
                tau_mat=config.TRAIN.TAU_MAT,
                gamma=config.TRAIN.GAMMA,
                mode="image",
                log_wandb=config.TRAIN.WANDB,
            )
        else:
            image_embeds_loss = None
        if tune_text:
            text_embeds_loss = EmbeddingsLoss(
                S=S,
                alpha=config.TRAIN.ALPHA,
                beta=config.TRAIN.BETA,
                tau=config.TRAIN.TAU,
                tau_cat=config.TRAIN.TAU_CAT,
                tau_mat=config.TRAIN.TAU_MAT,
                gamma=config.TRAIN.GAMMA,
                mode="text",
                log_wandb=config.TRAIN.WANDB,
            )
        else:
            text_embeds_loss = None

        if text_embeds_loss is None and image_embeds_loss is None:
            raise RuntimeError("Text loss or Image loss have to be not None.")
            
        criterion = CombinedLoss(
            clip_loss=clip_loss,
            image_embeds_loss=image_embeds_loss,
            text_embeds_loss=text_embeds_loss,
            log_wandb=config.TRAIN.WANDB,
        )
    else:
        raise NotImplementedError(f"Loss {loss_type} is not implemented.")

    return criterion


def build_optimizer(
    model: torch.nn.Module,
    criterion: Callable,
    config: Config,
    weight_decay: float = 0.1,
    betas: Tuple[float] = (0.9, 0.98),
):
    model_type = config.MODEL.TYPE

    param_groups = []
    if model_type in ["vanilla_clip", "siglip"]:
        param_groups.append({"params": model.parameters()})

    elif model_type == "vanilla_clip_text":
        param_groups.append({"params": model.clip.text_model.parameters()})
        param_groups.append({"params": model.clip.text_projection.parameters()})

    elif model_type == "vanilla_clip_imgs":
        param_groups.append({"params": model.clip.vision_model.parameters()})
        param_groups.append({"params": model.clip.visual_projection.parameters()})

    elif model_type == "late_fusion_clip":
        if not config.MODEL.FREEZE_CLIP:
            param_groups.append({"params": model.clip.parameters()})
        param_groups.append(
            {"params": model.fusion.parameters(), "lr": config.TRAIN.FUSION_LR}
        )

    elif model_type == "mlp_clip":
        param_groups.extend(
            [
                {"params": model.clip.parameters()},
                {"params": model.mlp.parameters(), "lr": config.TRAIN.MLP_LR},
            ]
        )

    if isinstance(criterion, torch.nn.Module):
        param_groups.append(
            {"params": criterion.parameters(), "lr": config.TRAIN.TEMP_LR}
        )

    optimizer = torch.optim.AdamW(
        param_groups, lr=config.TRAIN.LR, betas=betas, weight_decay=weight_decay
    )

    return optimizer


def build_dataset(
    config: Config,
    preprocess: Callable,
    categories_dict: Dict = None,
    materials_dict: Dict = None,
):
    dataset = MaterialsDataset(
        config.IMAGES_PATH,
        config.CAPTIONS_PATH,
        captions_key=config.CAPTION_KEY,
        add_materials_prefix=config.ADD_MATERIALS_PREFIX,
        categories=categories_dict,
        materials=materials_dict,
        embeddings_dir=config.EMBEDDINGS_PATH,
        preprocess=preprocess,
        augmentations=config.AUGMENTATIONS,
    )

    return dataset


def build_categories(
    caption_paths: List[str],
    key_: str = "material",
    model_name: str = "all-mpnet-base-v2",
    device="cpu",
):
    if model_name != "all-mpnet-base-v2":
        raise NotImplementedError(
            f"Model {model_name} is not supported for builing embeddings."
        )

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(
        "sentence-transformers/all-mpnet-base-v2", device=device
    )

    all_categories = set()
    for path in caption_paths:
        with open(path, "r") as file:
            captions = json.load(file)
            for item in captions:
                category = item.get(key_)
                if category is not None:
                    all_categories.update(category)

    unique_categories = sorted(list(all_categories))
    if "棉" in unique_categories:
        unique_categories.remove("棉")
    if "泥土" in unique_categories:
        unique_categories.remove("泥土")

    embeddings = model.encode(unique_categories)
    normalized_embeddings = torch.nn.functional.normalize(
        torch.tensor(embeddings), dim=1
    )
    S = normalized_embeddings @ normalized_embeddings.T

    del model
    del embeddings
    del normalized_embeddings

    return S, {item: idx for idx, item in enumerate(unique_categories)}
