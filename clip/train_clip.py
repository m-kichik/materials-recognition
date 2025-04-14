from argparse import ArgumentParser
import json
import logging
import os

# import clip

import torch
from torch.utils.data import DataLoader

import wandb

from datasets.materials_dataset import MaterialsDataset
from engine import train, train_iterations, evaluate
from utils import build_clip, set_seed, parse_config

set_seed(0)


def define_device(suggested_device: str = "cpu"):
    if suggested_device.startswith("cuda") and torch.cuda.is_available():
        device = torch.device(suggested_device)
    elif suggested_device == "mps" and torch.mps.is_available():
        device = torch.device(suggested_device)
    else:
        device = torch.device("cpu")

    return device


def get_args():
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, help="path to config file")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="device to run the experiment",
    )
    return parser.parse_args()


def main():
    args = get_args()
    config = parse_config(args.config)
    device = define_device(args.device)

    train_batch_size = config.TRAIN.BATCH_SIZE
    val_batch_size = config.EVAL.BATCH_SIZE
    exp_name = config.EXPERIMENT_NAME
    model_name = config.MODEL.BACKBONE

    pretrained = config.TRAIN.PRETRAINED

    lr = config.TRAIN.LR
    clip_lr = config.TRAIN.CLIP_LR
    warmup_fraction = config.TRAIN.WARMUP
    accumulation_steps = config.TRAIN.GRADIENT_ACCUMULATION_STEPS
    n_iters = config.TRAIN.ITERS
    eval_interval = config.TRAIN.EVAL_INTERVAL
    freeze_text = config.TRAIN.FREEZE_TEXT
    add_materials_prefix = config.TRAIN.ADD_MATERIALS_PREFIX
    log_wandb = config.TRAIN.WANDB

    if log_wandb:
        wandb_config = {
            "lr": lr,
            "train_batch_size": train_batch_size,
            "val_batch_size": val_batch_size,
            "clip_lr": clip_lr,
            "warmup_fraction": warmup_fraction,
            "accumulation_steps": accumulation_steps,
            "freeze_text": freeze_text,
            "add_materials_prefix": add_materials_prefix
        }

        wandb.init(
            project=config.PROJECT_NAME,
            name=exp_name,
            config=wandb_config,
        )

    save_dir = f"training_results/{exp_name}"

    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    log_filename = f"{save_dir}/metrics.log"
    log_format = "%(asctime)s - %(levelname)s - %(message)s"
    logging.basicConfig(filename=log_filename, level=logging.INFO, format=log_format)

    # model, preprocess = clip.load(model_name, device=device)
    # model = model.to(torch.float32)
    model, preprocess = build_clip(model_name, pretrained=pretrained, device=device)

    train_images_path = config.TRAIN.IMAGES_PATH
    val_images_path = config.EVAL.IMAGES_PATH

    with open(
        config.TRAIN.CAPTIONS_PATH,
        "r",
    ) as f:
        train_data = json.load(f)

    with open(
        config.EVAL.CAPTIONS_PATH,
        "r",
    ) as f:
        val_data = json.load(f)

    train_dataset = MaterialsDataset(
        train_images_path,
        train_data,
        add_materials_prefix=add_materials_prefix,
        preprocess=preprocess,
    )
    eval_dataset = MaterialsDataset(
        val_images_path,
        val_data,
        add_materials_prefix=add_materials_prefix,
        preprocess=preprocess,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=4,
        drop_last=True,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=val_batch_size,
        shuffle=True,
        num_workers=4,
        drop_last=True,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.98))

    clip_metrics, _ = evaluate(model, eval_loader, device=device)
    clip_metrics = {"pretrain/" + k: v for k, v in clip_metrics.items()}
    if wandb.run is not None:
        wandb.log(clip_metrics)

    # train(
    #     model,
    #     optimizer,
    #     train_loader,
    #     eval_loader,
    #     n_epochs=n_epochs,
    #     eval_interval=eval_interval,
    #     device=device,
    #     model_name=model_name,
    #     save_path=save_dir,
    # )

    train_iterations(
        model,
        optimizer,
        train_loader,
        eval_loader,
        n_iterations=n_iters,
        clip_lr=clip_lr,
        warmup_fraction=warmup_fraction,
        eval_interval=eval_interval,
        accumulation_steps=accumulation_steps,
        device=device,
        model_name=model_name,
        freeze_text=freeze_text,
        save_path=save_dir,
    )


if __name__ == "__main__":
    main()
