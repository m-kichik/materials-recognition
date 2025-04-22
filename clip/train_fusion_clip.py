from argparse import ArgumentParser
import json
import logging
import os

import torch
from torch.utils.data import DataLoader

import wandb

from datasets.materials_dataset import MaterialsDataset
from engine import train_fusion_iterations
from utils import build_experiment, set_seed, parse_config

set_seed(0)


def define_device(requested_device: str = "cpu"):
    if requested_device.startswith("cuda") and torch.cuda.is_available():
        device = torch.device(requested_device)
    elif requested_device == "mps" and torch.mps.is_available():
        device = torch.device(requested_device)
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

    exp_name = config.EXPERIMENT_NAME
    model_name = config.MODEL.CLIP_BACKBONE

    shedule_lr = config.TRAIN.CLIP_LR
    warmup_fraction = config.TRAIN.WARMUP
    accumulation_steps = config.TRAIN.GRADIENT_ACCUMULATION_STEPS
    n_iters = config.TRAIN.ITERS
    eval_interval = config.TRAIN.EVAL_INTERVAL
    freeze_text = config.TRAIN.FREEZE_TEXT
    add_materials_prefix = config.TRAIN.ADD_MATERIALS_PREFIX
    log_wandb = config.TRAIN.WANDB

    if log_wandb:
        wandb_config = {
            "lr": config.TRAIN.LR,
            "train_batch_size": config.TRAIN.BATCH_SIZE,
            "val_batch_size": config.EVAL.BATCH_SIZE,
            "shedule_lr": shedule_lr,
            "warmup_fraction": warmup_fraction,
            "accumulation_steps": accumulation_steps,
            "freeze_text": freeze_text,
            "add_materials_prefix": add_materials_prefix,
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

    model, preprocess, loss, optimizer = build_experiment(config, device=device)

    train_dataset = MaterialsDataset(
        config.TRAIN.IMAGES_PATH,
        config.TRAIN.CAPTIONS_PATH,
        config.TRAIN.EMBEDDINGS_PATH,
        add_materials_prefix=add_materials_prefix,
        preprocess=preprocess
    )
    eval_dataset = MaterialsDataset(
        config.EVAL.IMAGES_PATH,
        config.EVAL.CAPTIONS_PATH,
        config.EVAL.EMBEDDINGS_PATH,
        add_materials_prefix=add_materials_prefix,
        preprocess=preprocess
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.TRAIN.BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        drop_last=True,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=config.EVAL.BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        drop_last=True,
    )

    train_fusion_iterations(
        model,
        loss,
        optimizer,
        train_loader,
        eval_loader,
        n_iterations=n_iters,
        shedule_lr=shedule_lr,
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
