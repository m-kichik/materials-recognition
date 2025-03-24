import json
import logging
import os

import clip

import torch
from torch.utils.data import DataLoader

import wandb

from datasets.materials_dataset import MaterialsDataset
from engine import train, train_iterations, evaluate
from utils import set_seed

set_seed(0)

if torch.cuda.is_available():
    device = torch.device("cuda:0")
elif torch.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


def main():
    train_batch_size = 256
    val_batch_size = 64
    exp_name = "clip-BIG-aug-cap-blur-no-small-lr-1e-6"
    model_name = "ViT-B/32"
    lr = 1e-6
    # n_epochs = 20
    # eval_interval = 1
    n_iters = 100000
    eval_interval = 1000
    # log_wandb = False
    log_wandb = True

    if log_wandb:
        wandb_config = {
            "lr": lr,
            "train_batch_size": train_batch_size,
            "val_batch_size": val_batch_size,
        }

        wandb.init(
            project="MATERIALS",
            name=exp_name,
            config=wandb_config,
        )

    save_dir = exp_name

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    log_filename = f"{save_dir}/metrics.log"
    log_format = "%(asctime)s - %(levelname)s - %(message)s"
    logging.basicConfig(filename=log_filename, level=logging.INFO, format=log_format)

    model, preprocess = clip.load(model_name, device=device)
    model = model.to(torch.float32)

    train_images_path = "/home/docker_user/datasets/train2017_cropped_blurred"
    val_images_path = "/home/docker_user/datasets/val2017_cropped_blurred"

    with open(
        "/home/docker_user/datasets/captions_augmented_train_no_small_final.json", "r"
    ) as f:
        train_data = json.load(f)

    with open(
        "/home/docker_user/datasets/captions_augmented_val_no_small_final.json", "r"
    ) as f:
        val_data = json.load(f)

    train_dataset = MaterialsDataset(train_images_path, train_data, preprocess)
    eval_dataset = MaterialsDataset(val_images_path, val_data, preprocess)
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=1,
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
        eval_interval=eval_interval,
        device=device,
        model_name=model_name,
        save_path=save_dir,
    )


if __name__ == "__main__":
    main()
