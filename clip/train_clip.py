import json
import logging
import os
import random

import clip
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

import wandb

from datasets.materials_dataset import MaterialsDataset
from engine import train, evaluate


random.seed(0)

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


def main():
    batch_size = 64
    exp_name = "clip-aug-cap-blur-no-small-lr-1e-6"
    model_name = "ViT-B/32"
    lr = 1e-6
    n_epochs = 20
    eval_interval = 1
    # log_wandb = False
    log_wandb = True

    if log_wandb:
        wandb_config = {
            "lr": lr,
            "batch_size": batch_size,
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

    # data_path = "val2017_cropped"
    data_path = "val2017_cropped_blurred"

    # with open("captions_val_no_small_final.json", "r") as f:
    with open("captions_augmented_val_no_small_final.json", "r") as f:
        # with open("captions_augmented_final.json", "r") as f:
        # with open("captions_val_final.json", "r") as f:
        data = json.load(f)

    # with open("captions_augmented_val_no_small_final.json", "r") as f:
    #     data_ = json.load(f)

    # data_all = []
    # for idx in range(len(data)):
    #     data_all.append(
    #         {
    #             "image": data[idx]["image"],
    #             "caption": [data[idx]["caption"], data_[idx]["caption"]],
    #         }
    #     )

    # data = data_all

    random.shuffle(data)
    sep_idx = len(data) * 4 // 5

    train_dataset = MaterialsDataset(data_path, data[:sep_idx], preprocess)
    eval_dataset = MaterialsDataset(data_path, data[sep_idx:], preprocess)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.98))

    clip_metrics = evaluate(model, eval_loader, device=device)
    clip_metrics = {"pretrain/" + k: v for k, v in clip_metrics.items()}
    if wandb.run is not None:
        wandb.log(clip_metrics)

    train(
        model,
        optimizer,
        train_loader,
        eval_loader,
        n_epochs=n_epochs,
        eval_interval=eval_interval,
        device=device,
        model_name=model_name,
        save_path=save_dir,
    )


if __name__ == "__main__":
    main()
