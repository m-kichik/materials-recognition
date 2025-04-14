"""CLIP train"""

import logging
import math
import random

import torch
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
import wandb

import clip

from .evaluation import evaluate
from .loss import clip_loss
from utils import save_ckpt


def get_cosine_with_warmup_scheduler(optimizer, num_warmup_steps, num_training_steps):
    """
    Returns a LambdaLR scheduler with a linear warmup phase and a cosine annealing decay.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            # Linear warmup
            return float(current_step) / float(max(1, num_warmup_steps))
        else:
            # Cosine decay
            progress = float(current_step - num_warmup_steps) / float(
                max(1, num_training_steps - num_warmup_steps)
            )
            return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda)


def train(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader: torch.utils.data.DataLoader,
    eval_loader: torch.utils.data.DataLoader,
    n_epochs: int = 10,
    eval_interval: int = 5,
    device: str = "cpu",
    model_name: str = "",
    freeze_text: bool = False,
    save_path: str = "clip_train",
) -> None:
    best_metrics = {
        "recall": 0.0,
        "mrr": 0.0,
        "mcs": 0.0,
    }

    for epoch in range(n_epochs):
        total_loss = 0
        for images, captions in (pbar := tqdm(train_loader)):
            images = images.to(device)
            if len(captions) == 2:
                captions = random.choice(captions)
            text_tokens = clip.tokenize(captions).to(device)

            image_features = model.encode_image(images)

            if freeze_text:
                with torch.no_grad():
                    text_features = model.encode_text(text_tokens)
            else:
                text_features = model.encode_text(text_tokens)

            loss = clip_loss(image_features, text_features, freeze_text=freeze_text)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss = loss.item()
            pbar.set_description("epoch {}, loss: {}".format(epoch, loss))
            total_loss += loss

        log_dict = {"train/loss": total_loss / len(train_loader)}

        if (epoch + 1) % eval_interval == 0:
            eval_metrics, _ = evaluate(model, eval_loader, device=device)

            save_ckpt(
                model,
                model_name.replace("/", "_"),
                {
                    "recall": eval_metrics["Recall@1"],
                    "mrr": eval_metrics["MRR"],
                    "mcs": eval_metrics["Mean Cosine Similarity"],
                },
                best_metrics,
                best_metrics.keys(),
                save_path,
            )

            log_msg = "Metrics - " + f"Epoch {epoch + 1}: "
            log_msg += " | ".join(
                [f"{key}: {value:.4f}" for key, value in eval_metrics.items()]
            )

            logging.info(log_msg)
            print(log_msg)

            eval_metrics = {"eval/" + k: v for k, v in eval_metrics.items()}

            log_dict.update(eval_metrics)

        if wandb.run is not None:
            wandb.log(log_dict)


def train_iterations(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader: torch.utils.data.DataLoader,
    eval_loader: torch.utils.data.DataLoader,
    n_iterations: int = 10,
    clip_lr: bool = False,
    warmup_fraction: float = 0.1,
    eval_interval: int = 5,
    accumulation_steps: int = 1,
    device: str = "cpu",
    model_name: str = "",
    freeze_text: bool = False,
    save_path: str = "clip_train",
) -> None:
    best_metrics = {
        "recall_1": 0.0,
        "recall_5": 0.0,
        "recall_10": 0.0,
        "mrr": 0.0,
        "mcs": 0.0,
    }

    if clip_lr:
        total_optimizer_steps = math.ceil(n_iterations / accumulation_steps)
        num_warmup_steps = int(total_optimizer_steps * warmup_fraction)
        scheduler = get_cosine_with_warmup_scheduler(
            optimizer, num_warmup_steps, total_optimizer_steps
        )

    # If shuffle=True in Dataloader initial args,
    # random permutations are applied in __iter__ of the loader.
    train_iter = iter(train_loader)
    optimizer.zero_grad()

    for iter_ in (pbar := tqdm(range(n_iterations))):
        model.train()
        try:
            images, captions = next(train_iter)
            images = images.to(device)
            if len(captions) == 2:
                captions = random.choice(captions)
            text_tokens = clip.tokenize(captions).to(device)

            image_features = model.encode_image(images)
            if freeze_text:
                with torch.no_grad():
                    text_features = model.encode_text(text_tokens)
            else:
                text_features = model.encode_text(text_tokens)

            loss = clip_loss(image_features, text_features, freeze_text=freeze_text)
            loss = loss / accumulation_steps

            # optimizer.zero_grad()
            loss.backward()
            # optimizer.step()

            if (iter_ + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                if clip_lr:
                    scheduler.step()

            # loss = loss.item()
            loss_val = loss.item() * accumulation_steps
            pbar.set_description(
                f"loss: {loss_val:.6f}, lr: {scheduler.get_last_lr()[0]:.6e}"
            )

            if wandb.run is not None:
                wandb.log({"train/loss": loss_val})

        except StopIteration:
            train_iter = iter(train_loader)

        if (iter_ + 1) % eval_interval == 0:
            eval_metrics, _ = evaluate(model, eval_loader, device=device)

            save_ckpt(
                model,
                model_name.replace("/", "_"),
                {
                    "recall_1": eval_metrics["Recall@1"],
                    "recall_5": eval_metrics["Recall@5"],
                    "recall_10": eval_metrics["Recall@10"],
                    "mrr": eval_metrics["MRR"],
                    "mcs": eval_metrics["Mean Cosine Similarity"],
                },
                best_metrics,
                best_metrics.keys(),
                save_path,
            )

            log_msg = "Metrics - " + f"Iter {iter_ + 1}: "
            log_msg += " | ".join(
                [f"{key}: {value:.4f}" for key, value in eval_metrics.items()]
            )

            logging.info(log_msg)
            print(log_msg)

            if wandb.run is not None:
                eval_metrics = {"eval/" + k: v for k, v in eval_metrics.items()}
                wandb.log(eval_metrics)

    if (n_iterations % accumulation_steps) != 0:
        optimizer.step()
        optimizer.zero_grad()
