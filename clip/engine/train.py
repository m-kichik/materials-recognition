"""CLIP train"""

import logging
import random

import torch
from tqdm import tqdm
import wandb

import clip

from .evaluation import evaluate
from .loss import clip_loss
from utils import save_ckpt


def train(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader: torch.utils.data.DataLoader,
    eval_loader: torch.utils.data.DataLoader,
    n_epochs: int = 10,
    eval_interval: int = 5,
    device: str = "cpu",
    model_name: str = "",
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
            text_features = model.encode_text(text_tokens)

            loss = clip_loss(image_features, text_features)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss = loss.item()
            pbar.set_description("epoch {}, loss: {}".format(epoch, loss))
            total_loss += loss

        log_dict = {"train/loss": total_loss / len(train_loader)}

        if (epoch + 1) % eval_interval == 0:
            eval_metrics = evaluate(model, eval_loader)

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

            eval_metrics = {"eval/" + k: v for k, v in eval_metrics.items()}

            log_dict.update(eval_metrics)

            log_msg = "Metrics - " + f"Epoch {epoch}: "
            log_msg += " | ".join(
                [f"{key}: {value:.4f}" for key, value in eval_metrics.items()]
            )

            logging.info(log_msg)

        if wandb.run is not None:
            wandb.log(log_dict)
