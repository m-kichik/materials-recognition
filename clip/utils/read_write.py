"""Simple functions for read/write jsons, checkpoints, etc"""

import torch


def save_ckpt(model, model_name, metrics, best_metrics, keys, save_dir="."):
    for k in keys:
        if metrics[k] > best_metrics[k]:
            torch.save(model.state_dict(), f"{save_dir}/{model_name}_best_{k}.pth")
            best_metrics[k] = metrics[k]
