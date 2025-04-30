from argparse import ArgumentParser
import json

import torch
from torch.utils.data import DataLoader

from datasets.materials_dataset import MaterialsDataset
from engine import evaluate_fusion_lazy
from modelling import LFCLIP
from utils import set_seed, parse_config

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

    exp_name = config.EXPERIMENT_NAME
    model_name = config.MODEL.CLIP_BACKBONE
    val_batch_size = 64

    ckpt_metric = "mrr"  # mcs, mrr, recall_1, recall_5, recall_10
    ckpt_path = f"training_results/{exp_name}/{model_name.replace('/', '_')}_best_{ckpt_metric}.pth"
    add_materials_prefix = config.TRAIN.ADD_MATERIALS_PREFIX

    model = LFCLIP(
            clip_model_name=model_name,
            num_heads=config.MODEL.FUSION_HEADS,
            mode="train",
            device=device,
        )

    preprocess = model.preprocess

    model.load_state_dict(torch.load(ckpt_path, weights_only=True))

    eval_dataset = MaterialsDataset(
        images_dir=config.EVAL.IMAGES_PATH,
        captions=config.EVAL.CAPTION_PATH,
        embeddings_dir=config.EVAL.EMBEDDINGS_PATH,
        add_materials_prefix=add_materials_prefix,
        preprocess=preprocess
    )

    eval_loader = DataLoader(
        eval_dataset,
        batch_size=val_batch_size,
        shuffle=True,
        num_workers=4,
        drop_last=False,
    )

    print(f"Evaluate {exp_name} with {ckpt_metric} checkpoint selection.")
    eval_metrics, mean_batch_time = evaluate_fusion_lazy(model, eval_loader, device=device)
    for mname, mvalue in eval_metrics.items():
        print(f"{mname}:\t{mvalue}")

    print(
        f"Mean inference time for image and text: {mean_batch_time / val_batch_size:.3f} s."
    )


if __name__ == "__main__":
    main()
