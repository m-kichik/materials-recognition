from argparse import ArgumentParser

import torch
from torch.utils.data import DataLoader

from engine import evaluate_embeddings
from utils import set_seed, parse_config, build_dataset, build_model

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
    model_name = config.MODEL.BACKBONE
    val_batch_size = 64

    ckpt_metric = "mrr"  # mcs, mrr, recall_1, recall_5, recall_10
    ckpt_path = f"training_results/{exp_name}/{model_name.replace('/', '_')}_best_{ckpt_metric}.pth"
    add_materials_prefix = config.ADD_MATERIALS_PREFIX
    if add_materials_prefix is None and config.TRAIN is not None:
        add_materials_prefix = config.TRAIN.ADD_MATERIALS_PREFIX

    model, preprocess = build_model(config, device=device)
    return

    if not config.PRETRAINED:
        model.load_state_dict(torch.load(ckpt_path, weights_only=True))

    eval_dataset = build_dataset(
        config,
        preprocess
    )
    
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=val_batch_size,
        shuffle=True,
        num_workers=4,
        drop_last=False,
    )

    print(f"Evaluate {exp_name} with {ckpt_metric} checkpoint selection.")
    eval_metrics, mean_batch_time = evaluate_embeddings(
        model,
        eval_loader,
        device=device
    )

    for mname, mvalue in eval_metrics.items():
        print(f"{mname}:\t{mvalue}")

    print(
        f"Mean inference time for image and text: {mean_batch_time / val_batch_size:.3f} s."
    )


if __name__ == "__main__":
    main()