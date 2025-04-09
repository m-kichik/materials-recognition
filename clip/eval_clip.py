import json
import time

import clip

import torch
from torch.utils.data import DataLoader

from datasets.materials_dataset import MaterialsDataset
from engine import evaluate
from utils import set_seed

set_seed(0)

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


def main():
    val_batch_size = 64
    # exp_name = "clip-BIG-aug-cap-blur-no-small-lr-1e-6"
    # exp_name = "clip-BIG-freeze-text-blur-no-small-lr-1e-6"
    # exp_name = "clip-BIG-freeze-text-only-materials-blur-no-small-lr-1e-6"
    # exp_name = "clip-BIG-only-materials-blur-no-small-lr-1e-6"
    # exp_name = "clip-BIG-only-materials-blur-no-small-small-batch-lr-1e-6"
    # exp_name = "clip-BIG-freeze-text-only-materials-blur-no-small-small-batch-lr-1e-6"
    exp_name = "clip-BIG-freeze-text-only-materials-smart-blur-no-small-small-batch-lr-1e-6"
    # exp_name = "clip-RN50x4-freeze-text-only-materials-smart-blur-no-small-small-batch-lr-1e-6"
    model_name = "ViT-B/32"
    ckpt_metric = "mcs"  # mcs, mrr, recall_1, recall_5, recall_10
    ckpt_path = f"{exp_name}/{model_name.replace('/', '_')}_best_{ckpt_metric}.pth"
    add_materials_prefix = True

    model, preprocess = clip.load(model_name, device=device)
    model = model.to(torch.float32)

    model.load_state_dict(torch.load(ckpt_path, weights_only=True))

    val_images_path = "/home/docker_user/work/clip/data/val2017_cropped_blurred"

    with open(
        # "/home/docker_user/datasets/captions_augmented_val_no_small_final.json", "r"
        "/home/docker_user/work/clip/data/captions_material_val_no_small_final.json",
        "r",
    ) as f:
        val_data = json.load(f)

    eval_dataset = MaterialsDataset(val_images_path, val_data, add_materials_prefix=add_materials_prefix, preprocess=preprocess)
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=val_batch_size,
        shuffle=True,
        num_workers=4,
        drop_last=False,
    )

    print(f"Evaluate {exp_name} with {ckpt_metric} checkpoint selection.")
    eval_metrics, mean_batch_time = evaluate(model, eval_loader, device=device)
    for mname, mvalue in eval_metrics.items():
        print(f"{mname}:\t{mvalue}")

    print(
        f"Mean inference time for image and text: {mean_batch_time / val_batch_size:.3f} s."
    )


if __name__ == "__main__":
    main()
