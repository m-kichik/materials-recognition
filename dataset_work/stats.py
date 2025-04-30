import json
import os

from tqdm import tqdm

def main():
    split = "val"
    # split = "train"
    # path = f"/home/m-kichik/Desktop/work/datasets/LVIS/base_anno/first_iter_vlm/{split}_gpt-4.1-mini"
    path = f"/home/m-kichik/Desktop/work/datasets/LVIS/base_anno/first_iter_vlm/{split}_gpt-4.1-mini_postproc"

    good = 0
    bad = 0
    fixed = 0

    for fname in tqdm(os.listdir(path)):
        with open(f"{path}/{fname}", "r") as file:
            data = json.load(file)
        for label in data:
            if label["is_good"]:
                good += 1
                if "ID" in label["new_caption"]:
                    print(label["new_caption"])
            else:
                bad += 1
                if label["new_caption"] is not None:
                    fixed += 1
                # else:
                #     if "plush" in label["caption"]:
                #         print(label["caption"])
    
    total = good + bad
    still_bad = bad - fixed

    print(f"Total:  {total}")
    print(f"Good:   {good} ({good * 100 / total :.3f} %)")
    print(f"Bad:    {bad} ({bad * 100 / total :.3f} %)")
    print(f"Fixed:  {fixed} ({fixed * 100 / total :.3f} %)")
    print(f"No fix: {still_bad} ({still_bad * 100 / total :.3f} %)")

if __name__ == "__main__":
    main()