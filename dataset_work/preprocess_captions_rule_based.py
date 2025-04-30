import json
import os
import re

from tqdm import tqdm

patterns = [
    re.compile(r' with a bright numeric ID(?:\s+(?:\d+|of\s+\d+|is\s+\d+))?'),
    re.compile(r' with bright ID(?:\s+(?:\d+|of\s+\d+|is\s+\d+))?'),
    re.compile(r' with a bright ID(?:\s+(?:\d+|of\s+\d+|is\s+\d+))?'),
    re.compile(r' with bright numeric ID(?:\s+(?:\d+|of\s+\d+|is\s+\d+))?'),
    re.compile(r' with a numeric ID(?:\s+(?:\d+|of\s+\d+|is\s+\d+))?'),
    re.compile(r' with numeric ID(?:\s+(?:\d+|of\s+\d+|is\s+\d+))?'),
    re.compile(r' with the numeric ID(?:\s+(?:\d+|of\s+\d+|is\s+\d+))?'),
    re.compile(r' with a bright white ID(?:\s+(?:\d+|of\s+\d+|is\s+\d+))?'),
    re.compile(r' with a bright red numeric ID(?:\s+(?:\d+|of\s+\d+|is\s+\d+))?'),
    re.compile(r' with a white numeric ID(?:\s+(?:\d+|of\s+\d+|is\s+\d+))?'),
    re.compile(r' with a bright white numeric ID(?:\s+(?:\d+|of\s+\d+|is\s+\d+))?'),
    re.compile(r' with bright white numeric ID(?:\s+(?:\d+|of\s+\d+|is\s+\d+))?'),
    re.compile(r' and has a bright numeric ID(?:\s+(?:\d+|of\s+\d+|is\s+\d+))?'),
    re.compile(r' and a bright numeric ID(?:\s+(?:\d+|of\s+\d+|is\s+\d+))?'),
    re.compile(r' has a bright numeric ID(?:\s+(?:\d+|of\s+\d+|is\s+\d+))?'),
    re.compile(r' and bright numeric ID(?:\s+(?:\d+|of\s+\d+|is\s+\d+))?'),
    re.compile(r' with a bright outline and numeric ID(?:\s+(?:\d+|of\s+\d+|is\s+\d+))?'),
    re.compile(r' with bright outline and numeric ID(?:\s+(?:\d+|of\s+\d+|is\s+\d+))?'),
    re.compile(r' with yellow outline and numeric ID(?:\s+(?:\d+|of\s+\d+|is\s+\d+))?'),
    re.compile(r'Bright numeric ID tag(?:\s+(?:\d+|of\s+\d+|is\s+\d+))?\s*on '),
    re.compile(r'A bright numeric ID(?:\s+(?:\d+|of\s+\d+|is\s+\d+))?\s*on '),
    re.compile(r' and ID'),
    re.compile(r', (?:\s+(?:\d+|of\s+\d+|is\s+\d+))?\s*ID'),
    re.compile(r', (?:\s+(?:\d+|of\s+\d+|is\s+\d+))?\s*ID'),
    re.compile(r' with a numeric label(?:\s+(?:\d+|of\s+\d+|is\s+\d+))'),
    re.compile(r' with a bright numeric label(?:\s+(?:\d+|of\s+\d+|is\s+\d+))'),
]

def fix_caption(caption):
    print(caption)
    for p in patterns:
        caption = re.sub(p, "", caption)
    return caption

def main():
    # split = "val"
    split = "train"
    path = f"/home/m-kichik/Desktop/work/datasets/LVIS/base_anno/first_iter_vlm/{split}"

    for fname in tqdm(os.listdir(path)):
        if fname.startswith("prompt"):
            continue

        with open(f"{path}/{fname}", "r") as file:
            content = file.read()
            if not content:
                continue
            data = json.loads(content)

        for label in data:
            if "ID" in label["caption"]:
                label["caption"] = fix_caption(label["caption"])

        with open(f"{path}/{fname}", "w") as file:
            json.dump(data, file, indent=4)

if __name__ == "__main__":
    main()
