import json
import os

def main():
    split = "val"
    path = f"/home/m-kichik/Desktop/work/datasets/LVIS/base_anno/first_iter_vlm/{split}_gpt-4.1-mini"
    out_path = f"/home/m-kichik/Desktop/work/datasets/LVIS/base_anno/first_iter_vlm/{split}_gpt-4.1-mini_postproc"
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    with open("updated_categories_1.json", "r") as f:
        cats = json.load(f)

    c = 0
    for fname in os.listdir(path):
        with open(os.path.join(path, fname), "r") as file:
            data = json.load(file)
        
        for item in data:
            if (
                item["new_caption"] is None
                and cats[item["category"]] != "others"
                and not "ID" in item["caption"]
                and not "label" in item["caption"]
                and not "numeric" in item["caption"]
                ):
                c += 1
                item["new_caption"] = item["caption"]
                item["reason"] = "non-material"

        with open(os.path.join(out_path, fname), "w") as file:
            json.dump(data, file, indent=4)

    print("Total: ",c)

if __name__ == "__main__":
    main()