import json
import os

from tqdm import tqdm

from openai import OpenAI

base_prompt = """
You will be given a list of objects and their captions in form of "{id}. {object}: {caption}". Please validate captions. Good caption contain the following:
- Object name (or synonym);
- Material of which the object is made.
If caption is good, please response "{id}. Good caption.". If you can restore missed material from the context or from the object category, update caption by adding material name to it and return "{id}. {new caption}". Otherwise, return "{id}. Bad caption.". Do not return any additional info.
"""

def main():
    split = "val"
    path = f"/Users/rito4ka/dev/diploma/materials-recognition/{split}"
    out_path = f"/Users/rito4ka/dev/diploma/materials-recognition/{split}_deepseek"

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    client = OpenAI()

    for fname in tqdm(os.listdir(path)[:1]):
        if (fname.endswith(".txt") or fname.endswith(".json")) and not fname.startswith("prompt"):
            with open(os.path.join(path, fname), "r") as f:
                data = json.load(f)

            request = ""
            for item in data:
                cap_id = item["id"]
                category = item["category"]
                caption = item["caption"]
                request += f"{cap_id}. {category}: {caption}\n"
            request = request.strip()

            request = base_prompt + request

            response = client.responses.create(
                model="gpt-4.1-mini",
                input=request
            ).output_text.strip().split("\n")

            for idx, res in enumerate(response):
                res = res.strip()
                if "Good caption" in res:
                    data[idx]["is_good"] = True
                    data[idx]["new_caption"] = data[idx]["caption"]
                elif "Bad caption" in res:
                    data[idx]["is_good"] = False
                    data[idx]["new_caption"] = None
                else:
                    data[idx]["response"] = res
                    cap_id, caption, = res.split(". ")
                    if ":" in caption:
                        caption = caption.split(":")[1].strip()
                    data[idx]["is_good"] = False
                    data[idx]["new_caption"] = caption
            
            with open(os.path.join(out_path, fname.replace("txt", "json")), "w") as file:
                json.dump(data, file, indent=4)

if __name__ == "__main__":
    main()