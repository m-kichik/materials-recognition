import torch

def build_clip(
    model_name: str = "ViT-B/32", pretrained: bool = False, device: str = "cpu"
):
    if pretrained:
        import clip

        model, preprocess = clip.load(model_name, device=device)
    else:
        import open_clip

        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name, device=device, pretrained=None
        )
    model = model.to(torch.float32)
    return model, preprocess
