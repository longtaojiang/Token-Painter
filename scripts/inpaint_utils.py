import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter, ImageOps


DEFAULT_NEGATIVE_PROMPT = (
    "low quality, deformed, distorted, disfigured, fused fingers, bad anatomy, weird hand"
)

DEFAULT_COND_PARAMS = {
    "func": "gaussian_controlled",
    "reverse": False,
    "sigma": 250,
    "k": 6.0,
    "condweight": 0.8,
    "ada_in": True,
}


def build_cond_params(args):
    return {
        "func": args.fusion_func,
        "reverse": args.reverse_fusion,
        "sigma": args.sigma,
        "k": args.k,
        "condweight": args.condweight,
        "ada_in": not args.no_adain,
    }


def load_pipeline(model, device, precision="float16", cpu_offload=False):
    import torch
    from diffnext.pipelines import NOVAInpaintPipeline

    model_args = {"torch_dtype": getattr(torch, precision.lower()), "trust_remote_code": True}
    pipe = NOVAInpaintPipeline.from_pretrained(model, **model_args)
    if cpu_offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe = pipe.to(torch.device(device))
    return pipe


def resize_and_pad(image, mask, target_size):
    original_w, original_h = image.size
    if original_w > original_h:
        scale = target_size / original_w
        new_w = target_size
        new_h = int(original_h * scale)
    else:
        scale = target_size / original_h
        new_h = target_size
        new_w = int(original_w * scale)

    image_resized = image.resize((new_w, new_h), Image.Resampling.BILINEAR)
    mask_resized = mask.resize((new_w, new_h), Image.Resampling.BILINEAR)
    pad_w = (target_size - new_w) // 2
    pad_h = (target_size - new_h) // 2
    padding = (pad_w, pad_h, target_size - new_w - pad_w, target_size - new_h - pad_h)
    image_padded = ImageOps.expand(image_resized, padding, (255, 255, 255))
    mask_padded = ImageOps.expand(mask_resized, padding, 0)
    return image_padded, mask_padded, (original_w, original_h), (scale, pad_w, pad_h)


def unpad_and_resize_back(image_padded, original_size, original_mask, original_image, params):
    original_w, original_h = original_size
    scale, pad_w, pad_h = params
    crop = image_padded.crop(
        (pad_w, pad_h, pad_w + int(original_w * scale), pad_h + int(original_h * scale))
    )
    restored = crop.resize((original_w, original_h), Image.Resampling.BILINEAR)
    soft_mask = original_mask.filter(ImageFilter.GaussianBlur(radius=5))
    return Image.composite(restored, original_image, soft_mask).convert("RGB")


def resize_and_crop(image, mask, target_size):
    original_w, original_h = image.size
    scale = target_size / min(original_w, original_h)
    new_w, new_h = int(original_w * scale), int(original_h * scale)
    image_resized = image.resize((new_w, new_h), Image.Resampling.BILINEAR)
    mask_resized = mask.resize((new_w, new_h), Image.Resampling.BILINEAR)

    mask_np = np.array(mask_resized)
    y_indices, x_indices = np.where(mask_np > 0)
    if len(x_indices) == 0:
        raise ValueError("Mask does not contain any non-zero pixels.")
    min_x, max_x = np.min(x_indices), np.max(x_indices)
    min_y, max_y = np.min(y_indices), np.max(y_indices)

    crop_x1 = max(0, min(min_x - (target_size - (max_x - min_x)) // 2, new_w - target_size))
    crop_y1 = max(0, min(min_y - (target_size - (max_y - min_y)) // 2, new_h - target_size))
    crop_box = (crop_x1, crop_y1, crop_x1 + target_size, crop_y1 + target_size)
    return image_resized.crop(crop_box), mask_resized.crop(crop_box), (
        original_w,
        original_h,
        scale,
        crop_box,
    )


def restore_crop(image_cropped, params, original_image, original_mask):
    original_w, original_h, scale, crop_box = params
    new_w, new_h = int(original_w * scale), int(original_h * scale)
    canvas = original_image.resize((new_w, new_h), Image.Resampling.BILINEAR)
    canvas.paste(image_cropped, crop_box[:2])
    soft_mask = original_mask.resize((new_w, new_h), Image.Resampling.BILINEAR)
    soft_mask = soft_mask.filter(ImageFilter.GaussianBlur(radius=5))
    composed = Image.composite(canvas, original_image.resize((new_w, new_h)), soft_mask)
    return composed.resize((original_w, original_h), Image.Resampling.BILINEAR).convert("RGB")


def prepare_image_and_mask(image, mask, target_size=1024, preserve_original_size=True):
    image = image.convert("RGB")
    mask = mask.convert("L")
    if mask.getbbox() is None:
        raise ValueError("Mask does not contain any non-zero pixels.")

    if not preserve_original_size:
        return (
            image.resize((target_size, target_size), Image.Resampling.BILINEAR),
            mask.resize((target_size, target_size), Image.Resampling.NEAREST),
            lambda result: result,
        )

    width, height = image.size
    min_side = min(width, height)
    mask_box = mask.getbbox()
    mask_max_side = max(mask_box[2] - mask_box[0], mask_box[3] - mask_box[1])
    if mask_max_side > min_side:
        processed_image, processed_mask, original_size, params = resize_and_pad(image, mask, target_size)
        restore = lambda result: unpad_and_resize_back(result, original_size, mask, image, params)
    else:
        processed_image, processed_mask, params = resize_and_crop(image, mask, target_size)
        restore = lambda result: restore_crop(result, params, image, mask)
    return processed_image, processed_mask, restore


def save_image(image, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    return path


def image_names(image_dir):
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sorted(p.name for p in Path(image_dir).iterdir() if p.suffix.lower() in valid_exts)


def overlay_mask_on_image(image, mask, color=(255, 0, 0), alpha=0.35):
    mask = mask.convert("L")
    color_layer = Image.new("RGB", mask.size, color)
    mask_rgba = Image.composite(color_layer, Image.new("RGB", mask.size), mask).convert("RGBA")
    mask_rgba.putalpha(int(alpha * 255))
    result = Image.alpha_composite(image.convert("RGBA"), mask_rgba)
    return Image.composite(result, image.convert("RGBA"), mask).convert("RGB")


def make_contact_sheet(images):
    widths, heights = zip(*(img.size for img in images))
    total_width = sum(widths)
    max_height = max(heights)
    sheet = Image.new("RGB", (total_width, max_height), (255, 255, 255))
    x_offset = 0
    for image in images:
        sheet.paste(image.convert("RGB"), (x_offset, 0))
        x_offset += image.size[0]
    return sheet


def set_cache_dir(cache_dir):
    if cache_dir:
        os.environ["HF_HOME"] = str(cache_dir)
