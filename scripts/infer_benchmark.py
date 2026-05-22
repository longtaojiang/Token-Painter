import argparse
import json
from pathlib import Path

from PIL import Image

from inpaint_utils import (
    DEFAULT_NEGATIVE_PROMPT,
    build_cond_params,
    image_names,
    load_pipeline,
    make_contact_sheet,
    overlay_mask_on_image,
    prepare_image_and_mask,
    save_image,
    set_cache_dir,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Run Token Painter on an inpainting benchmark.")
    parser.add_argument("--model", default="BAAI/nova-d48w1024-sdxl1024")
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--mask_dir", required=True)
    parser.add_argument("--prompt_file", required=True, help="JSON mapping image names to captions.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--precision", default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_inference_steps", type=int, default=64)
    parser.add_argument("--num_diffusion_steps", type=int, default=25)
    parser.add_argument("--guidance_scale", type=float, default=9.0)
    parser.add_argument("--envratio", type=float, default=1.0)
    parser.add_argument("--negative_prompt", default=DEFAULT_NEGATIVE_PROMPT)
    parser.add_argument("--fusion_func", default="gaussian_controlled")
    parser.add_argument("--sigma", type=float, default=250)
    parser.add_argument("--k", type=float, default=6.0)
    parser.add_argument("--condweight", type=float, default=0.8)
    parser.add_argument("--reverse_fusion", action="store_true")
    parser.add_argument("--no_adain", action="store_true")
    parser.add_argument("--no_adae", action="store_true")
    parser.add_argument("--no_preserve_original_size", action="store_true")
    parser.add_argument("--cpu_offload", action="store_true")
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument("--limit", type=int, default=None, help="Optional smoke-test sample limit.")
    return parser.parse_args()


def read_prompt(mapping, name):
    item = mapping[name]
    if isinstance(item, str):
        return item
    return item["caption"]


def main():
    args = parse_args()
    set_cache_dir(args.cache_dir)

    import torch
    from tqdm.auto import tqdm

    output_dir = Path(args.output_dir)
    inpaint_dir = output_dir / "inpaint"
    image_out_dir = output_dir / "image"
    mask_out_dir = output_dir / "mask"
    concat_dir = output_dir / "concat"
    for path in [inpaint_dir, image_out_dir, mask_out_dir, concat_dir]:
        path.mkdir(parents=True, exist_ok=True)

    with open(args.prompt_file, "r", encoding="utf-8") as f:
        prompt_mapping = json.load(f)

    pipe = load_pipeline(args.model, args.device, args.precision, args.cpu_offload)
    device = getattr(pipe, "_offload_device", pipe.device)
    generator = torch.Generator(device=device).manual_seed(args.seed) if args.seed is not None else None
    target_size = pipe.transformer.config.image_size
    cond_params = build_cond_params(args)

    names = [name for name in image_names(args.image_dir) if name in prompt_mapping]
    if args.limit is not None:
        names = names[: args.limit]

    results = []
    for name in tqdm(names, desc="Token Painter inference"):
        output_path = inpaint_dir / name
        if output_path.exists():
            results.append({"image": name, "status": "skipped"})
            continue

        original_image = Image.open(Path(args.image_dir) / name).convert("RGB")
        original_mask = Image.open(Path(args.mask_dir) / name).convert("L")
        prompt = read_prompt(prompt_mapping, name)
        image, mask, restore = prepare_image_and_mask(
            original_image,
            original_mask,
            target_size=target_size,
            preserve_original_size=not args.no_preserve_original_size,
        )

        result = pipe(
            origin_image=image,
            mask=mask,
            prompt=prompt,
            negative_prompt=args.negative_prompt,
            generator=generator,
            guidance_scale=args.guidance_scale,
            envratio=args.envratio,
            cond_para=cond_params,
            atten_entro=not args.no_adae,
            num_inference_steps=args.num_inference_steps,
            num_diffusion_steps=args.num_diffusion_steps,
        ).images[0]
        result = restore(result)

        save_image(result, output_path)
        save_image(original_image, image_out_dir / name)
        save_image(original_mask, mask_out_dir / name)
        save_image(
            make_contact_sheet([original_image, original_mask.convert("RGB"), overlay_mask_on_image(original_image, original_mask), result]),
            concat_dir / name,
        )
        results.append({"image": name, "prompt": prompt, "status": "done"})

    with open(output_dir / "result.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Saved benchmark outputs to {output_dir}")


if __name__ == "__main__":
    main()
