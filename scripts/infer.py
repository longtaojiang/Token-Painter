import argparse
from pathlib import Path

from PIL import Image

from inpaint_utils import (
    DEFAULT_NEGATIVE_PROMPT,
    build_cond_params,
    load_pipeline,
    prepare_image_and_mask,
    save_image,
    set_cache_dir,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Run Token Painter on a single image.")
    parser.add_argument("--model", default="BAAI/nova-d48w1024-sdxl1024")
    parser.add_argument("--image", required=True, help="Path to the input RGB image.")
    parser.add_argument("--mask", required=True, help="Path to the binary mask. White pixels are inpainted.")
    parser.add_argument("--prompt", required=True, help="Text prompt for the masked region.")
    parser.add_argument("--negative_prompt", default=DEFAULT_NEGATIVE_PROMPT)
    parser.add_argument("--output", default="outputs/inpaint.png")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--precision", default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--seed", type=int, default=1237)
    parser.add_argument("--num_inference_steps", type=int, default=64)
    parser.add_argument("--num_diffusion_steps", type=int, default=25)
    parser.add_argument("--guidance_scale", type=float, default=5.0)
    parser.add_argument("--envratio", type=float, default=1.0)
    parser.add_argument("--fusion_func", default="gaussian_controlled")
    parser.add_argument("--sigma", type=float, default=250)
    parser.add_argument("--k", type=float, default=6.0)
    parser.add_argument("--condweight", type=float, default=0.8)
    parser.add_argument("--reverse_fusion", action="store_true")
    parser.add_argument("--no_adain", action="store_true")
    parser.add_argument("--no_adae", action="store_true", help="Disable ADAE attention score enhancing.")
    parser.add_argument("--no_preserve_original_size", action="store_true")
    parser.add_argument("--cpu_offload", action="store_true")
    parser.add_argument("--cache_dir", default=None, help="Optional Hugging Face cache directory.")
    return parser.parse_args()


def main():
    args = parse_args()
    set_cache_dir(args.cache_dir)

    import torch

    pipe = load_pipeline(args.model, args.device, args.precision, args.cpu_offload)
    image = Image.open(args.image)
    mask = Image.open(args.mask)
    target_size = pipe.transformer.config.image_size
    image, mask, restore = prepare_image_and_mask(
        image,
        mask,
        target_size=target_size,
        preserve_original_size=not args.no_preserve_original_size,
    )

    device = getattr(pipe, "_offload_device", pipe.device)
    generator = torch.Generator(device=device).manual_seed(args.seed) if args.seed is not None else None
    result = pipe(
        origin_image=image,
        mask=mask,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        generator=generator,
        guidance_scale=args.guidance_scale,
        envratio=args.envratio,
        cond_para=build_cond_params(args),
        atten_entro=not args.no_adae,
        num_inference_steps=args.num_inference_steps,
        num_diffusion_steps=args.num_diffusion_steps,
    ).images[0]

    output_path = save_image(restore(result), args.output)
    print(f"Saved result to {Path(output_path)}")


if __name__ == "__main__":
    main()
