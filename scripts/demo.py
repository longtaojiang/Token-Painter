import argparse

import numpy as np

from inpaint_utils import (
    DEFAULT_NEGATIVE_PROMPT,
    build_cond_params,
    load_pipeline,
    prepare_image_and_mask,
    set_cache_dir,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Launch the Token Painter Gradio demo.")
    parser.add_argument("--model", default="BAAI/nova-d48w1024-sdxl1024")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--precision", default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--server_name", default="0.0.0.0")
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--cpu_offload", action="store_true")
    parser.add_argument("--cache_dir", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    set_cache_dir(args.cache_dir)
    import gradio as gr
    import torch

    pipe = load_pipeline(args.model, args.device, args.precision, args.cpu_offload)
    target_size = pipe.transformer.config.image_size

    def generate(
        image,
        mask,
        prompt,
        negative_prompt,
        seed,
        randomize_seed,
        guidance_scale,
        num_inference_steps,
        num_diffusion_steps,
        envratio,
        fusion_func,
        reverse_fusion,
        sigma,
        k,
        condweight,
        no_adain,
        no_adae,
    ):
        if image is None or mask is None:
            raise gr.Error("Please upload both an image and a mask.")
        seed = int(np.random.randint(0, 2147483647)) if randomize_seed else int(seed)
        image, mask, restore = prepare_image_and_mask(image, mask, target_size=target_size)
        device = getattr(pipe, "_offload_device", pipe.device)
        generator = torch.Generator(device=device).manual_seed(seed)

        args_like = argparse.Namespace(
            fusion_func=fusion_func,
            reverse_fusion=reverse_fusion,
            sigma=sigma,
            k=k,
            condweight=condweight,
            no_adain=no_adain,
        )
        result = pipe(
            origin_image=image,
            mask=mask,
            prompt=prompt,
            negative_prompt=negative_prompt,
            generator=generator,
            guidance_scale=guidance_scale,
            envratio=envratio,
            cond_para=build_cond_params(args_like),
            atten_entro=not no_adae,
            num_inference_steps=int(num_inference_steps),
            num_diffusion_steps=int(num_diffusion_steps),
        ).images[0]
        return restore(result), seed

    with gr.Blocks(title="Token Painter") as app:
        gr.Markdown("# Token Painter")
        with gr.Row():
            with gr.Column():
                image = gr.Image(label="Image", type="pil")
                mask = gr.Image(label="Mask (white region will be inpainted)", type="pil")
                prompt = gr.Textbox(label="Prompt", lines=3, value="a shiba inu wearing a beret and black turtleneck.")
                negative_prompt = gr.Textbox(label="Negative Prompt", lines=2, value=DEFAULT_NEGATIVE_PROMPT)
                run = gr.Button("Generate", variant="primary")
            with gr.Column():
                output = gr.Image(label="Result", type="pil")
                used_seed = gr.Number(label="Seed", precision=0)

        with gr.Accordion("Advanced Options", open=False):
            seed = gr.Slider(label="Seed", minimum=0, maximum=2147483647, step=1, value=1237)
            randomize_seed = gr.Checkbox(label="Randomize seed", value=False)
            guidance_scale = gr.Slider(label="Guidance scale", minimum=1, maximum=12, step=0.1, value=5.0)
            with gr.Row():
                num_inference_steps = gr.Slider(label="Inference steps", minimum=1, maximum=128, step=1, value=64)
                num_diffusion_steps = gr.Slider(label="Diffusion steps", minimum=1, maximum=50, step=1, value=25)
            envratio = gr.Slider(label="Background token ratio", minimum=0, maximum=4, step=0.05, value=1.0)
            fusion_func = gr.Dropdown(
                label="Frequency fusion function",
                choices=["gaussian_controlled", "convex_quad", "gaussian_func", "linear_func", "linear_inter"],
                value="gaussian_controlled",
            )
            reverse_fusion = gr.Checkbox(label="Reverse frequency fusion", value=False)
            sigma = gr.Slider(label="sigma", minimum=1, maximum=1000, step=1, value=250)
            k = gr.Slider(label="k", minimum=0.1, maximum=10, step=0.1, value=6.0)
            condweight = gr.Slider(label="condweight", minimum=0, maximum=2, step=0.01, value=0.8)
            no_adain = gr.Checkbox(label="Disable AdaIN", value=False)
            no_adae = gr.Checkbox(label="Disable ADAE", value=False)

        run.click(
            fn=generate,
            inputs=[
                image,
                mask,
                prompt,
                negative_prompt,
                seed,
                randomize_seed,
                guidance_scale,
                num_inference_steps,
                num_diffusion_steps,
                envratio,
                fusion_func,
                reverse_fusion,
                sigma,
                k,
                condweight,
                no_adain,
                no_adae,
            ],
            outputs=[output, used_seed],
        )

    app.queue().launch(server_name=args.server_name, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
