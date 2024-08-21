import logging
import os
import tempfile
import time

import gradio as gr
import numpy as np
import rembg
import torch
from PIL import Image
from functools import partial

from tsr.system import TSR
from tsr.utils import remove_background, resize_foreground, to_gradio_3d_orientation

import argparse
import aspose.threed as a3d  # Import Aspose.3D

if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"

model = TSR.from_pretrained(
    "stabilityai/TripoSR",
    config_name="config.yaml",
    weight_name="model.ckpt",
)

# Adjust the chunk size to balance between speed and memory usage
model.renderer.set_chunk_size(8192)
model.to(device)

rembg_session = rembg.new_session()

def check_input_image(input_image):
    if input_image is None:
        raise gr.Error("No image uploaded!")

def dalle_preprocess(input_image):
    # Step 1: 이미지를 base64로 인코딩
    base64_image = model.encode_image(input_image)

    # Step 2: GPT-4를 통해 이미지 내용 설명 얻기
    image_content = model.get_image_content_from_gpt4(base64_image)

    # Step 3: DALLE-3을 사용해 이미지 생성
    processed_image = model.generate_image_with_dalle(image_content)

    return processed_image

def preprocess(input_image, do_remove_background, foreground_ratio):
    # Ensure the input image is a PIL Image object
    if isinstance(input_image, np.ndarray):
        input_image = Image.fromarray(input_image)
    
    def fill_background(image):
        image = np.array(image).astype(np.float32) / 255.0
        image = image[:, :, :3] * image[:, :, 3:4] + (1 - image[:, :, 3:4]) * 0.5
        image = Image.fromarray((image * 255.0).astype(np.uint8))
        return image

    if do_remove_background:
        image = input_image.convert("RGB")
        image = remove_background(image, rembg_session)
        image = resize_foreground(image, foreground_ratio)
        image = fill_background(image)
    else:
        image = input_image
        if image.mode == "RGBA":
            image = fill_background(image)
    return image

def generate(image, mc_resolution, formats=["obj", "glb", "stl"]):
    scene_codes = model(image, device=device)
    mesh = model.extract_mesh(scene_codes, True, resolution=mc_resolution)[0]
    mesh = to_gradio_3d_orientation(mesh)
    
    rv = []
    temp_files = []
    for format in formats:
        mesh_path = tempfile.NamedTemporaryFile(suffix=f".{format}", delete=False)
        temp_files.append(mesh_path.name)
        if format in ["obj", "glb"]:
            mesh.export(mesh_path.name)
        elif format == "stl":
            if "obj" in formats:
                scene = a3d.Scene.from_file(temp_files[formats.index("obj")])
            elif "glb" in formats:
                scene = a3d.Scene.from_file(temp_files[formats.index("glb")])
            scene.save(mesh_path.name)
        rv.append(mesh_path.name)
    
    return rv

def generate_from_text(text, mc_resolution, formats=["obj", "glb", "stl"]):
    # Step 1: Text to Image using DALL-E
    generated_image = model.generate_image_from_text(text)

    # Step 2: Process the generated image (you can add background removal, etc., if needed)
    processed_image = preprocess(generated_image, False, 0.9)

    # Step 3: Generate 3D model from the processed image
    mesh_names = generate(processed_image, mc_resolution, formats)

    return processed_image, mesh_names[0], mesh_names[1], mesh_names[2]

def run_example(image_pil):
    preprocessed = dalle_preprocess(image_pil)
    processed_image = preprocess(preprocessed, False, 0.9)
    mesh_names = generate(processed_image, 256, ["obj", "glb", "stl"])
    return processed_image, mesh_names[0], mesh_names[1], mesh_names[2]

with gr.Blocks(title="TripoSR") as interface:
    gr.Markdown(
        """
    # TripoSR Demo
    [TripoSR](https://github.com/VAST-AI-Research/TripoSR) is a state-of-the-art open-source model for **fast** feedforward 3D reconstruction from a single image, collaboratively developed by [Tripo AI](https://www.tripo3d.ai/) and [Stability AI](https://stability.ai/).
    
    **Tips:**
    1. If you find the result is unsatisfactory, please try changing the foreground ratio. It might improve the results.
    2. It's better to disable "Remove Background" for the provided examples (except for the last one) since they have been already preprocessed.
    3. Otherwise, please disable "Remove Background" option only if your input image is RGBA with a transparent background, image contents are centered and occupy more than 70% of the image width or height.
    """
    )
    with gr.Row(variant="panel"):
        with gr.Column():
            with gr.Row():
                input_image = gr.Image(
                    label="Input Image",
                    image_mode="RGBA",
                    sources="upload",
                    type="pil",
                    elem_id="content_image",
                )
                processed_image = gr.Image(label="Processed Image", interactive=False)
            with gr.Row():
                with gr.Group():
                    input_text = gr.Textbox(label="Input Text", placeholder="Describe the object you want to generate in 3D")
                    do_remove_background = gr.Checkbox(
                        label="Remove Background", value=True
                    )
                    foreground_ratio = gr.Slider(
                        label="Foreground Ratio",
                        minimum=0.5,
                        maximum=1.0,
                        value=0.85,
                        step=0.05,
                    )
                    mc_resolution = gr.Slider(
                        label="Marching Cubes Resolution",
                        minimum=32,
                        maximum=320,
                        value=256,
                        step=32
                    )
            with gr.Row():
                submit_image = gr.Button("Generate from Image", elem_id="generate_image", variant="primary")
                submit_text = gr.Button("Generate from Text", elem_id="generate_text", variant="primary")
        with gr.Column():
            with gr.Tab("OBJ"):
                output_model_obj = gr.Model3D(
                    label="Output Model (OBJ Format)",
                    interactive=False,
                )
                gr.Markdown("Note: The model shown here is flipped. Download to get correct results.")
            with gr.Tab("GLB"):
                output_model_glb = gr.Model3D(
                    label="Output Model (GLB Format)",
                    interactive=False,
                )
                gr.Markdown("Note: The model shown here has a darker appearance. Download to get correct results.")
            with gr.Tab("STL"):
                output_model_stl = gr.Model3D(
                    label="Output Model (STL Format)",
                    interactive=False,
                )
                gr.Markdown("Note: The model shown here may appear different. Download to get correct results.")
    with gr.Row(variant="panel"):
        gr.Examples(
            examples=[
                "examples/poly_fox.png",
                "examples/robot.png",
                "examples/toy_bingbong.png",
                "examples/toy_lion.jpg",
                "examples/toy_sword.png",
                "examples/toy_teddybear.png",
                "examples/image_0.png",
                "examples/image_1.png",
                "examples/image_2.png",
                "examples/image_3.png",
                "examples/image_4.png",
            ],
            inputs=[input_image],
            outputs=[processed_image, output_model_obj, output_model_glb, output_model_stl],
            cache_examples=False,
            fn=partial(run_example),
            label="Examples",
            examples_per_page=20,
        )
    submit_image.click(fn=check_input_image, inputs=[input_image])\
        .success(fn=dalle_preprocess, inputs=[input_image], outputs=[processed_image])\
        .success(fn=preprocess, inputs=[processed_image, do_remove_background, foreground_ratio], outputs=[processed_image])\
        .success(fn=generate, inputs=[processed_image, mc_resolution], outputs=[output_model_obj, output_model_glb, output_model_stl])

    submit_text.click(fn=generate_from_text, inputs=[input_text, mc_resolution], outputs=[processed_image, output_model_obj, output_model_glb, output_model_stl])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--username', type=str, default=None, help='Username for authentication')
    parser.add_argument('--password', type=str, default=None, help='Password for authentication')
    parser.add_argument('--port', type=int, default=7860, help='Port to run the server listener on')
    parser.add_argument("--listen", action='store_true', help="launch gradio with 0.0.0.0 as server name, allowing to respond to network requests")
    parser.add_argument("--share", action='store_true', help="use share=True for gradio and make the UI accessible through their site")
    parser.add_argument("--queuesize", type=int, default=1, help="launch gradio queue max_size")
    args = parser.parse_args()
    interface.queue(max_size=args.queuesize)
    interface.launch(
        auth=(args.username, args.password) if (args.username and args.password) else None,
        share=args.share,
        server_name="0.0.0.0" if args.listen else None, 
        server_port=args.port
    )
