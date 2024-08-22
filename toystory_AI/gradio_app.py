import os
import tempfile
import argparse
import logging

import gradio as gr
import numpy as np
import rembg
import torch
import boto3
from PIL import Image
from botocore.exceptions import NoCredentialsError, PartialCredentialsError
import requests
from io import BytesIO

import aspose.threed as a3d  # Import Aspose.3D
from tsr.system import TSR
from tsr.utils import remove_background, resize_foreground, to_gradio_3d_orientation

from dotenv import load_dotenv

load_dotenv()

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

# OPENAI
OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")

# AWS S3
S3_ACCESS_KEY=os.getenv("S3_ACCESS_KEY")
S3_PRIVATE_KEY=os.getenv("S3_PRIVATE_KEY")
S3_BUCKET_NAME=os.getenv("S3_BUCKET_NAME")

def process_and_generate(input_image=None, input_text=None, input_s3_url=None, mc_resolution=256, do_remove_background=True, foreground_ratio=0.9, formats=["obj", "glb", "stl"],
                         title=None):
    # Step 1: Fetch and preprocess the input
    if input_s3_url:
        # Download the image from S3
        response = requests.get(input_s3_url)
        if response.status_code == 200:
            input_image = Image.open(BytesIO(response.content))
        else:
            raise ValueError("Failed to download image from S3 URL.")
    
    if input_image:
        # DALL-E preprocessing for an input image
        base64_image = model.encode_image(input_image)
        image_content = model.get_image_content_from_gpt4(base64_image)
        processed_image = model.generate_image_with_dalle(image_content)
    elif input_text:
        # DALL-E processing for an input text
        generated_image = model.generate_image_from_text(input_text)
        processed_image = generated_image
    else:
        raise ValueError("Either input_image, input_text, or input_s3_url must be provided")
    
    # Step 2: Further processing (background removal, resizing)
    if isinstance(processed_image, np.ndarray):
        processed_image = Image.fromarray(processed_image)
    
    def fill_background(image):
        image = np.array(image).astype(np.float32) / 255.0
        image = image[:, :, :3] * image[:, :, 3:4] + (1 - image[:, :, 3:4]) * 0.5
        image = Image.fromarray((image * 255.0).astype(np.uint8))
        return image

    if do_remove_background:
        processed_image = processed_image.convert("RGB")
        processed_image = remove_background(processed_image, rembg_session)
        processed_image = resize_foreground(processed_image, foreground_ratio)
        processed_image = fill_background(processed_image)
    else:
        if processed_image.mode == "RGBA":
            processed_image = fill_background(processed_image)
    
    # Step 3: Generate 3D model and upload to S3
    scene_codes = model(processed_image, device=device)
    mesh = model.extract_mesh(scene_codes, True, resolution=mc_resolution)[0]
    mesh = to_gradio_3d_orientation(mesh)
    
    s3_urls = []
    obj_file_path = None

    for format in formats:
        mesh_path = tempfile.NamedTemporaryFile(suffix=f".{format}", delete=False)
        print(f"Creating file: {mesh_path.name}")

        flag = True
        
        if format == "obj":
            mesh.export(mesh_path.name)
            obj_file_path = mesh_path.name  # Save OBJ file path
            continue
        elif format == "glb":
            mesh.export(mesh_path.name)
            loc = "0-glb"
        elif format == "stl":
            if obj_file_path is None:
                raise RuntimeError("OBJ file does not exist. OBJ file is required to create STL.")
            print(f"Using OBJ file: {obj_file_path} to create STL")
            scene = a3d.Scene.from_file(obj_file_path)
            scene.save(mesh_path.name)
            loc = "1-stl"
        
        # Upload to S3 and collect URLs
        s3_cli = boto3.client(
            's3',
            aws_access_key_id=S3_ACCESS_KEY,
            aws_secret_access_key=S3_PRIVATE_KEY,
            region_name='ap-northeast-2'
        )

        try:
            with open(mesh_path.name, 'rb') as file:
                s3_cli.upload_fileobj(
                    file,
                    S3_BUCKET_NAME,
                    f"3d-contents/{loc}/{title}-{os.path.basename(mesh_path.name)}",
                    ExtraArgs={'ContentType': "application/octet-stream"}
                )
            url = f"https://{S3_BUCKET_NAME}.s3.ap-northeast-2.amazonaws.com/3d-contents/{loc}/{title}-{os.path.basename(mesh_path.name)}"
            print(f"File uploaded to S3: {url}")
            s3_urls.append(url)
        except NoCredentialsError:
            raise Exception("AWS credentials not found.")
        except PartialCredentialsError:
            raise Exception("Incomplete AWS credentials provided.")
        except Exception as e:
            raise Exception(f"Failed to upload file: {str(e)}")
    
    # Return the processed image and individual URLs for the 3D models
    return processed_image, s3_urls[0], s3_urls[1]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--username', type=str, default=None, help='Username for authentication')
    parser.add_argument('--password', type=str, default=None, help='Password for authentication')
    parser.add_argument('--port', type=int, default=7860, help='Port to run the server listener on')
    parser.add_argument("--listen", action='store_true', help="launch gradio with 0.0.0.0 as server name, allowing to respond to network requests")
    parser.add_argument("--share", action='store_true', help="use share=True for gradio and make the UI accessible through their site")
    parser.add_argument("--queuesize", type=int, default=1, help="launch gradio queue max_size")
    args = parser.parse_args()

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
        # Input section
        with gr.Row(variant="panel"):
            with gr.Column():
                input_image = gr.Image(
                    label="Input Image",
                    image_mode="RGBA",
                    sources="upload",
                    type="pil",
                    elem_id="content_image",
                )
                input_text = gr.Textbox(label="Input Text", placeholder="Describe the object you want to generate in 3D")
                input_s3_url = gr.Textbox(label="Input S3 URL", placeholder="Enter S3 URL of the image")
                do_remove_background = gr.Checkbox(label="Remove Background", value=True)
                foreground_ratio = gr.Slider(label="Foreground Ratio", minimum=0.5, maximum=1.0, value=0.85, step=0.05)
                mc_resolution = gr.Slider(label="Marching Cubes Resolution", minimum=32, maximum=320, value=256, step=32)
                submit_button = gr.Button("Generate 3D Model", elem_id="generate", variant="primary")
            # Output section
            with gr.Column():
                processed_image = gr.Image(label="Processed Image", interactive=False)
                with gr.Tab("OBJ"):
                    output_model_obj = gr.Model3D(label="Output Model (OBJ Format)", interactive=False)
                with gr.Tab("GLB"):
                    output_model_glb = gr.Model3D(label="Output Model (GLB Format)", interactive=False)
                with gr.Tab("STL"):
                    output_model_stl = gr.Model3D(label="Output Model (STL Format)", interactive=False)

        submit_button.click(
            fn=process_and_generate,
            inputs=[input_image, input_text, input_s3_url, mc_resolution, do_remove_background, foreground_ratio],
            outputs=[processed_image, output_model_glb, output_model_stl]
        )

        # Examples section
        gr.Examples(
            examples=[
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
            fn=process_and_generate,
            label="Examples",
            examples_per_page=20,
        )

    interface.queue(max_size=args.queuesize)
    interface.launch(
        auth=(args.username, args.password) if (args.username and args.password) else None,
        share=args.share,
        server_name="0.0.0.0" if args.listen else None,
        server_port=args.port
    )
