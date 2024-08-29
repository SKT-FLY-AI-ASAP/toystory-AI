import argparse
import os
import tempfile
import subprocess
import random

import gradio as gr
import numpy as np
import rembg
import torch
from PIL import Image
import requests
from io import BytesIO

import aspose.threed as a3d  # Import Aspose.3D
from tsr.system import TSR
from tsr.utils import remove_background, resize_foreground, to_gradio_3d_orientation

from dotenv import load_dotenv

load_dotenv()

if torch.cuda.is_available():
    device = "cpu"  # "cuda:0"
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

def process_and_generate(input_image=None, input_text=None, input_s3_url=None, mc_resolution=256, do_remove_background=True, foreground_ratio=0.9, formats=["obj", "stl", "glb"],
                         title=None):
    if not title:
        title = "output"  # Default title if none is provided

    # Step 1: Fetch and preprocess the input
    image_content = None
    prompt = input_text  # Define the prompt

    if input_s3_url:
        response = requests.get(input_s3_url)
        if response.status_code == 200:
            input_image = Image.open(BytesIO(response.content))
        else:
            raise ValueError("Failed to download image from URL.")

    if input_image:
        base64_image = model.encode_image(input_image)
        image_content = model.get_image_content_from_gpt4(base64_image)
        processed_image = model.generate_image_with_dalle(image_content)
    elif input_text:
        image_content = input_text  # Assigning input_text to image_content
        generated_image = model.generate_image_from_text(input_text)
        processed_image = generated_image
    else:
        raise ValueError("Either input_image, input_text, or input_s3_url must be provided")

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

    # Save the processed image locally
    output_dir = "output_files"
    os.makedirs(output_dir, exist_ok=True)

    processed_image_path = os.path.join(output_dir, f"{title}_processed.png")
    processed_image.save(processed_image_path)

    # Generate 3D model and save locally
    scene_codes = model(processed_image, device=device)
    mesh = model.extract_mesh(scene_codes, True, resolution=mc_resolution)[0]
    mesh = to_gradio_3d_orientation(mesh)

    # Set the GLB and animated GLB paths
    static_glb_path = os.path.join(output_dir, f"{title}.glb")
    animated_glb_path = os.path.join(output_dir, f"{title}_animated.glb")

    # Modify the blender script with the correct file paths and animate the model
    blender_script = f"""
import bpy
import random

# Blender 초기화 - 기존 씬 초기화
bpy.ops.wm.read_factory_settings(use_empty=True)

# GLB 파일 불러오기
bpy.ops.import_scene.gltf(filepath="{static_glb_path}")

# GLB 파일의 오브젝트들을 축소하고 Y축 방향으로 이동
for obj in bpy.context.selected_objects:
    obj.scale = (0.3, 0.3, 0.3)  # 오브젝트 크기를 30%로 축소
    obj.location.y += 0.3  # 원하는 만큼 Y축 방향으로 이동
    obj.location.z -= 0.2

# 애니메이션 설정
frame_start = 1
frames = []

# X축과 Y축 애니메이션 설정 (총 5번 반복)
for i in range(5):
    duration = random.randint(40, 80)  # 40~80프레임 동안 이동 및 회전 (약 1.5초~3초)
    x_target = random.uniform(-0.2, 0.2)  # X축 목표 위치
    rotation_target = random.uniform(-0.523599, 0.523599)  # -30도 ~ 30도 회전 (라디안 값)

    frame_end = frame_start + duration
    frames.append((frame_start, frame_end, x_target, rotation_target))

    # 애니메이션 설정
    for obj in bpy.context.selected_objects:
        # X축 이동 설정
        obj.location.x = x_target
        obj.keyframe_insert(data_path="location", frame=frame_end)

        # 회전 설정
        obj.rotation_euler = (0, 0, rotation_target)
        obj.keyframe_insert(data_path="rotation_euler", frame=frame_end)

    frame_start = frame_end

# 애니메이션을 포함하여 GLB 파일로 내보내기
bpy.ops.export_scene.gltf(filepath="{animated_glb_path}", export_format='GLB')
print("Animated GLB file saved to:", "{animated_glb_path}")
"""

    # Save the script to a temporary file
    blender_script_path = os.path.join(output_dir, f"{title}_blender_script.py")
    with open(blender_script_path, 'w') as f:
        f.write(blender_script)

    # Save the static GLB file
    mesh.export(static_glb_path)

    # Execute the Blender script for animated GLB
    try:
        subprocess.run(["blender", "--background", "--python", blender_script_path], check=True)
    except subprocess.CalledProcessError as e:
        raise Exception(f"Blender script execution failed: {str(e)}")

    # Generate STL file
    stl_file_path = os.path.join(output_dir, f"{title}.stl")
    scene = a3d.Scene.from_file(animated_glb_path)
    scene.save(stl_file_path)

    # Generate background image and save locally
    backgrounded_image = model.generate_image_with_background(image_content, input_text or "Generated Image")
    backgrounded_image_path = os.path.join(output_dir, f"{title}_backgrounded.png")
    backgrounded_image.save(backgrounded_image_path)

    # Generate music and save locally
    mp3_file_path = model.generate_music(image_content, input_text)

    # Return the individual file paths for Gradio to handle each output separately
    return processed_image_path, backgrounded_image_path, static_glb_path, animated_glb_path, stl_file_path, mp3_file_path

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--username', type=str, default=None, help='Username for authentication')
    parser.add_argument('--password', type=str, default=None, help='Password for authentication')
    parser.add_argument('--port', type=int, default=7860, help='Port to run the server listener on')
    parser.add_argument("--listen", action='store_true', help="launch gradio with 0.0.0.0 as server name, allowing to respond to network requests")
    parser.add_argument("--share", action='store_true', help="use share=True for gradio and make the UI accessible through their site")
    parser.add_argument("--queuesize", type=int, default=1, help="launch gradio queue max_size")
    args = parser.parse_args()

    with gr.Blocks(title="Toystory") as interface:
        gr.Markdown(
            """
        # Toystory Demo
        [Toystory](https://github.com/SKT-FLY-AI-ASAP/toystory-AI) is a model for 3D reconstruction from a single image, based on TripoSR(https://github.com/VAST-AI-Research/TripoSR).
        
        **How To Use:**
        1. Use the picture in the sample or try putting in the 2D image yourself.
        * It's better to disable "Remove Background" for the provided examples since they have been already preprocessed.
        2. Describe the image you want to create in Text.
        3. Enter the S3 URL as input.
        4. If you press Create, it corrects the 2D image and recommends the appropriate background.
        5. You can download the desired results.
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
            with gr.Column():
                processed_image = gr.Image(label="Processed Image", interactive=False)
                backgrounded_image = gr.Image(label="Backgrounded Image (PNG)", interactive=False)
                output_glb = gr.Model3D(label="Output GLB (Static)", interactive=False)
                output_animated_glb = gr.Model3D(label="Output Animated GLB", interactive=False)
                output_stl = gr.Model3D(label="Output STL Format", interactive=False)
                mp3_output = gr.Audio(label="Generated Audio (MP3)", interactive=False)

        submit_button.click(
            fn=process_and_generate,
            inputs=[input_image, input_text, input_s3_url, mc_resolution, do_remove_background, foreground_ratio],
            outputs=[processed_image, backgrounded_image, output_glb, output_animated_glb, output_stl, mp3_output]
        )

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
                "examples/bird.PNG",
                "examples/dinosaur.png",
                "examples/frog.png",
            ],
            inputs=[input_image],
            outputs=[processed_image, backgrounded_image, output_glb, output_animated_glb, output_stl, mp3_output],
            cache_examples=False
        )

        interface.queue(max_size=args.queuesize)
        interface.launch(
            auth=(args.username, args.password) if (args.username and args.password) else None,
            share=args.share,
            server_name="0.0.0.0" if args.listen else None,
            server_port=args.port
        )
