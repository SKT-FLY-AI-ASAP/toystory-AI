import os
import tempfile
import argparse
import logging
import subprocess

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
    # Step 1: Fetch and preprocess the input
    image_content = None

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
    processed_image_path = os.path.join(tempfile.gettempdir(), f"{title}_processed.png")
    processed_image.save(processed_image_path)

    # Generate 3D model and save locally
    scene_codes = model(processed_image, device=device)
    mesh = model.extract_mesh(scene_codes, True, resolution=mc_resolution)[0]
    mesh = to_gradio_3d_orientation(mesh)

    file_paths = []
    obj_file_path = None

    for format in formats:
        mesh_path = os.path.join(tempfile.gettempdir(), f"{title}.{format}")
        print(f"Creating file: {mesh_path}")

        # Set the appropriate location path for each format
        if format == "obj":
            mesh.export(mesh_path)
            obj_file_path = mesh_path  # Save OBJ file path
        elif format == "stl":
            if obj_file_path is None:
                raise RuntimeError("OBJ file does not exist. OBJ file is required to create STL.")
            print(f"Using OBJ file: {obj_file_path} to create STL")
            scene = a3d.Scene.from_file(obj_file_path)
            scene.save(mesh_path)
        elif format == "glb":
            mesh.export(mesh_path)
        else:
            raise ValueError(f"Unsupported format: {format}")

        file_paths.append(mesh_path)

    # Generate a backgrounded image GLB
    backgrounded_image = model.generate_image_with_background(image_content, input_text or "Generated Image")

    backgrounded_scene_codes = model(backgrounded_image, device=device)
    backgrounded_mesh = model.extract_mesh(backgrounded_scene_codes, True, resolution=mc_resolution)[0]
    backgrounded_mesh = to_gradio_3d_orientation(backgrounded_mesh)

    backgrounded_mesh_path = os.path.join(tempfile.gettempdir(), f"{title}_backgrounded.glb")
    print(f"Creating file with background: {backgrounded_mesh_path}")
    backgrounded_mesh.export(backgrounded_mesh_path)

    # Download the generated GLB files
    glb_1_path = file_paths[-1]  # Assuming GLB is the last format generated
    glb_2_path = backgrounded_mesh_path

    # Modify the blender script with the correct file paths
    blender_script = f"""
import bpy

# Blender 초기화 - 기존 씬 초기화
bpy.ops.wm.read_factory_settings(use_empty=True)

# 첫 번째 GLB 파일 불러오기
bpy.ops.import_scene.gltf(filepath="{glb_1_path}")

# 첫 번째 GLB 파일의 오브젝트들을 축소하고 Y축 방향으로 이동
for obj in bpy.context.selected_objects:
    obj.scale = (0.3, 0.3, 0.3)  # 오브젝트 크기를 30%로 축소
    obj.location.y += 0.3  # 원하는 만큼 Y축 방향으로 이동
    obj.location.z -= 0.2

# 두 번째 GLB 파일 불러오기
bpy.ops.import_scene.gltf(filepath="{glb_2_path}")

# 모든 오브젝트를 선택
bpy.ops.object.select_all(action='SELECT')

# 객체 모드로 전환
bpy.ops.object.mode_set(mode='OBJECT')

# 모든 선택된 오브젝트를 메쉬로 변환
bpy.ops.object.convert(target='MESH')

# 활성 오브젝트 설정 및 결합
if bpy.context.selected_objects:
    bpy.context.view_layer.objects.active = bpy.context.selected_objects[0]
    
    # 선택된 오브젝트를 하나로 결합
    bpy.ops.object.join()
else:
    print("No objects are selected. Please check the imported GLB files.")

# 결합된 오브젝트를 새로운 GLB 파일로 내보내기
combined_path = "{os.path.join(tempfile.gettempdir(), 'combined_file.glb')}"
bpy.ops.export_scene.gltf(filepath=combined_path, export_format='GLB')
print("Combined GLB file saved to:", combined_path)
    """

    # Save the script to a temporary file
    blender_script_path = os.path.join(tempfile.gettempdir(), f"{title}_blender_script.py")
    with open(blender_script_path, 'w') as f:
        f.write(blender_script)

    # Execute the Blender script
    try:
        subprocess.run(["blender", "--background", "--python", blender_script_path], check=True)
        combined_glb_path = os.path.join(tempfile.gettempdir(), 'combined_file.glb')
    except subprocess.CalledProcessError as e:
        raise Exception(f"Blender script execution failed: {str(e)}")

    # Return the file paths for the processed image and models for Gradio to display
    return processed_image_path, backgrounded_image, file_paths[1], backgrounded_mesh_path, file_paths[2], combined_glb_path


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
        # Toystory Demo
        [Toystory](https://github.com/SKT-FLY-AI-ASAP/toystory-AI) is a model for 3D reconstruction from single image, based on TripoSR(https://github.com/VAST-AI-Research/TripoSR).
        
        **How To Use:**
        1. Use the picture in the sample or try putting in the 2D image yourself.
        * It's better to disable "Remove Background" for the provided examples since they have been already preprocessed.
        2. Describe the image you want to create in Text.
        3. Enter S3 url as input.
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
            # Output section
            with gr.Column():
                processed_image = gr.Image(label="Processed Image", interactive=False)
                backgrounded_image = gr.Image(label="Backgrounded Image", interactive=False)
                with gr.Tab("GLB"):
                    output_model_glb = gr.Model3D(label="Output Model (GLB Format)", interactive=False)
                with gr.Tab("STL"):
                    output_model_stl = gr.Model3D(label="Output Model (STL Format)", interactive=False)
                with gr.Tab("Background GLB"):
                    output_model_glb_bg = gr.Model3D(label="Output Model (Background GLB)", interactive=False)
                with gr.Tab("Combined GLB"):
                    output_combined_glb = gr.Model3D(label="Output Combined GLB", interactive=False)

        submit_button.click(
            fn=process_and_generate,
            inputs=[input_image, input_text, input_s3_url, mc_resolution, do_remove_background, foreground_ratio],
            outputs=[processed_image, backgrounded_image, output_model_stl, output_model_glb_bg, output_model_glb, output_combined_glb]
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
            outputs=[processed_image, backgrounded_image, output_model_stl, output_model_glb_bg, output_model_glb, output_combined_glb],
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
