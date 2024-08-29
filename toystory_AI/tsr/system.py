import math
import os
from dataclasses import dataclass, field
from typing import List, Union
import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F
import trimesh
from einops import rearrange
from huggingface_hub import hf_hub_download
from omegaconf import OmegaConf
from PIL import Image
from .models.isosurface import MarchingCubeHelper
from .utils import (
    BaseModule,
    ImagePreprocessor,
    find_class,
    get_spherical_cameras,
    scale_tensor,
    remove_background  # 이미 정의된 remove_background 함수 임포트
)
import openai
import requests
import base64
from io import BytesIO
import rembg  # rembg 임포트
import soundfile as sf  # 오디오 파일을 저장하기 위한 라이브러리
from pydub import AudioSegment  # mp3 변환을 위한 라이브러리

from dotenv import load_dotenv

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")
if openai.api_key is None:
    raise ValueError("API key is not set")

# Initialize rembg session
rembg_session = rembg.new_session()

class TSR(BaseModule):
    @dataclass
    class Config(BaseModule.Config):
        cond_image_size: int
        image_tokenizer_cls: str
        image_tokenizer: dict
        tokenizer_cls: str
        tokenizer: dict
        backbone_cls: str
        backbone: dict
        post_processor_cls: str
        post_processor: dict
        decoder_cls: str
        decoder: dict
        renderer_cls: str
        renderer: dict
    cfg: Config
    
    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path: str, config_name: str, weight_name: str
    ):
        if os.path.isdir(pretrained_model_name_or_path):
            config_path = os.path.join(pretrained_model_name_or_path, config_name)
            weight_path = os.path.join(pretrained_model_name_or_path, weight_name)
        else:
            config_path = hf_hub_download(
                repo_id=pretrained_model_name_or_path, filename=config_name
            )
            weight_path = hf_hub_download(
                repo_id=pretrained_model_name_or_path, filename=weight_name
            )
        cfg = OmegaConf.load(config_path)
        OmegaConf.resolve(cfg)
        model = cls(cfg)
        ckpt = torch.load(weight_path, map_location="cpu")
        model.load_state_dict(ckpt)
        return model
    
    def configure(self):
        self.image_tokenizer = find_class(self.cfg.image_tokenizer_cls)(
            self.cfg.image_tokenizer
        )
        self.tokenizer = find_class(self.cfg.tokenizer_cls)(self.cfg.tokenizer)
        self.backbone = find_class(self.cfg.backbone_cls)(self.cfg.backbone)
        self.post_processor = find_class(self.cfg.post_processor_cls)(
            self.cfg.post_processor
        )
        self.decoder = find_class(self.cfg.decoder_cls)(self.cfg.decoder)
        self.renderer = find_class(self.cfg.renderer_cls)(self.cfg.renderer)
        self.image_processor = ImagePreprocessor()
        self.isosurface_helper = None

    def forward(
        self,
        image: Union[
            PIL.Image.Image,
            np.ndarray,
            torch.FloatTensor,
            List[PIL.Image.Image],
            List[np.ndarray],
            List[torch.FloatTensor],
        ],
        device: str,
    ) -> torch.FloatTensor:
        rgb_cond = self.image_processor(image, self.cfg.cond_image_size)[:, None].to(device)
        batch_size = rgb_cond.shape[0]
        input_image_tokens: torch.Tensor = self.image_tokenizer(
            rearrange(rgb_cond, "B Nv H W C -> B Nv C H W", Nv=1),
        )
        input_image_tokens = rearrange(
            input_image_tokens, "B Nv C Nt -> B (Nv Nt) C", Nv=1
        )
        tokens: torch.Tensor = self.tokenizer(batch_size)
        tokens = self.backbone(
            tokens,
            encoder_hidden_states=input_image_tokens,
        )
        scene_codes = self.post_processor(self.tokenizer.detokenize(tokens))
        return scene_codes

    def encode_image(self, image):
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image.astype(np.uint8))
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    def get_image_content_from_gpt4(self, base64_image):
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {openai.api_key}"
        }
        payload = {
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "What’s in this image?"
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 300
        }
        try:
            response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
            response.raise_for_status()
            response_json = response.json()
            if 'choices' not in response_json or not response_json['choices']:
                raise KeyError("'choices' key not found in the response or it's empty.")
            return response_json['choices'][0]['message']['content']
        except requests.exceptions.HTTPError as http_err:
            print(f"HTTP error occurred: {http_err}")
            raise
        except Exception as err:
            print(f"An error occurred: {err}")
            raise
    
    # image to image
    def generate_image_with_dalle(self, image_content):
        lines = [
            "Modify the image based on the following instructions:",
            "1. Image content: " + image_content,
            "2. Change the background to a solid white color.",
            "3. Ensure that only one object is present in the image.",
            "4. Apply a 3D style to the object.",
            "5. Position the object at a 15-degree angle to show a side view.",
            "6. Choose and apply colors that complement the image content and enhance the overall theme. The colors should harmonize with the mood and atmosphere of the object."
        ]
        prompt = "\n".join(lines)
        response = openai.images.generate(
            model="dall-e-3",
            prompt=prompt,
            size="1024x1024",
            quality="standard",
            n=1,
        )
        image_url = response.data[0].url
        print(f"Generated Image URL: {image_url}")
        image_response = requests.get(image_url)
        return Image.open(BytesIO(image_response.content))

    def generate_image_from_text(self, text: str) -> Image.Image:
        lines = [
            "Generate one background image based on the following instructions:",
            "1. The background should depict a location or setting that reflects the mood, colors, and overall atmosphere of the input image content. Use the implied theme and tone to guide the choice of location.",
            "2. Apply colors that complement the image content and align with the overall mood, ensuring the color scheme is harmonious.",
            "3. Do not include any specific objects or characters from the input image; focus on creating a cohesive environment or scene that captures the intended feeling.",
            "4. Ensure the location has a playful, childlike atmosphere, suitable for the theme, while still being recognizable as a specific place."
        ]
        prompt = "\n".join(lines)
        response = openai.images.generate(
            model="dall-e-3",
            prompt=prompt,
            size="1024x1024",
            quality="standard",
            n=1
        )
        
        image_url = response.data[0].url
        print(f"Generated Image URL: {image_url}")
        image_response = requests.get(image_url)
        image = Image.open(BytesIO(image_response.content))
        image = remove_background(image, rembg_session)
        return image
    
    def generate_image_with_background(self, image_content, text: str) -> Image.Image:
        lines = [
            "1. The background should depict a location or setting that reflects the mood, colors, and overall atmosphere of the input image content.",
            "2. Use the implied theme and tone to guide the choice of location.",
            "3. Apply colors that complement the image content and align with the overall mood, ensuring the color scheme is harmonious.",
            "4. Do not include any specific objects or characters from the input image; focus on creating a cohesive environment or scene that captures the intended feeling.",
            "5. Ensure the location has a playful, childlike atmosphere, suitable for the theme, while still being recognizable as a specific place.",
            "Based on the above conditions, generate a background that matches the image content or text."
        ]
        prompt = "\n".join(lines)
        response = openai.images.generate(
            model="dall-e-3",
            prompt=prompt,
            size="1024x1024",
            quality="standard",
            n=1
        )
        
        image_url = response.data[0].url
        print(f"Generated Image URL: {image_url}")
        image_response = requests.get(image_url)
        return Image.open(BytesIO(image_response.content))

    def render(
        self,
        scene_codes,
        n_views: int,
        elevation_deg: float = 0.0,
        camera_distance: float = 1.9,
        fovy_deg: float = 40.0,
        height: int = 256,
        width: int = 256,
        return_type: str = "pil",
    ):
        rays_o, rays_d = get_spherical_cameras(
            n_views, elevation_deg, camera_distance, fovy_deg, height, width
        )
        rays_o, rays_d = rays_o.to(scene_codes.device), rays_d.to(scene_codes.device)
        def process_output(image: torch.FloatTensor):
            if return_type == "pt":
                return image
            elif return_type == "np":
                return image.detach().cpu().numpy()
            elif return_type == "pil":
                return Image.fromarray(
                    (image.detach().cpu().numpy() * 255.0).astype(np.uint8)
                )
            else:
                raise NotImplementedError
        images = []
        for scene_code in scene_codes:
            images_ = []
            for i in range(n_views):
                with torch.no_grad():
                    image = self.renderer(
                        self.decoder, scene_code, rays_o[i], rays_d[i]
                    )
                images_.append(process_output(image))
            images.append(images_)
        return images
    
    def set_marching_cubes_resolution(self, resolution: int):
        if (
            self.isosurface_helper is not None
            and self.isosurface_helper.resolution == resolution
        ):
            return
        self.isosurface_helper = MarchingCubeHelper(resolution)
        
    def extract_mesh(self, scene_codes, has_vertex_color, resolution: int = 256, threshold: float = 25.0):
        self.set_marching_cubes_resolution(resolution)
        meshes = []
        for scene_code in scene_codes:
            with torch.no_grad():
                density = self.renderer.query_triplane(
                    self.decoder,
                    scale_tensor(
                        self.isosurface_helper.grid_vertices.to(scene_codes.device),
                        self.isosurface_helper.points_range,
                        (-self.renderer.cfg.radius, self.renderer.cfg.radius),
                    ),
                    scene_code,
                )["density_act"]
            v_pos, t_pos_idx = self.isosurface_helper(-(density - threshold))
            v_pos = scale_tensor(
                v_pos,
                self.isosurface_helper.points_range,
                (-self.renderer.cfg.radius, self.renderer.cfg.radius),
            )
            color = None
            if has_vertex_color:
                with torch.no_grad():
                    color = self.renderer.query_triplane(
                        self.decoder,
                        v_pos,
                        scene_code,
                    )["color"]
            mesh = trimesh.Trimesh(
                vertices=v_pos.cpu().numpy(),
                faces=t_pos_idx.cpu().numpy(),
                vertex_colors=color.cpu().numpy() if has_vertex_color else None,
            )
            meshes.append(mesh)
        return meshes

    def select_music(self, category):
        music_directory = 'music'
        music_files = {
            'superhero': '1_superhero.mp3',
            'fantasy': '2_fantasy.mp3',
            'universe': '3_universe.mp3',
            'robot': '4_robot.mp3',
            'vehicle': '5_car_racing.wav',
            'dinosaur': '6_dinosaur.mp3',
            'doll': '7_doll.mp3',
            'animal': '8_animal.mp3',
            'adventure': '9_adventure.mp3',
            'fairytale': '10_fairytale.mp3'
        }

        # Default to 'fairytale' if the category is not found
        music_file = music_files.get(category, '10_fairytale.mp3')
        music_path = os.path.join(music_directory, music_file)

        # Check if the file exists
        if not os.path.exists(music_path):
            raise FileNotFoundError(f"Music file not found: {music_path}")

        # Load and return the selected music file
        music = AudioSegment.from_file(music_path)
        return music

    def generate_music(self, image_content, text):
        # Refined description to align more closely with the image content
        additional_description = (
            "Imagine a magical world full of surprises and adventures. "
            "Music has to be light and soft melodies that make you feel at ease. "
            "Draw a soft, hopeful rhythm. "
            "Melody has to be delicate and inspiring, and it has to be suitable for scenes where magic and nature are intertwined."
        )
        
        combined_prompt = f"{image_content} {text} {additional_description}"
        print(f"Music Generation Prompt: {combined_prompt}")

        # API call to categorize the image content or text
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {openai.api_key}"
        }
        payload = {
            "model": "gpt-4o",
            "messages": [
                {
                  "role": "user",
                  "content": [
                    {
                      "type": "text",
                      "text": f"Answer in one word. Among these categories in the list: [superhero, fantasy, universe, robot, vehicle, dinosaur, doll, animal, adventure, fairytale], this sentence fits which category? sentence: {combined_prompt}"
                    }
                  ]
                }
            ],
            "max_tokens": 300
        }

        category_response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        category_response_json = category_response.json()
        category_content = category_response_json['choices'][0]['message']['content'].lower().strip()

        # Select the appropriate music file based on the category
        selected_music = self.select_music(category_content)

        # Save the selected music to an output path
        output_music_path = f"output_audio_{category_content}.mp3"
        selected_music.export(output_music_path, format="mp3", bitrate="320k")
        
        print(f"Selected music category: {category_content}")
        print(f"Music file saved to: {output_music_path}")
        
        return output_music_path
