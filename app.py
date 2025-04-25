import os
import random
import sys
import subprocess
import spaces
import torch
import gradio as gr

from typing import Sequence, Mapping, Any, Union
from examples_db import ZEN_EXAMPLES
from PIL import Image, ImageChops
from huggingface_hub import hf_hub_download

# Setup ComfyUI if not already set up
# if not os.path.exists("ComfyUI"):
#    print("Setting up ComfyUI...")
#    subprocess.run(["bash", "setup_comfyui.sh"], check=True)

# Ensure the output directory exists
os.makedirs("output", exist_ok=True)

# Download models if not already present
print("Checking and downloading models...")
hf_hub_download(
    repo_id="black-forest-labs/FLUX.1-Redux-dev",
    filename="flux1-redux-dev.safetensors",
    local_dir="models/style_models",
)
hf_hub_download(
    repo_id="black-forest-labs/FLUX.1-Depth-dev",
    filename="flux1-depth-dev.safetensors",
    local_dir="models/diffusion_models",
)
hf_hub_download(
    repo_id="black-forest-labs/FLUX.1-Canny-dev",
    filename="flux1-canny-dev.safetensors",
    local_dir="models/controlnet",
)
hf_hub_download(
    repo_id="XLabs-AI/flux-controlnet-collections",
    filename="flux-canny-controlnet-v3.safetensors",
    local_dir="models/controlnet",
)
hf_hub_download(
    repo_id="Comfy-Org/sigclip_vision_384",
    filename="sigclip_vision_patch14_384.safetensors",
    local_dir="models/clip_vision",
)
hf_hub_download(
    repo_id="Kijai/DepthAnythingV2-safetensors",
    filename="depth_anything_v2_vitl_fp32.safetensors",
    local_dir="models/depthanything",
)
hf_hub_download(
    repo_id="black-forest-labs/FLUX.1-dev",
    filename="ae.safetensors",
    local_dir="models/vae/FLUX1",
)
hf_hub_download(
    repo_id="comfyanonymous/flux_text_encoders",
    filename="clip_l.safetensors",
    local_dir="models/text_encoders",
)
t5_path = hf_hub_download(
    repo_id="comfyanonymous/flux_text_encoders",
    filename="t5xxl_fp16.safetensors",
    local_dir="models/text_encoders/t5",
)

# Import required functions and setup ComfyUI path
import folder_paths


def get_value_at_index(obj: Union[Sequence, Mapping], index: int) -> Any:
    try:
        return obj[index]
    except KeyError:
        return obj["result"][index]


def find_path(name: str, path: str = None) -> str:
    if path is None:
        path = os.getcwd()
    if name in os.listdir(path):
        path_name = os.path.join(path, name)
        print(f"{name} found: {path_name}")
        return path_name
    parent_directory = os.path.dirname(path)
    if parent_directory == path:
        return None
    return find_path(name, parent_directory)


def add_comfyui_directory_to_sys_path() -> None:
    comfyui_path = find_path("ComfyUI")
    if comfyui_path is not None and os.path.isdir(comfyui_path):
        sys.path.append(comfyui_path)
        print(f"'{comfyui_path}' added to sys.path")


def add_extra_model_paths() -> None:
    try:
        from main import load_extra_path_config
    except ImportError:
        from utils.extra_config import load_extra_path_config
    extra_model_paths = find_path("extra_model_paths.yaml")
    if extra_model_paths is not None:
        load_extra_path_config(extra_model_paths)
    else:
        print("Could not find the extra_model_paths config file.")


# Initialize paths
add_comfyui_directory_to_sys_path()
add_extra_model_paths()


def import_custom_nodes() -> None:
    import asyncio
    import execution
    from nodes import init_extra_nodes
    import server

    # Create a new event loop if running in a new thread
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    server_instance = server.PromptServer(loop)
    execution.PromptQueue(server_instance)
    init_extra_nodes()


# Import all necessary nodes
print("Importing ComfyUI nodes...")
try:
    from nodes import (
        StyleModelLoader,
        VAEEncode,
        NODE_CLASS_MAPPINGS,
        LoadImage,
        CLIPVisionLoader,
        SaveImage,
        VAELoader,
        CLIPVisionEncode,
        DualCLIPLoader,
        EmptyLatentImage,
        VAEDecode,
        UNETLoader,
        CLIPTextEncode,
    )

    # Initialize all constant nodes and models in global context
    import_custom_nodes()
except Exception as e:
    print(f"Error importing ComfyUI nodes: {e}")
    raise

print("Setting up models...")
# Global variables for preloaded models and constants
intconstant = NODE_CLASS_MAPPINGS["INTConstant"]()
CONST_1024 = intconstant.get_value(value=1024)

# Load CLIP
dualcliploader = DualCLIPLoader()
CLIP_MODEL = dualcliploader.load_clip(
    clip_name1="t5/t5xxl_fp16.safetensors",
    clip_name2="clip_l.safetensors",
    type="flux",
)

# Load VAE
vaeloader = VAELoader()
VAE_MODEL = vaeloader.load_vae(vae_name="FLUX1/ae.safetensors")

# Load UNET
unetloader = UNETLoader()
UNET_MODEL = unetloader.load_unet(
    unet_name="flux1-depth-dev.safetensors", weight_dtype="default"
)

# Load CLIP Vision
clipvisionloader = CLIPVisionLoader()
CLIP_VISION_MODEL = clipvisionloader.load_clip(
    clip_name="sigclip_vision_patch14_384.safetensors"
)

# Load Style Model
stylemodelloader = StyleModelLoader()
STYLE_MODEL = stylemodelloader.load_style_model(
    style_model_name="flux1-redux-dev.safetensors"
)

# Initialize samplers
ksamplerselect = NODE_CLASS_MAPPINGS["KSamplerSelect"]()
SAMPLER = ksamplerselect.get_sampler(sampler_name="euler")

# Initialize depth model
cr_clip_input_switch = NODE_CLASS_MAPPINGS["CR Clip Input Switch"]()
downloadandloaddepthanythingv2model = NODE_CLASS_MAPPINGS[
    "DownloadAndLoadDepthAnythingV2Model"
]()
DEPTH_MODEL = downloadandloaddepthanythingv2model.loadmodel(
    model="depth_anything_v2_vitl_fp32.safetensors"
)

controlnetloader = NODE_CLASS_MAPPINGS["ControlNetLoader"]()
CANNY_XLABS_MODEL = controlnetloader.load_controlnet(
    control_net_name="flux-canny-controlnet-v3.safetensors"
)

# Initialize nodes
cliptextencode = CLIPTextEncode()
loadimage = LoadImage()
vaeencode = VAEEncode()
fluxguidance = NODE_CLASS_MAPPINGS["FluxGuidance"]()
controlNetApplyAdvanced = NODE_CLASS_MAPPINGS["ControlNetApplyAdvanced"]()
instructpixtopixconditioning = NODE_CLASS_MAPPINGS["InstructPixToPixConditioning"]()
clipvisionencode = CLIPVisionEncode()
stylemodelapplyadvanced = NODE_CLASS_MAPPINGS["StyleModelApplyAdvanced"]()
emptylatentimage = EmptyLatentImage()
basicguider = NODE_CLASS_MAPPINGS["BasicGuider"]()
basicscheduler = NODE_CLASS_MAPPINGS["BasicScheduler"]()
randomnoise = NODE_CLASS_MAPPINGS["RandomNoise"]()
samplercustomadvanced = NODE_CLASS_MAPPINGS["SamplerCustomAdvanced"]()
vaedecode = VAEDecode()
cr_text = NODE_CLASS_MAPPINGS["CR Text"]()
saveimage = SaveImage()
getimagesizeandcount = NODE_CLASS_MAPPINGS["GetImageSizeAndCount"]()
depthanything_v2 = NODE_CLASS_MAPPINGS["DepthAnything_V2"]()
canny_prossessor = NODE_CLASS_MAPPINGS["Canny"]()
imageresize = NODE_CLASS_MAPPINGS["ImageResize+"]()

from comfy import model_management

model_loaders = [CLIP_MODEL, VAE_MODEL, UNET_MODEL, CLIP_VISION_MODEL]

print("Loading models to GPU...")
model_management.load_models_gpu(
    [
        loader[0].patcher if hasattr(loader[0], "patcher") else loader[0]
        for loader in model_loaders
    ]
)

print("Setup complete!")


@spaces.GPU
def generate_image(
    prompt,
    structure_image,
    style_image,
    depth_strength=15,
    canny_strength=30,
    style_strength=0.5,
    steps=28,
    progress=gr.Progress(track_tqdm=True),
):
    """Main generation function that processes inputs and returns the path to the generated image."""
    timestamp = random.randint(10000, 99999)
    output_filename = f"flux_zen_{timestamp}.png"

    with torch.inference_mode():
        # Set up CLIP
        clip_switch = cr_clip_input_switch.switch(
            Input=1,
            clip1=get_value_at_index(CLIP_MODEL, 0),
            clip2=get_value_at_index(CLIP_MODEL, 0),
        )

        # Encode text
        text_encoded = cliptextencode.encode(
            text=prompt,
            clip=get_value_at_index(clip_switch, 0),
        )
        empty_text = cliptextencode.encode(
            text="",
            clip=get_value_at_index(clip_switch, 0),
        )

        # Process structure image
        structure_img = loadimage.load_image(image=structure_image)

        # Resize image
        resized_img = imageresize.execute(
            width=get_value_at_index(CONST_1024, 0),
            height=get_value_at_index(CONST_1024, 0),
            interpolation="bicubic",
            method="keep proportion",
            condition="always",
            multiple_of=16,
            image=get_value_at_index(structure_img, 0),
        )

        # Get image size
        size_info = getimagesizeandcount.getsize(
            image=get_value_at_index(resized_img, 0)
        )

        # Encode VAE
        vae_encoded = vaeencode.encode(
            pixels=get_value_at_index(size_info, 0),
            vae=get_value_at_index(VAE_MODEL, 0),
        )

        # Process canny
        canny_processed = canny_prossessor.detect_edge(
            image=get_value_at_index(size_info, 0),
            low_threshold=0.4,
            high_threshold=0.8,
        )

        # Apply canny Advanced
        canny_conditions = controlNetApplyAdvanced.apply_controlnet(
            positive=get_value_at_index(text_encoded, 0),
            negative=get_value_at_index(empty_text, 0),
            control_net=get_value_at_index(CANNY_XLABS_MODEL, 0),
            image=get_value_at_index(canny_processed, 0),
            strength=canny_strength,
            start_percent=0.0,
            end_percent=0.5,
            vae=get_value_at_index(VAE_MODEL, 0),
        )

        # Process depth
        depth_processed = depthanything_v2.process(
            da_model=get_value_at_index(DEPTH_MODEL, 0),
            images=get_value_at_index(size_info, 0),
        )

        # Apply Flux guidance
        flux_guided = fluxguidance.append(
            guidance=depth_strength,
            conditioning=get_value_at_index(canny_conditions, 0),
        )

        # Process style image
        style_img = loadimage.load_image(image=style_image)

        # Encode style with CLIP Vision
        style_encoded = clipvisionencode.encode(
            crop="center",
            clip_vision=get_value_at_index(CLIP_VISION_MODEL, 0),
            image=get_value_at_index(style_img, 0),
        )

        # Set up conditioning
        conditioning = instructpixtopixconditioning.encode(
            positive=get_value_at_index(flux_guided, 0),
            negative=get_value_at_index(canny_conditions, 1),
            vae=get_value_at_index(VAE_MODEL, 0),
            pixels=get_value_at_index(depth_processed, 0),
        )

        # Apply style
        style_applied = stylemodelapplyadvanced.apply_stylemodel(
            strength=style_strength,
            conditioning=get_value_at_index(conditioning, 0),
            style_model=get_value_at_index(STYLE_MODEL, 0),
            clip_vision_output=get_value_at_index(style_encoded, 0),
        )

        # Set up empty latent
        empty_latent = emptylatentimage.generate(
            width=get_value_at_index(resized_img, 1),
            height=get_value_at_index(resized_img, 2),
            batch_size=1,
        )

        # Set up guidance
        guided = basicguider.get_guider(
            model=get_value_at_index(UNET_MODEL, 0),
            conditioning=get_value_at_index(style_applied, 0),
        )

        # Set up scheduler
        schedule = basicscheduler.get_sigmas(
            scheduler="simple",
            steps=steps,
            denoise=1,
            model=get_value_at_index(UNET_MODEL, 0),
        )

        # Generate random noise
        noise = randomnoise.get_noise(noise_seed=random.randint(1, 2**64))

        # Sample
        sampled = samplercustomadvanced.sample(
            noise=get_value_at_index(noise, 0),
            guider=get_value_at_index(guided, 0),
            sampler=get_value_at_index(SAMPLER, 0),
            sigmas=get_value_at_index(schedule, 0),
            latent_image=get_value_at_index(empty_latent, 0),
        )

        # Decode VAE
        decoded = vaedecode.decode(
            samples=get_value_at_index(sampled, 0),
            vae=get_value_at_index(VAE_MODEL, 0),
        )

        # Create text node for prefix
        prefix = cr_text.text_multiline(text=f"flux_zen_{timestamp}")

        # Use SaveImage node to save the image
        saved_data = saveimage.save_images(
            filename_prefix=get_value_at_index(prefix, 0),
            images=get_value_at_index(decoded, 0),
        )

        try:
            saved_path = f"output/{saved_data['ui']['images'][0]['filename']}"

            return saved_path
        except Exception as e:
            print(f"Error getting saved image path: {e}")
            # Fall back to the expected path
            return os.path.join("output", output_filename)
css = """
footer {
    visibility: hidden;
}

.title {
    font-size: 1em;
    background: linear-gradient(109deg, rgba(34,193,195,1) 0%, rgba(67,253,45,1) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: bold;
}
"""

header = """
<div align="center" style="line-height: 1;">
    <a href="https://github.com/FotographerAI/Zen-style" target="_blank" style="margin: 2px;" name="github_repo_link"><img src="https://img.shields.io/badge/GitHub-Repo-181717.svg" alt="GitHub Repo" style="display: inline-block; vertical-align: middle;"></a>
    <a href="https://huggingface.co/spaces/fotographerai/ZenCtrl" target="_blank" style="margin: 2px;" name="hugging_face_space_link"><img src="https://img.shields.io/badge/🤗_HuggingFace-Space-ffbd45.svg" alt="ZenCtrl Space" style="display: inline-block; vertical-align: middle;"></a>
    <a href="https://discord.com/invite/b9RuYQ3F8k" target="_blank" style="margin: 2px;" name="discord_link"><img src="https://img.shields.io/badge/Discord-Join-7289da.svg?logo=discord" alt="Discord Link" style="display: inline-block; vertical-align: middle;"></a>
</div>
"""

with gr.Blocks(css=css) as demo:
    gr.HTML(header)
    
    gr.HTML(
        """
        <h1><center>🎨 FLUX <span class="title">Zen Style</span> Depth+Canny 🎨</center></h1>
        """
    )
    gr.Markdown(
        "Flux[dev] Redux + Flux[dev] Canny. This project implements a custom image-to-image style transfer pipeline that blends the style of one image (Image A) into the structure of another image (Image B).We just added canny to the previous work of Nathan Shipley, where the fusion of style and structure creates artistic visual outputs."
    )

    with gr.Row():
        with gr.Column(scale=2):
            prompt_input = gr.Textbox(
                label="Prompt",
                placeholder="Enter your prompt here...",
                info="Describe the image you want to generate",
            )
            with gr.Row():
                with gr.Column(scale=1):
                    structure_image = gr.Image(
                        image_mode="RGB", label="Structure Image", type="filepath"
                    )
                    depth_strength = gr.Slider(
                        minimum=0,
                        maximum=50,
                        value=15,
                        label="Depth Strength",
                        info="Controls how much the depth map influences the result",
                    )
                    canny_strength = gr.Slider(
                        minimum=0,
                        maximum=1.0,
                        value=0.30,
                        label="Canny Strength",
                        info="Controls how much the edge detection influences the result",
                    )
                    steps = gr.Slider(
                        minimum=10,
                        maximum=50,
                        value=28,
                        label="Steps",
                        info="More steps = better quality but slower generation",
                    )
                with gr.Column(scale=1):
                    style_image = gr.Image(label="Style Image", type="filepath")
                    style_strength = gr.Slider(
                        minimum=0,
                        maximum=1,
                        value=0.5,
                        label="Style Strength",
                        info="Controls how much the style image influences the result",
                    )

            with gr.Row():
                generate_btn = gr.Button("Generate", value=True, variant="primary")

        with gr.Column(scale=1):
            output_image = gr.Image(label="Generated Image")

    gr.Examples(
        examples=ZEN_EXAMPLES,
        inputs=[
            prompt_input,
            structure_image,
            style_image,
            output_image,
            depth_strength,
            canny_strength,
            style_strength,
            steps,
        ],
        fn=generate_image,
        label="Presets",
        examples_per_page=6,
    )

    generate_btn.click(
        fn=generate_image,
        inputs=[
            prompt_input,
            structure_image,
            style_image,
            depth_strength,
            canny_strength,
            style_strength,
            steps,
        ],
        outputs=[output_image],
    )

    gr.Markdown(
        """
    ## How to use
    1. Enter a prompt describing the image you want to generate
    2. Upload a structure image to provide the basic shape/composition
    3. Upload a style image to influence the visual style
    4. Adjust the sliders to control the effect strength
    5. Click "Generate" to create your image
    
    ## Follow us for more 
    If you enjoyed this project, you may also like ZenCtrl, our open-source agentic visual control toolkit for generative image pipelines that we are developing.
    ZenCtrl space : https://huggingface.co/spaces/fotographerai/ZenCtrl and 
    Discord : https://discord.com/invite/b9RuYQ3F8k
    """
    )

if __name__ == "__main__":
    # Create an examples directory if it doesn't exist , for now it is empty
    os.makedirs("examples", exist_ok=True)

    # Launch the app
    demo.launch(share=True)
