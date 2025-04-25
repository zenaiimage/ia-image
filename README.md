AIImage / FLUX
Overview
AIImage is a work-in-progress (WIP) project aimed at generating multi-view, diverse-scene, and task-specific high-resolution images from a single subject image without requiring fine-tuning. The project leverages a unified Large Language Model (LLM) to streamline various visual content generation tasks. This repository contains the code for a demo interface built with Gradio, integrating with Baseten for image generation.
Features

Task-Specific Modes: Supports multiple generation modes, including:
Subject Generation: Creates detailed subject portraits.
Background Generation: Produces vibrant, dynamic backgrounds with partial sketch-based generation.
Canny: Focuses on strong edge detection for futuristic visuals.
Depth: Generates images with realistic depth and perspective.
Deblurring: Enhances image clarity.


Customizable Parameters: Adjust steps, strength, height, width, and background removal settings.
Interactive UI: Built with Gradio, offering an intuitive interface for image input, prompt customization, and model selection.
Preset Examples: Includes example inputs and prompts for each mode to demonstrate capabilities.
Baseten Integration: Uses Baseten API for image generation (requires API key and URL configuration).

Installation

Clone the Repository:
git clone https://github.com/FotographerAI/AIImage.git
cd AIImage


Install Dependencies:Ensure Python 3.8+ is installed, then install the required packages:
pip install -r requirements.txt

Required packages include:

gradio
requests
Pillow
python-decouple (optional, for environment variable management)


Set Environment Variables:Configure the Baseten API key and URL by setting the following environment variables:
export API_KEY="your_baseten_api_key"
export URL="your_baseten_endpoint_url"

Alternatively, use a .env file with python-decouple:
API_KEY=your_baseten_api_key
URL=your_baseten_endpoint_url



Usage

Run the Demo:Launch the Gradio interface:
python main.py

This will start a local server, and a URL will be provided to access the interface in your browser.

Interact with the Interface:

Select a mode (e.g., Subject Generation, Background Generation).
Upload an input image or use preset examples.
Customize the prompt, model, and generation parameters (steps, strength, dimensions, etc.).
Click "Generate" to produce the output image.


Explore Presets:Each mode includes example inputs, prompts, and outputs to help you understand the capabilities of AIImage.


Modes and Configuration
AIImage supports the following modes with their default configurations:



Mode
Model Example
Default Prompt
Strength
Resolution
Remove Background



Background Generation
bg_canny_58000_1024
A vibrant background with dynamic lighting
1.2
1024x1024
Yes


Subject Generation
subject_99000_512
A detailed portrait with soft lighting
1.2
512x512
Yes


Canny
canny_21000_1024
A futuristic cityscape with neon lights
1.2
1024x1024
Yes


Depth
depth_9800_1024
A scene with pronounced depth and perspective
1.2
1024x1024
Yes


Deblurring
deblurr_1024_10000
A scene with pronounced depth and perspective
1.2
1024x1024
No


Each mode supports multiple models, accessible via a dropdown in the UI.
Contributing
We welcome contributions! To contribute:

Fork the repository.
Create a new branch for your feature or bug fix.
Submit a pull request with a clear description of your changes.

For more details or to discuss ideas, join our Discord community.
Links

GitHub Repository
Hugging Face Space
Discord Community

License
This project is licensed under the MIT License. See the LICENSE file for details.
Notes

The project is in active development, with plans to release additional task-specific weights and code.
The goal is to unify all visual content generation tasks using a single LLM.
For API-related queries or issues, ensure your Baseten API key and URL are correctly configured.

