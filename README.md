# 🌟 AIImage / FLUX 🚀

Welcome to **AIImage**! 🎉 This is an exciting work-in-progress (WIP) project designed to generate **multi-view**, **diverse-scene**, and **task-specific high-resolution images** from a single subject image—*without fine-tuning*! Powered by a unified Large Language Model (LLM), AIImage is your go-to tool for creating stunning visual content with ease. 😎

This repository contains a vibrant demo interface built with **Gradio** and integrated with **Baseten** for image generation. Let's dive into the magic! ✨

---

## 🎨 Features That Wow

- **Task-Specific Modes**: Choose from a variety of modes to suit your creative needs:
  - **Subject Generation**: Craft detailed, portrait-style images with soft lighting. 📸
  - **Background Generation**: Generate vibrant, dynamic backgrounds, with partial sketch-based creation. 🌌
  - **Canny**: Create futuristic visuals with strong edge detection. 🏙️
  - **Depth**: Produce images with realistic depth and perspective. 🏞️
  - **Deblurring**: Sharpen blurry images for crystal-clear results. 🔍
- **Customizable Parameters**: Tweak steps, strength, height, width, and background removal to get the perfect output. 🎚️
- **Interactive UI**: A user-friendly Gradio interface makes image generation a breeze. 🖼️
- **Preset Examples**: Try out pre-configured inputs and prompts to see AIImage in action! 🚀
- **Baseten Integration**: Seamlessly generate images using the Baseten API (requires API key and URL). 🌐

---

## 🛠️ Installation

Get started in just a few steps! 🏃‍♂️

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/zenaiimage/AIImage.git
   cd AIImage
   ```

2. **Install Dependencies**:
   Make sure you have **Python 3.8+**, then install the required packages:
   ```bash
   pip install -r requirements.txt
   ```
   Key packages include:
   - `gradio` 🎨
   - `requests` 🌐
   - `Pillow` 🖼️
   - `python-decouple` (optional, for environment variables) ⚙️

3. **Set Environment Variables**:
   Configure your **Baseten API key** and **URL**:
   ```bash
   export API_KEY="your_baseten_api_key"
   export URL="your_baseten_endpoint_url"
   ```
   Or use a `.env` file with `python-decouple`:
   ```plaintext
   API_KEY=your_baseten_api_key
   URL=your_baseten_endpoint_url
   ```

---

## 🚀 Usage

Ready to create some visual magic? Here's how! 🪄

1. **Run the Demo**:
   Launch the Gradio interface:
   ```bash
   python main.py
   ```
   This starts a local server, and you'll get a URL to access the interface in your browser. 🌐

2. **Play with the Interface**:
   - Pick a mode (e.g., **Subject Generation**, **Background Generation**).
   - Upload an image or use preset examples. 🖼️
   - Customize the prompt, model, and parameters (steps, strength, etc.). ✍️
   - Hit **Generate** and watch the magic happen! 🎉

3. **Explore Presets**:
   Each mode comes with example inputs, prompts, and outputs to spark your creativity. 🔥

---

## 🎯 Modes & Configurations

AIImage offers a range of modes, each with tailored settings for stunning results. Here's a quick look:

| **Mode**               | **Default Model**         | **Default Prompt**                              | **Strength** | **Resolution** | **Remove BG** |
|------------------------|---------------------------|------------------------------------------------|--------------|----------------|---------------|
| **Background Generation** | `bg_canny_58000_1024`    | A vibrant background with dynamic lighting     | 1.2          | 1024x1024      | Yes           |
| **Subject Generation**    | `subject_99000_512`      | A detailed portrait with soft lighting        | 1.2          | 512x512        | Yes           |
| **Canny**                | `canny_21000_1024`       | A futuristic cityscape with neon lights       | 1.2          | 1024x1024      | Yes           |
| **Depth**                | `depth_9800_1024`        | A scene with pronounced depth and perspective | 1.2          | 1024x1024      | Yes           |
| **Deblurring**           | `deblurr_1024_10000`     | A scene with pronounced depth and perspective | 1.2          | 1024x1024      | No            |

Each mode supports multiple models, selectable via a dropdown in the UI. Experiment to find your perfect combo! 🎨

---

## 🤝 Contributing

We ❤️ contributions! Want to make AIImage even better? Here's how to join the fun:
1. **Fork** the repository. 🍴
2. Create a new branch for your feature or fix. 🌿
3. Submit a **Pull Request** with a clear description. 📬


---

## 🔗 Links & Resources

- 📂 [GitHub Repository](https://github.com/zenaiimage/AIImage)

---

## 📜 License

AIImage is licensed under the **MIT License**. Check out the `LICENSE` file for details. 📝

---

## 🌈 Notes & Future Plans

- **Work in Progress**: We're actively developing AIImage, with more task-specific weights and code releases coming soon! 🚧
- **Unified Vision**: Our goal is to streamline all visual content generation tasks with a single LLM. Stay tuned for updates! 🌟
- **API Tips**: Ensure your Baseten API key and URL are set correctly to avoid hiccups. ⚙️

Got feedback or need help? Ping us on [Discord](https://discord.com/invite/b9RuYQ3F8k) or open an issue on GitHub. Let's create something amazing together! 🎉
