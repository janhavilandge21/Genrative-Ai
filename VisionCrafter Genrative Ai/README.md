# 🧠 VisionCrafter – Text-to-Image Generation using CLIP & VQGAN

VisionCrafter is a Generative AI project that converts natural-language prompts into high-quality images by combining OpenAI’s CLIP for text–image alignment and VQGAN (Vector Quantized Generative Adversarial Network) for image synthesis.
It demonstrates the creative fusion of language and vision models, enabling AI-driven art generation directly from user imagination.

# 🌟 Features

🧩 CLIP + VQGAN Integration – bridges text understanding and image generation.


🎨 Prompt-based Creativity – generates unique, AI-crafted visuals from user input.


⚡ GPU-Accelerated Optimization – powered by PyTorch and CUDA.


📈 Iterative Visualization – view intermediate outputs to track generation progress.


📹 Smooth Interpolation – creates transition videos between generated images.


💻 Streamlit Web App (VisionCrafter Studio) – interactive interface for HRs & recruiters to try text-to-image generation live.

# 🧰 Tech Stack
Component	Technology

Language	Python

Frameworks	PyTorch, Streamlit

Models	CLIP (OpenAI), VQGAN (CompVis)

Libraries	torch, torchvision, omegaconf, taming-transformers, clip, ftfy, regex, tqdm, imageio

Deployment	Streamlit Cloud / Hugging Face Spaces

# 🚀 Installation
1. Install Dependencies

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

pip install streamlit git+https://github.com/openai/CLIP.git git+https://github.com/CompVis/taming-transformers.git

pip install omegaconf pytorch-lightning tqdm ftfy regex imageio einops


Download model files

File	Description	Link

model.yaml	VQGAN configuration	Download

last.ckpt	Pre-trained weights	Download

Save them into the respective configs/ and checkpoints/ folders.

# 🖥️ Run the Streamlit App

streamlit run VisionCrafter.py

Then open the local URL in your browser (usually http://localhost:8501
).

# 🧩 Streamlit UI Highlights

App Title: 🎨 VisionCrafter AI Studio
Inputs:

Text Prompt

Optional Negative Prompt (Exclude words)

Iteration & Noise Controls

Outputs:

Real-time generated images

Downloadable final image

Optional interpolation video

# 📊 Project Workflow
User Prompt
    ↓
Text Encoding via CLIP
    ↓
Latent Representation Optimization in VQGAN
    ↓
Image Generation & Refinement
    ↓
Final AI-Generated Image

# 🧠 Core Concepts

CLIP (Contrastive Language–Image Pretraining)
Used to understand the semantics of the text prompt and guide the generation toward meaningful image features.

VQGAN (Vector-Quantized GAN)
Generates high-resolution images from latent representations, balancing fidelity and creativity.

Loss Optimization
Uses cosine similarity between text and image embeddings to iteratively refine visuals.

# 💡 Example Prompts
Input Prompt	Description
“A futuristic city at sunset”	Generates a glowing cyberpunk skyline.
“An astronaut riding a horse on Mars”	Creates a surreal space-art scene.
“A forest with neon trees and mist”	Outputs fantasy landscapes.
🎬 Output Samples

🖼️ images/sample1.png
🖼️ images/sample2.png
🖼️ images/sample3.png

📹 Optional – Generate Video from Interpolations

You can merge generated frames into a video:

writer = imageio.get_writer('output.mp4', fps=25)
for img in frames:
    writer.append_data(np.array(img))
writer.close(


# 🏆 Achievements
✅ Created a functional Text-to-Image AI Generator using research-grade models.

✅ Implemented CLIP–VQGAN architecture from scratch.

✅ Built and deployed Streamlit interface for interactive use.

✅ Demonstrated expertise in Deep Learning, Generative AI, and Python.

# 🧩 Keywords

#GenerativeAI 

#VQGAN 

#CLIP 

#ComputerVision 

#DeepLearning

#Streamlit 

#DataScience 

#AIProjects
