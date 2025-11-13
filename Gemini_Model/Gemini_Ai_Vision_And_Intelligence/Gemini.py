import streamlit as st
import os
import google.generativeai as genai
from PIL import Image
import pathlib
import textwrap

st.set_page_config(page_title="Gemini AI Vision & Intelligence", layout="wide")

os.environ['GEMINI_API_KEY'] = "AIzaSyBI-nZp0N5T9LYaJgdJ3M-j5j3fD_Gyhho"
genai.configure(api_key=os.environ['GEMINI_API_KEY'])

# Model
model_name = "models/gemini-2.5-flash-preview-09-2025"


# SIDEBAR
st.sidebar.title("⚙️ Settings")
model_name = st.sidebar.selectbox(
    "Choose Gemini Model",
    [
        "models/gemini-2.5-flash-preview-09-2025",
        "models/gemini-2.0-pro-exp-02-05"
    ]
)

model = genai.GenerativeModel(model_name)

st.sidebar.markdown("### 🔍 Token Counter")
input_text = st.sidebar.text_area("Enter text to calculate tokens")
if st.sidebar.button("Count Tokens"):
    tokens = model.count_tokens(input_text)
    st.sidebar.success(f"Tokens Used: {tokens.total_tokens}")


# Front
st.title("🤖 Gemini AI Vision & Intelligence")
st.write("A powerful Streamlit app using Gemini Vision + Text + Chat + Image-to-Text + Streaming")

tabs = st.tabs([
    "📋 Table Generator",
    "🖼️ Image Analysis",
    "🎥 Instagram Script Generator",
    "💬 Chatbot",
    "🧠 Image → Text Explanation (Lite Model)", 
    "📚 Model Explorer"                           
])

# TAB 1: TABLE GENERATOR
with tabs[0]:
    st.subheader("📋 Generate AI Comparison Table")

    user_prompt = st.text_area("Enter a topic:",
                               "Tabular format of difference between AI, ML, GenAI, DL")

    if st.button("Generate Table"):
        with st.spinner("Generating..."):
            response = model.generate_content(user_prompt)
            st.markdown(response.text)

# TAB 2: IMAGE ANALYSIS (VISION)

with tabs[1]:
    st.subheader("🖼️ Upload Image for AI Analysis")

    uploaded_img = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_img:
        img = Image.open(uploaded_img)
        st.image(img, caption="Uploaded Image", width=350)

        if st.button("Analyze Image"):
            with st.spinner("Analyzing..."):
                response = model.generate_content(img)
                st.markdown(response.text)

# TAB 3: INSTAGRAM SCRIPT (IMAGE + PROMPT)

with tabs[2]:
    st.subheader("🎥 AI Instagram Reel Script Generator")

    uploaded_img2 = st.file_uploader("Upload image for reel idea", type=["jpg","jpeg","png"])

    prompt = st.text_area("Enter your script prompt:",
                          "Write a 5-second Instagram reel script to gain good views")

    if uploaded_img2 and st.button("Generate Script"):
        img = Image.open(uploaded_img2)
        st.image(img, width=350)

        with st.spinner("Generating script..."):
            response = model.generate_content([prompt, img], stream=True)
            output = ""
            for chunk in response:
                if chunk.text:
                    output += chunk.text
                    st.write(chunk.text)

            st.success("Final Script:")
            st.write(output)


# TAB 4: CHATBOT
with tabs[3]:
    st.subheader("💬 Smart AI Chatbot")

    if "chat_session" not in st.session_state:
        st.session_state.chat_session = model.start_chat(history=[])

    user_msg = st.text_input("Enter your message:")

    if st.button("Send"):
        response = st.session_state.chat_session.send_message(user_msg)

        for message in st.session_state.chat_session.history:
            role = "**You:**" if message.role == "user" else "**AI:**"
            st.markdown(f"{role} {message.parts[0].text}")


# TAB 5: Image → Text Explanation (Lite Model)
with tabs[4]:
    st.subheader("🧠 Image → Text Explanation (Using: gemini-2.5-flash-lite-preview-06-17)")

    def get_gemini_response(input_text, image):
        model_lite = genai.GenerativeModel("models/gemini-2.5-flash-lite-preview-06-17")
        if input_text:
            response = model_lite.generate_content([input_text, image])
        else:
            response = model_lite.generate_content(image)
        return response.text

    input_prompt = st.text_input("Enter your prompt for the image:")
    uploaded_file = st.file_uploader("Upload an image for explanation...", type=["jpg", "jpeg", "png"])

    image = ""
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("Explain Image"):
        if image:
            with st.spinner("Generating explanation..."):
                response = get_gemini_response(input_prompt, image)
                st.subheader("AI Explanation:")
                st.write(response)
        else:
            st.warning("Please upload an image first.")

# TAB 6: Model Explorer 
with tabs[5]:
    st.subheader("📚 Explore Available Gemini Models")

    if st.button("Show Models"):
        models = genai.list_models()
        for m in models:
            if "generateContent" in m.supported_generation_methods:
                st.write(f"- {m.name}")

