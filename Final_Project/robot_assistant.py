# robot_assistant.py
import streamlit as st
from streamlit_lottie import st_lottie
from gtts import gTTS
import json
import os

# 1. Load robot animation
def load_robot_animation():
    with open("robot.json", "r", encoding="utf-8") as f:
        return json.load(f)

# 2. Dynamic messages for different ML steps
def get_robot_message(context):
    messages = {
        "upload": "You've uploaded your dataset — great! Think of it as your source of truth.",
        "select_features": "Now, choose which columns will help the model make predictions — these are features.",
        "select_target": "Pick the target — this is what you're trying to predict!",
        "train_model": "The model is learning patterns in the data — this is called training.",
        "metrics": "These numbers tell us how well your model did. Higher is usually better!",
        "predict": "Prediction time! Let's see what the model has learned.",
        "test_model": "You're testing the model on unseen data — like a final exam.",
        "distribution": "Distribution helps you understand how your data is spread — very useful!",
    }
    return messages.get(context, "Hi, I'm Robo! Let's learn ML together step-by-step.")

# 3. Audio support
def speak_text(text, filename="robot_message.mp3"):
    tts = gTTS(text)
    tts.save(filename)
    with open(filename, "rb") as f:
        st.audio(f.read(), format="audio/mp3")

# 4. Display robot assistant
def show_robot(context, show_audio=True):
    lottie = load_robot_animation()
    message = get_robot_message(context)

    col1, col2 = st.columns([1, 2])
    with col1:
        st_lottie(lottie, height=100, key=f"robot_{context}")
    with col2:
        st.markdown(f"**🤖 Robo says:** {message}")
        if show_audio and st.button("🔊 Hear Robo", key=f"audio_{context}"):
            speak_text(message)
