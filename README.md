# Ravidu-AI
index.com 
import gradio as gr
from gradio_client import Client
from moviepy.editor import VideoFileClip, concatenate_videoclips
from PIL import Image, ImageEnhance, ImageFilter
from gtts import gTTS
import os

# 1. Chat AI (Sinhala & Live)
def chat_ai(message):
    if not message.strip(): return "කරුණාකර ප්‍රශ්නයක් ලියන්න."
    try:
        client = Client("Qwen/Qwen2.5-72B-Instruct")
        res = client.predict(query=message, history=[], system_prompt="Respond in Sinhala.", api_name="/model_chat")
        return res[1][0][1]
    except: return "Chat Error!"

# 2. Voice Generator (Text to Speech)
def text_to_speech(text):
    if not text.strip(): return None
    tts = gTTS(text=text, lang='si')
    tts.save("voice.mp3")
    return "voice.mp3"

# 3. Photo Generator (Text to Image)
def generate_photo(prompt):
    if not prompt.strip(): return None
    try:
        client = Client("black-forest-labs/FLUX.1-schnell")
        res = client.predict(prompt=prompt, seed=0, randomize_seed=True, width=1024, height=1024, num_inference_steps=4, api_name="/predict")
        return res[0]
    except: return None

# 4. Photo Editor (Brightness & Blur)
def edit_photo(image, brightness, blur):
    if image is None: return None
    img = Image.fromarray(image)
    if brightness != 1.0:
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(brightness)
    if blur > 0:
        img = img.filter(ImageFilter.GaussianBlur(radius=blur))
    return img

# 5. Video Generator (Text to Video)
def generate_video(prompt):
    if not prompt.strip(): return None
    try:
        client = Client("Wan-AI/Wan2.1-T2V-1.3B")
        res = client.predict(prompt=prompt, api_name="/generate")
        return res
    except: return None

# 6. Photo to Video Generator
def photo_to_video(image_path, prompt):
    if not image_path: return None
    try:
        client = Client("Wan-AI/Wan2.1-I2V-480P")
        res = client.predict(image=image_path, prompt=prompt, api_name="/generate")
        return res
    except: return None

# 7. Video Editor (Join Videos)
def edit_videos(v1, v2):
    if not v1 or not v2: return None
    try:
        c1 = VideoFileClip(v1)
        c2 = VideoFileClip(v2)
        final = concatenate_videoclips([c1, c2])
        out = "edited_video.mp4"
        final.write_videofile(out, codec="libx264", audio_codec="aac")
        return out
    except: return None

# App UI Architecture
with gr.Blocks(theme=gr.themes.Soft()) as app:
    gr.Markdown("# 🚀 Ravidu Ultimate AI Studio")
    
    with gr.Tab("💬 Live Chat"):
        c_in = gr.Textbox(label="Message")
        c_btn = gr.Button("Send")
        c_out = gr.Textbox(label="Response")
        c_btn.click(chat_ai, c_in, c_out)

    with gr.Tab("🎙️ Voice Gen"):
        v_in = gr.Textbox(label="Text to Speak")
        v_btn = gr.Button("Generate Voice")
        v_out = gr.Audio(label="Audio Output")
        v_btn.click(text_to_speech, v_in, v_out)

    with gr.Tab("🎨 Photo Gen"):
        p_in = gr.Textbox(label="Photo Prompt")
        p_btn = gr.Button("Generate Photo")
        p_out = gr.Image(label="Output Image")
        p_btn.click(generate_photo, p_in, p_out)

    with gr.Tab("🖼️ Photo Editor"):
        e_img = gr.Image(label="Upload Photo")
        e_bright = gr.Slider(0.5, 2.0, value=1.0, label="Brightness")
        e_blur = gr.Slider(0, 10, value=0, label="Blur")
        e_btn = gr.Button("Apply Edits")
        e_out = gr.Image(label="Edited Photo")
        e_btn.click(edit_photo, inputs=[e_img, e_bright, e_blur], outputs=e_out)

    with gr.Tab("🎬 Text to Video"):
        tv_in = gr.Textbox(label="Video Prompt")
        tv_btn = gr.Button("Generate Video")
        tv_out = gr.Video(label="Output Video")
        tv_btn.click(generate_video, tv_in, tv_out)

    with gr.Tab("🖼️➡️🎬 Photo to Video"):
        pv_img = gr.Image(type="filepath", label="Upload Photo")
        pv_in = gr.Textbox(label="Motion Prompt")
        pv_btn = gr.Button("Animate Photo")
        pv_out = gr.Video(label="Output Video")
        pv_btn.click(photo_to_video, inputs=[pv_img, pv_in], outputs=pv_out)

    with gr.Tab("✂️ Video Editor"):
        v1 = gr.Video(label="Video 1")
        v2 = gr.Video(label="Video 2")
        ve_btn = gr.Button("Join Videos")
        ve_out = gr.Video(label="Output Video")
        ve_btn.click(edit_videos, inputs=[v1, v2], outputs=ve_out)

app.launch()
