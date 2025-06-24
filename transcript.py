# Install and Import

!pip install -q --upgrade torch==2.5.1+cu124 torchvision==0.20.1+cu124 torchaudio==2.5.1+cu124 --index-url https://download.pytorch.org/whl/cu124
!pip install -q requests bitsandbytes==0.46.0 transformers==4.48.3 accelerate==1.3.0 openai
!pip install gradio
import os
import threading
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer, BitsAndBytesConfig
from huggingface_hub import login
from google.colab import drive, userdata
from openai import OpenAI
import gradio as gr


# Setup

# Constants
AUDIO_MODEL = "whisper-1"
LLAMA = "meta-llama/Meta-Llama-3.1-8B-Instruct"

# Authentication
drive.mount("/content/drive")
login(userdata.get('HF_TOKEN'), add_to_git_credential=True)
openai = OpenAI(api_key=userdata.get('OPENAI_API_KEY'))

# Initialize model
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4"
)

tokenizer = AutoTokenizer.from_pretrained(LLAMA)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    LLAMA, device_map="auto", quantization_config=quant_config)

# Functions


def transcribe_audio(audio_file):
    if audio_file is None:
        return "No audio file provided"

    with open(audio_file, "rb") as file:
        transcription = openai.audio.transcriptions.create(
            model=AUDIO_MODEL, file=file, response_format="text")
    return transcription


def transcribe_from_drive(file_path):
    full_path = f"/content/drive/MyDrive/{file_path}"
    if os.path.exists(full_path):
        return transcribe_audio(full_path)
    return f"File not found: {full_path}"


def generate_minutes_stream(transcription):
    if not transcription or transcription.startswith("No audio") or transcription.startswith("File not found"):
        yield "Please transcribe audio first"
        return

    system_message = "You are an assistant that produces minutes of meetings from transcripts, with summary, key discussion points, takeaways and action items with owners, in markdown."
    user_prompt = f"Below is a transcript. Write minutes in markdown with summary, discussion points, takeaways, and action items:\n{transcription}"

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_prompt}
    ]

    inputs = tokenizer.apply_chat_template(
        messages, return_tensors="pt").to("cuda")
    streamer = TextIteratorStreamer(
        tokenizer, skip_prompt=True, skip_special_tokens=True)

    generation_kwargs = {
        "input_ids": inputs,
        "max_new_tokens": 2000,
        "streamer": streamer,
        "temperature": 0.7,
        "pad_token_id": tokenizer.eos_token_id
    }

    thread = threading.Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    generated_text = ""
    for new_text in streamer:
        generated_text += new_text
        yield generated_text

#  Gradio Interface


with gr.Blocks() as demo:
    gr.Markdown("# Audio Transcription & Meeting Minutes")

    with gr.Tab("Upload Audio"):
        audio_input = gr.Audio(type="filepath")
        transcribe_btn = gr.Button("Transcribe")
        transcription_output = gr.Textbox(lines=8, label="Transcription")

    with gr.Tab("Google Drive"):
        drive_path = gr.Textbox(label="File Path", value="denver_extract.mp3")
        drive_btn = gr.Button("Transcribe from Drive")
        drive_output = gr.Textbox(lines=8, label="Transcription")

    generate_btn = gr.Button("Generate Minutes")
    minutes_output = gr.Markdown()

    # Events
    transcribe_btn.click(transcribe_audio, [audio_input], [
                         transcription_output])
    drive_btn.click(transcribe_from_drive, [drive_path], [drive_output])
    generate_btn.click(generate_minutes_stream, [
                       transcription_output], [minutes_output])

demo.launch(share=True)
