Audio Transcription & Meeting Minutes Generator

This project is a Gradio-based web application that allows you to upload or select an audio file from Google Drive, transcribe it using OpenAI's Whisper model, and generate structured meeting minutes using Meta LLaMA 3.1-8B-Instruct via Hugging Face.

Minutes include:


 Summary
 

 Key Discussion Points
 

 Takeaways
 

 Action Items with Owners
 

📦 Features

🔊 Transcribe audio using OpenAI Whisper


 Generate meeting minutes using LLaMA 3.1-8B-Instruct
 

 Read audio files from Google Drive
 

 Outputs formatted minutes in Markdown
 

 Streamed text generation with Transformers
 

 Easy-to-use Gradio interface


 Setup Instructions
 
1. Clone the GitHub Repo
   
 git clone https://github.com/your-username/audio-minutes-app.git
 
 cd audio-to-text

Replace your-username with your GitHub handle.

Open in Google Colab

The recommended environment is Google Colab, since it provides GPU and integrated Drive access.


 Upload the Notebook
 
Upload the .ipynb notebook to a new GitHub repo.


Open the notebook in Colab via:


Click the notebook on GitHub ➝ Click the Open in Colab badge (optional)


Or paste this Colab link manually:

https://colab.research.google.com/github/your-username/audio-minutes-app/blob/main/notebook.ipynb

3. 🔑 Authentication

Hugging Face

Store your HF token in Colab like this:


from google.colab import userdata

userdata.set('HF_TOKEN', 'your_huggingface_token')

Get your token from: https://huggingface.co/settings/tokens


OpenAI

Store your OpenAI API key:

userdata.set('OPENAI_API_KEY', 'your_openai_api_key')

Get your key from: https://platform.openai.com/account/api-keys


4. 📥 Mount Google Drive
   

from google.colab import drive

drive.mount('/content/drive')

Place your .mp3 or .wav files in /MyDrive/ and use their filenames in the app.


🚀 Run the App

Once all cells are run, the Gradio app will launch with:


demo.launch(share=True)

You’ll see a public URL like:

Running on public URL: https://your-gradio-app.gradio.live

🎯 How to Use

🗂 Tab 1: Upload Audio

Upload an audio file


Click Transcribe


Click Generate Minutes


🗃 Tab 2: Google Drive

Enter filename in /MyDrive/, e.g., meeting_recording.mp3


Click Transcribe from Drive


Click Generate Minutes


📚 Models Used

meta-llama/Meta-Llama-3.1-8B-Instruct (Hugging Face)


whisper-1 (OpenAI)


⚠️ Notes

Ensure GPU runtime is selected in Colab:

Runtime > Change Runtime Type > GPU


Whisper transcription might take ~30 seconds depending on file size.


LLaMA 3.1-8B requires a Hugging Face account with access approval.


Token streaming is used for real-time minute generation.


📌 Dependencies


pip install torch==2.5.1+cu124 torchvision==0.20.1+cu124 torchaudio==2.5.1+cu124 --index-url https://download.pytorch.org/whl/cu124

pip install requests bitsandbytes==0.46.0 transformers==4.48.3 accelerate==1.3.0 openai gradio
