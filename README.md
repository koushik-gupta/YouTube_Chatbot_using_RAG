# YouTube Chatbot

A minimal YouTube transcript chatbot built with Streamlit, LangChain, Hugging Face, and Chroma.

Paste a YouTube video URL, ask a question, and the app answers using the video's available captions.

## Features

- Fetches YouTube captions/transcripts
- Splits transcripts into searchable chunks
- Uses Hugging Face embeddings with Chroma retrieval
- Answers questions with `meta-llama/Llama-3.1-8B-Instruct`
- Provides both Streamlit and terminal interfaces

## Setup

1. Create and activate a virtual environment.

```powershell
python -m venv myenv
.\myenv\Scripts\Activate.ps1
```

2. Install dependencies.

```powershell
pip install -r requirements.txt
```

3. Create a `.env` file with your Hugging Face token.

```env
HUGGINGFACEHUB_API_TOKEN=your_token_here
```

## Run

Start the Streamlit app:

```powershell
streamlit run app.py
```

Or run the terminal version:

```powershell
python main.py
```

## Notes

- The video must have captions available.
- Local secrets, caches, and the virtual environment are excluded by `.gitignore`.
