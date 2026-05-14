# YouTube Chatbot

A minimal YouTube transcript chatbot built with Streamlit, LangChain, Hugging Face, Chroma, and Retrieval-Augmented Generation (RAG).

Paste a YouTube video URL, ask a question, and the app answers by retrieving relevant transcript chunks before generating a response.

## Features

- Fetches YouTube captions/transcripts
- Splits transcripts into searchable chunks
- Uses a RAG pipeline with Hugging Face embeddings and Chroma retrieval
- Answers questions with `meta-llama/Llama-3.1-8B-Instruct`
- Provides both Streamlit and terminal interfaces

## How It Works

1. The app fetches the video's transcript.
2. The transcript is split into smaller chunks.
3. Chunks are embedded and stored in Chroma.
4. Relevant chunks are retrieved for each question.
5. The LLM generates an answer using the retrieved context.

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
