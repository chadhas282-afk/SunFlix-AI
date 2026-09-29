# Local Mood-Based Movie Recommender

A completely offline, self-contained AI recommender system that matches your mood to movie plots using local vector embeddings (via `sentence-transformers`).

## Architecture
- **NLP Engine:** `all-MiniLM-L6-v2` via `sentence-transformers` (runs 100% locally).
- **Database:** Local JSON dataset (`database.json`).
- **Backend:** FastAPI.
- **Frontend:** Streamlit with a custom dark-themed UI.

## Setup Instructions

1. **Create and activate a virtual environment (Optional but recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On macOS/Linux
   # venv\Scripts\activate   # On Windows
   ```

2. **Install the dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Running the Application

You need to run two separate processes (one for the backend and one for the frontend).

1. **Start the FastAPI Backend:**
   Open a terminal and run:
   ```bash
   uvicorn main:app --reload
   ```
   *The server will start on http://127.0.0.1:8000*
   *(Note: The first time it runs, it will download the lightweight `all-MiniLM-L6-v2` model which is ~90MB).*

2. **Start the Streamlit Frontend:**
   Open a second terminal and run:
   ```bash
   streamlit run app.py
   ```
   *The UI will open in your browser automatically.*

## How to use
Type a mood like "I want to watch an intense mind-bending sci-fi movie" or "I'm feeling sad, I need a heartwarming comedy". The local NLP engine will semantically match your input against the movie plot descriptions in `database.json`.
