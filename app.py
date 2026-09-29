import streamlit as st
import json
import random
import datetime
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit.components.v1 as components

st.set_page_config(
    page_title="SunFlix AI ☀️",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* Hide default streamlit headers and footers */
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Main Background Override */
    [data-testid="stAppViewContainer"], [data-testid="stApp"] {
        background: linear-gradient(135deg, #0b0f19 0%, #1a1f35 100%) !important;
        color: #f8fafc;
        font-family: 'Inter', sans-serif;
    }
    [data-testid="stHeader"] {
        background: rgba(0,0,0,0) !important;
        }

    /* Style the Text Area Container and Input */
    div[data-baseweb="textarea"] {
        background: rgba(255, 255, 255, 0.05) !important;
         border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 12px !important;
        transition: all 0.3s ease;
    }
    div[data-baseweb="textarea"] > div {
    background: transparent !important;
    }
    div[data-baseweb="textarea"] textarea {
        color: #fff !important;
        font-size: 1.1rem !important;
        padding: 15px !important;
    }
    div[data-baseweb="textarea"]:focus-within {
        border-color: #ff4b4b !important;
        box-shadow: 0 0 15px rgba(255, 75, 75, 0.4) !important;
        }
    
    /* Text Area Label */
    .stTextArea label, label[data-testid="stWidgetLabel"] {
        font-size: 1.1rem !important;
        font-weight: 600 !important;
        color: #e2e8f0 !important;
    }

    /* Primary Generate Button */
     button[kind="primary"] {
        background: linear-gradient(45deg, #ff4b4b, #ff8f00) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 10px 24px !important;
        font-weight: 700 !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(255, 75, 75, 0.3) !important;
    }
    button[kind="primary"]:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(255, 75, 75, 0.5) !important;
        border: none !important;
    }

    /* Secondary Buttons */
    button[kind="secondary"] {
        background: rgba(255, 255, 255, 0.05) !important;
        color: #e2e8f0 !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 10px !important;
        padding: 10px 24px !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
        }
    button[kind="secondary"]:hover {
        background: rgba(255, 255, 255, 0.15) !important;
        border-color: #cbd5e1 !important;
        color: #fff !important;
        }

    /* Multiselect and Select Slider Tweaks */
    div[data-baseweb="select"] {
        background: rgba(255, 255, 255, 0.05) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 8px !important;
        color: white !important;
    }
    div[data-baseweb="select"] > div {
    div[data-baseweb="select"] > div {
        background: transparent !important;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
    background: rgba(11, 15, 25, 0.8) !important;
        backdrop-filter: blur(15px);
        border-right: 1px solid rgba(255, 255, 255, 0.05) !important;
    }
    /* Expander UI */
    [data-testid="stExpander"] {
        background: rgba(255, 255, 255, 0.03) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 8px !important;
         }
</style>
""", unsafe_allow_html=True)

if "history" not in st.session_state:
     st.session_state.history = []

@st.cache_data
def load_database():
    with open("database.json", "r") as f:
         return json.load(f)

@st.cache_resource
def load_model():
    print("Loading SentenceTransformer model...")
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_data
def compute_embeddings(_model, movies):
    print("Computing movie embeddings...")
    descriptions = [m['description'] for m in movies]
    return _model.encode(descriptions)

movies = load_database()
model = load_model()
movie_embeddings = compute_embeddings(model, movies)

def get_recommendations(user_query: str, intensity: str, preferred_languages: list = None):
    valid_langs = {"english", "hindi"}
    if preferred_languages and len(preferred_languages) > 0:
        pref_langs_lower = {lang.lower() for lang in preferred_languages}
        active_langs = pref_langs_lower.intersection(valid_langs)
        if not active_langs:
            active_langs = valid_langs
    else:
        active_langs = valid_langs

    filtered_movies = [m for m in movies if m['language'].lower() in active_langs]

    if not filtered_movies:
        return []

    query_embedding = model.encode([user_query])
    
    similarities = cosine_similarity(query_embedding, movie_embeddings)[0]
        
    scored_movies = []
    for idx, movie in enumerate(movies):
        if movie['language'].lower() not in active_langs:
            continue
                   
        score = float(similarities[idx])
        if intensity == "Low":
            score *= random.uniform(0.8, 1.0)
        elif intensity == "Intense":
            score *= 1.2
            
        scored_movies.append({
            "movie": movie,
            "score": score
            })
        
    scored_movies.sort(key=lambda x: x['score'], reverse=True)
    top_10 = scored_movies[:10]

    if len(top_10) >= 3:
        sampled = random.sample(top_10, 3)
    else:
        sampled = top_10

    sampled.sort(key=lambda x: x['score'], reverse=True)
    return [sm['movie'] for sm in sampled]

def get_surprise_recommendation():
    hour = datetime.datetime.now().hour
    time_mood = "I want to relax and wind down"
    if 6 <= hour < 12:
        time_mood = "I need high energy, motivation, and inspiration to start the day"
    elif 12 <= hour < 17:
        time_mood = "I am bored and need something fast-paced, exciting, and highly engaging"
    elif 17 <= hour < 22:
        time_mood = "I want a great drama or an emotionally engaging story to watch after work"
    else:
        time_mood = "I want something dark, psychological, or thought-provoking for a late night watch"

        return get_recommendations(time_mood, "Medium", ["English", "Hindi"])

def render_movie_card(movie):
    platforms_html = ""
    for p in movie.get("where_to_watch", []):
        p_class = p.split()[0].replace("+", "")
        platforms_html += f"<span class='badge badge-{p_class}'>{p}</span>"
        
    html = f"""
    <style>
     @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
        body {{ font-family: 'Inter', sans-serif; color: #f8fafc; margin: 0; padding: 0; }}
        .glass-card {{ background: rgba(255, 255, 255, 0.05); backdrop-filter: blur(10px); border: 1px solid rgba(255, 255, 255, 0.1); border-radius: 20px; padding: 25px; margin-bottom: 25px; }}
        .movie-title {{ font-size: 2rem; font-weight: 800; background: linear-gradient(45deg, #ff4b4b, #ff8f00); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 15px; }}
        .metadata-grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px; margin-bottom: 20px; }}
         .meta-box {{ background: rgba(0,0,0,0.3); padding: 15px; border-radius: 12px; text-align: center; }}
        .meta-box h4 {{ margin: 0; font-size: 0.8rem; color: #94a3b8; text-transform: uppercase; }}
        .meta-box p {{ margin: 5px 0 0 0; font-size: 1.1rem; font-weight: 600; color: #e2e8f0; }}
        .rating-wrap {{ display: inline-flex; align-items: center; gap: 8px; background: rgba(255,255,255,0.1); padding: 5px 12px; border-radius: 15px; margin-right: 10px; }}
        .badge {{ display: inline-block; padding: 6px 12px; border-radius: 20px; font-size: 0.85rem; font-weight: bold; margin-right: 8px; color: white; }}
        .badge-Netflix {{ background: linear-gradient(45deg, #E50914, #900000); }}
        .badge-Prime {{ background: linear-gradient(45deg, #00A8E1, #005A9E); }}
        .badge-Disney {{ background: linear-gradient(45deg, #113CCF, #000066); }}
        .badge-Hulu {{ background: linear-gradient(45deg, #1ce783, #0b6b3c); color: black; }}
        .badge-HBO {{ background: linear-gradient(45deg, #511257, #2c0830); }}
        .badge-Apple {{ background: linear-gradient(45deg, #a6b1b7, #555555); color: black; }}
    </style>
    <div class="glass-card">
        <div class="movie-title">{movie['title']}</div>
        
        <div style="margin-bottom: 20px;">
            <span class="rating-wrap">⭐ {movie['imdb_rating']} IMDb</span>
            <span class="rating-wrap">🍅 {movie['rotten_tomatoes']} RT</span>
            <span class="rating-wrap">🗣️ {movie['language']}</span>
        </div>
               
        <p style="font-size: 1.1rem; color: #cbd5e1; line-height: 1.6; margin-bottom: 20px;">
            {movie['description']}
        </p>
        
         <div class="metadata-grid">
            <div class="meta-box">
                <h4>Primary Emotion</h4>
                <p>{movie.get('primary_emotion', 'N/A')}</p>
            </div>
            <div class="meta-box">
                <h4>Tone Check</h4>
                <p>{movie.get('tone_check', 'N/A')}</p>
            </div>
            <div class="meta-box">
            <h4>Why Watch This</h4>
                <p style="font-size: 0.9rem;">{movie.get('why_watch', 'N/A')}</p>
            </div>
        </div>
        
        <div class="badge-container">
            <span style="color: #94a3b8; font-size: 0.9rem; margin-right: 10px;">Available on:</span>
            {platforms_html}
        </div>
    </div>