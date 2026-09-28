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