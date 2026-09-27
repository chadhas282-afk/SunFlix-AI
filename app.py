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