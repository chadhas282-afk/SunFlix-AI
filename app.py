import streamlit as st
import json
import random
import datetime
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit.components.v1 as components

st.set_page_config(
    page_title="SunFlix AI ☀️",