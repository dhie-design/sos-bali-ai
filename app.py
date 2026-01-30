import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 1. Setup
st.set_page_config(page_title="AI Food Identifier", page_icon="🥗")
st.title("🥗 AI Food Identifier")

# 2. Connect to your Google Sheet (PASTE YOUR LINK BELOW)
GOOGLE_SHEET_CSV_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vR4mdRkqAP4qYj1Ip4d3aNldc9gtKDb5wY1bEuZRGxL4PI5FbR6EEhVIsupJH6Az4FATRLPGmnJyDGh/pub?gid=315018304&single=true&output=csv"

@st.cache_data
def load_data():
    return pd.read_csv(GOOGLE_SHEET_CSV_URL)

try:
    df = load_data()
    user_input = st.text_input("Type food name (e.g., 'lettuce' or 'paskez'):")

    if user_input:
        # AI Logic
        vectorizer = TfidfVectorizer()
        names = df['product_name'].astype(str).tolist()
        tfidf = vectorizer.fit_transform(names + [user_input])
        sim = cosine_similarity(tfidf[-1], tfidf[:-1])
        idx = sim.argmax()
        
        res = df.iloc[idx]
        
        # UI Results
        st.success(f"Matched: {res['product_name']}")
        st.info(f"Category: {res['categories']}")
        st.image(res['url'], caption="Product Verification")
        st.write(f"**AI Confidence:** {round(sim.max()*100, 1)}%")

except Exception as e:
    st.warning("Connect your Google Sheet link in the code to start.")
