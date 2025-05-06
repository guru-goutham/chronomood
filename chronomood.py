import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

st.set_page_config(page_title="Chronomood", layout="wide")

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("sample_data/sample_combined.csv", parse_dates=['datetime'])
    return df

df = load_data()
df['hour'] = df['datetime'].dt.hour
df['day'] = df['datetime'].dt.day_name()

st.title("🧠 Chronomood: Tracking Emotional Patterns by Time")

# Sidebar
st.sidebar.header("Filters")
day_options = sorted(df['day'].dropna().astype(str).unique())
day = st.sidebar.selectbox("Day of Week", day_options)
hour = st.sidebar.slider("Hour of Day", 0, 23)

# Filtered Data
filtered = df[(df['day'] == day) & (df['hour'] == hour)]

# --- Visualization Section ---
st.subheader(f"Sentiment Distribution at {hour}:00 on {day}")
fig, ax = plt.subplots()
sns.countplot(data=filtered, x='sentiment', ax=ax, palette='coolwarm')
st.pyplot(fig)

# --- Word Cloud ---
st.subheader("Word Cloud for Selected Time")
sentiment = st.radio("Choose sentiment:", ['positive', 'negative'])
text = " ".join(filtered[filtered['sentiment'] == sentiment]['text'].astype(str))

if text:
    wc = WordCloud(width=800, height=300, background_color="white").generate(text)
    st.image(wc.to_array(), use_column_width=True)
else:
    st.warning("No tweets found for this selection.")

# --- Mood Prediction ---
st.subheader("Test the Mood of Your Tweet")
user_input = st.text_area("Write a tweet here:")
if user_input:
    analyzer = SentimentIntensityAnalyzer()
    score = analyzer.polarity_scores(user_input)['compound']
    mood = "positive" if score >= 0.05 else "negative" if score <= -0.05 else "neutral"
    st.info(f"Predicted Mood: **{mood.upper()}** (score: {score:.2f})")
