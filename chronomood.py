import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

st.set_page_config(page_title="Chronomood", layout="wide")

# --- Load dataset ---
@st.cache_data
def load_data():
    df = pd.read_csv("data/sample_combined.csv", parse_dates=['datetime'])
    return df

df = load_data()

# --- Preprocessing ---
sentiment_map = {'positive': 1, 'neutral': 0.5, 'negative': 0}
df['sentiment_score'] = df['sentiment'].map(sentiment_map)
df['hour'] = df['datetime'].dt.hour
df['day'] = df['datetime'].dt.day_name()

# --- Page title ---
st.title("🧠 Chronomood: Tracking Emotional Patterns by Time")

# --- Sidebar filters ---
st.sidebar.header("Filters")
day_options = sorted(df['day'].dropna().astype(str).unique())
day = st.sidebar.selectbox("Day of Week", day_options)
hour = st.sidebar.slider("Hour of Day", 0, 23)

# --- Filtered subset ---
filtered = df[(df['day'] == day) & (df['hour'] == hour)]
day_filtered = df[df['day'] == day]

# --- Sentiment Distribution ---
st.subheader(f"📊 Sentiment Distribution at {hour}:00 on {day}")
if not filtered.empty:
    fig, ax = plt.subplots()
    sns.countplot(data=filtered, x='sentiment', ax=ax, palette='coolwarm')
    ax.set_title("Sentiment Count")
    st.pyplot(fig)
else:
    st.warning("No tweets found for this time slot.")

# --- Word Cloud ---
st.subheader("☁️ Word Cloud for Selected Time")
sentiment = st.radio("Choose sentiment:", ['positive', 'negative'])
text = " ".join(filtered[filtered['sentiment'] == sentiment]['text'].astype(str))

if text.strip():
    wc = WordCloud(width=800, height=300, background_color="white").generate(text)
    st.image(wc.to_array(), use_container_width=True)
else:
    st.warning("No tweets found for this selection.")

# --- Mood Prediction ---
st.subheader("🧠 Test the Mood of Your Tweet")
user_input = st.text_area("Write a tweet here:")
if user_input:
    analyzer = SentimentIntensityAnalyzer()
    score = analyzer.polarity_scores(user_input)['compound']
    mood = "positive" if score >= 0.05 else "negative" if score <= -0.05 else "neutral"
    st.info(f"Predicted Mood: **{mood.upper()}** (score: {score:.2f})")

# --- Average Sentiment by Hour (Filtered by Day) ---
st.subheader(f"📈 Average Sentiment by Hour on {day}")
hourly_sentiment = day_filtered.groupby('hour')['sentiment_score'].mean().reset_index()

if not hourly_sentiment.empty:
    fig1, ax1 = plt.subplots()
    ax1.plot(hourly_sentiment['hour'], hourly_sentiment['sentiment_score'], marker='o')
    ax1.set_title(f"Average Sentiment on {day}")
    ax1.set_xlabel("Hour of Day")
    ax1.set_ylabel("Avg Sentiment (1 = 😊, 0 = 😠)")
    ax1.grid(True)
    st.pyplot(fig1)
else:
    st.warning("No sentiment data for this day.")

# --- Tweet Volume by Hour and Sentiment (Filtered by Day) ---
st.subheader(f"📊 Tweet Volume by Hour and Sentiment on {day}")
volume_df = day_filtered.groupby(['hour', 'sentiment']).size().unstack(fill_value=0)

if not volume_df.empty:
    fig2, ax2 = plt.subplots()
    volume_df.plot(kind='bar', stacked=True, ax=ax2, colormap='coolwarm')
    ax2.set_title(f"Tweet Volume on {day}")
    ax2.set_xlabel("Hour of Day")
    ax2.set_ylabel("Number of Tweets")
    st.pyplot(fig2)
else:
    st.warning("No tweet volume data to display for this day.")
