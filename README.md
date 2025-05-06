# 🧠 Chronomood: Tracking Emotional Patterns by Time

Chronomood is a data science project that analyzes Twitter sentiment over time to reveal how moods shift throughout the day and across the week. Using natural language processing (NLP) and interactive visualizations, it helps answer:  
> *"Are we really moodier on Mondays?"*  
> *"Do people get happier after midnight?"*

## 🚀 Features

- 📊 Sentiment analysis using VADER
- 🕐 Hour-by-hour and day-by-day mood tracking
- 🌈 Heatmaps, line charts, and word clouds
- 🧪 Real-time sentiment tester for your own tweets
- 💬 Streamlit-powered UI for full interactivity

## 📂 Project Structure

Chronomood/
│
├── chronomood.py # Streamlit app
├── requirements.txt # Python dependencies
├── data/
│ ├── combined.csv # Unified cleaned dataset
│ ├── training.1600000... # Raw Sentiment140 data
│ ├── sentimentdataset.csv # Another tweet dataset
│ ├── twitter.csv # Geotagged tweets
│ └── chatgpt1.csv # Miscellaneous tweets


## 🧠 How It Works

1. **Data Cleaning**: Combined 4 datasets (over 1 million tweets)
2. **Sentiment Scoring**: VADER assigns each tweet a sentiment score
3. **Time Features**: Extracts `hour` and `day` from tweet timestamps
4. **Visualizations**: Seaborn, Matplotlib, Plotly to explore trends
5. **Streamlit App**: Interactive UI with filters and tweet classifier

## 📸 Sample Visuals

- Mood heatmaps  
- Sentiment distribution by hour/day  
- Real-time tweet classifier  


## 🧪 Run Locally

```bash
git clone https://github.com/guru-goutham/chronomood.git
cd chronomood
pip install -r requirements.txt
streamlit run chronomood.py

👨‍💻 Built With
Python

Pandas, Seaborn, Matplotlib

NLTK (VADER)

Streamlit

📚 Use Cases
Academic: Temporal mood analysis from social media

Mental Health: Understanding online emotional trends

Product/UX: Identify optimal engagement times

🧑‍🎓 Author
Goutham Guru
Graduate Student | Data Science Enthusiast
🔗 Portfolio | LinkedIn | GitHub

