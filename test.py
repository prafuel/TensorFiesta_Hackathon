import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import streamlit as st
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string

# Download NLTK resources (if not already downloaded)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('vader_lexicon')

# Load the CSV file into a DataFrame
df = pd.read_csv('clean_reviews.csv')  # Update with your dataset file name

# Initialize VADER sentiment analyzer
sid = SentimentIntensityAnalyzer()

# Text preprocessing
stop_words = set(stopwords.words('english'))
lemmatizer = nltk.WordNetLemmatizer()

def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()
    # Remove punctuation and numbers
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Tokenize text
    tokens = word_tokenize(text)
    # Remove stopwords and lemmatize tokens
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
    return tokens

def classify_sentiment(negative_words_count):
    if negative_words_count >= 3:
        return 'Very Bad'
    elif negative_words_count == 2:
        return 'Bad'
    elif negative_words_count == 1:
        return 'Worse'
    else:
        return 'Good'

def plot_sentiment_distribution(selected_state):
    state_df = df[df['states'] == selected_state]
    sentiment_classification = []
    for review_text in state_df['review_content']:
        tokens = preprocess_text(review_text)
        negative_words_count = sum(sid.polarity_scores(token)['compound'] < -0.2 for token in tokens)
        sentiment = classify_sentiment(negative_words_count)
        sentiment_classification.append(sentiment)
    
    # Plot sentiment distribution for the selected state
    plt.figure(figsize=(8, 6))
    sns.countplot(x=sentiment_classification, order=['Good', 'Worse', 'Bad', 'Very Bad'], palette='viridis')
    plt.title(f'Sentiment Distribution for {selected_state}')
    plt.xlabel('Sentiment')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    st.pyplot()

# Streamlit app
def main():
    st.title("Sentiment Analysis Visualization by State")
    selected_state = st.radio("Select a State", df['states'].unique())
    plot_sentiment_distribution(selected_state)

if __name__ == "__main__":
    main()