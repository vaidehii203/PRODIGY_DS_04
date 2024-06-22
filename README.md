# PRODIGY_DS_04
Analysis and visualization of sentiment patterns in social media data to understand public opinion and attitudes towards specific topics or brands.

Dataset link:https://www.kaggle.com/datasets/jp797498e/twitter-entity-sentiment-analysis


```markdown
# Sentiment Analysis Comparison: TextBlob vs. VADER

This project also compares sentiment analysis using TextBlob and VADER (vader_lexicon) on a dataset. TextBlob is a simple NLP library, while VADER is specifically designed for social media sentiment analysis. The project analyzes sentiment and categorizes it into positive, negative, or neutral sentiments based on polarity scores.

## Table of Contents

- [Introduction](#introduction)
- [Setup](#setup)
- [Sentiment Analysis with VADER](#sentiment-analysis-with-vader)
- [Sentiment Analysis with TextBlob](#sentiment-analysis-with-textblob)
- [Results](#results)
- [Contact](#contact)

## Introduction

This project demonstrates sentiment analysis using TextBlob and VADER on textual data. Both tools analyze sentiment polarity and categorize it into predefined classes (positive, negative, neutral). Differences in results may arise due to their distinct lexicons and algorithms.

## Setup

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/vaidehii203/PRODIGY_DS_04.git
   
   ```

2. **Install Dependencies:**

   Ensure you have Python 3.x and pip installed. Install the required libraries:

   ```bash
   pip install pandas matplotlib nltk textblob
   ```

3. **Download NLTK Data:**

   TextBlob requires NLTK corpora. Run Python interpreter:

   ```python
   >>> import nltk
   >>> nltk.download('vader_lexicon')
   ```

## Sentiment Analysis with VADER

```python
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv("D:\\intership tasks\\twitter_training.csv", header=None)  # Assuming no header is present
df.columns = ['tweet_id', 'entity', 'sentiment', 'tweet_content']

# Initialize VADER sentiment analyzer
sid = SentimentIntensityAnalyzer()

# Function to classify sentiment based on compound score
def classify_sentiment(compound_score):
    if compound_score >= 0.05:
        return 'Positive'
    elif compound_score <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

# Apply sentiment analysis and store results in a new column 'sentiment'
df['sentiment_score'] = df['tweet_content'].apply(lambda x: sid.polarity_scores(str(x))['compound'])
df['sentiment'] = df['sentiment_score'].apply(classify_sentiment)

# Display the updated DataFrame (optional)
print(df.head())

# Plot sentiment distribution
plt.figure(figsize=(10, 6))
plt.hist(df['sentiment_score'], bins=30, edgecolor='black', alpha=0.7)
plt.title('Sentiment Distribution (VADER)')
plt.xlabel('Compound Sentiment Score')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# Plot sentiment counts
sentiment_counts = df['sentiment'].value_counts()
plt.figure(figsize=(8, 6))
plt.bar(sentiment_counts.index, sentiment_counts.values, color=['green', 'red', 'blue'])
plt.title('Number of Sentiments')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.grid(True)
plt.show()
```

## Sentiment Analysis with TextBlob

```python
import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv("D:\\intership tasks\\twitter_training.csv", header=None)  # Assuming no header is present
df.columns = ['tweet_id', 'entity', 'sentiment', 'tweet_content']

# Function to classify sentiment based on TextBlob sentiment polarity
def classify_sentiment(polarity):
    if polarity > 0.05:
        return 'Positive'
    elif polarity < -0.05:
        return 'Negative'
    else:
        return 'Neutral'

# Apply sentiment analysis and store results in a new column 'sentiment_score'
df['sentiment_score'] = df['tweet_content'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
df['sentiment'] = df['sentiment_score'].apply(classify_sentiment)

# Display the updated DataFrame with sentiment interpretation
print(df[['tweet_content', 'sentiment_score', 'sentiment']])

# Plot sentiment distribution using a histogram of sentiment scores
plt.figure(figsize=(10, 6))
plt.hist(df['sentiment_score'], bins=30, edgecolor='black', alpha=0.7)
plt.title('Sentiment Distribution (TextBlob)')
plt.xlabel('Sentiment Polarity Score')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()
```

## Results

- The above scripts output processed dataframes with sentiment scores categorized into positive, negative, or neutral.
- Visualizations depict the distribution of sentiment scores using histograms and bar charts for both VADER and TextBlob analyses.

## Contact

For questions, feedback, or collaboration opportunities, please feel free to contact me via [LinkedIn]( https://www.linkedin.com/in/vaidehi-kale-b635b7264/).
