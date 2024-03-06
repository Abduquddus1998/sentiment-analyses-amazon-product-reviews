import pandas as pd
import spacy

nlp = spacy.load('en_core_web_lg')

# loading amazon product reviews dataset
reviews_df = pd.read_csv("amazon_product_reviews.csv", low_memory=False)

# removing missing values
reviews_df = reviews_df.dropna(subset=['reviews.text'])


# creating tokens for each text
def preprocess(text):
    doc = nlp(text)
    return " ".join([token.lemma_ for token in doc if not (token.is_stop or token.is_punct or token.is_digit)])


# selecting 'review.text' column
reviews_data = reviews_df['reviews.text'].apply(preprocess)


def analyze_sentiment(review):
    doc = nlp(review)

    polarity = doc._.blob.polarity

    if polarity >= 0.5:
        return 'Positive'
    elif polarity < -0.5:
        return 'Negative'
    else:
        return 'Neutral'


# selecting sample reviews to test sentiment method
sample_reviews = reviews_data[:20]

for review in sample_reviews:
    sentiment = analyze_sentiment(review)
    print(f"Review: {review} --> Sentiment: {sentiment}")
