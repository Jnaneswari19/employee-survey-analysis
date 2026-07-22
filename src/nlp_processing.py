import pandas as pd
import numpy as np
import os
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer

def process_nlp(data_path):
    # Ensure NLTK resources are downloaded
    nltk.download('vader_lexicon', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    nltk.download('stopwords', quiet=True)
    
    df = pd.read_csv(data_path)
    
    stop_words = set(stopwords.words('english'))
    sia = SentimentIntensityAnalyzer()
    
    def clean_text(text):
        if pd.isna(text):
            return ""
        # Lowercase
        text = str(text).lower()
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        # Tokenize
        tokens = word_tokenize(text)
        # Remove stopwords
        tokens = [w for w in tokens if w not in stop_words]
        return " ".join(tokens)
        
    df['cleaned_comment'] = df['open_comment'].apply(clean_text)
    
    # Compute VADER sentiment
    df['sentiment_score'] = df['open_comment'].apply(lambda x: sia.polarity_scores(str(x))['compound'] if pd.notna(x) else 0.0)
    
    # Vectorize text using TF-IDF (max 50 features)
    vectorizer = TfidfVectorizer(max_features=50)
    tfidf_matrix = vectorizer.fit_transform(df['cleaned_comment'])
    
    # Create DataFrame for TF-IDF features
    feature_names = [f"tfidf_{name}" for name in vectorizer.get_feature_names_out()]
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names)
    
    # Merge back into dataset
    df = pd.concat([df, tfidf_df], axis=1)
    
    # Export back to the same path
    df.to_csv(data_path, index=False)
    print(f"NLP processing complete. Saved to {data_path}")
    return df

if __name__ == "__main__":
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_path, 'data', 'processed', 'clean_survey_data.csv')
    process_nlp(data_path)
