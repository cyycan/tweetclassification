import pandas as pd
import string
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from pathlib import Path
import pandas as pd
import os

from text_loader.loader import DataLoader

class DataLoader:
    def __init__(self):
        data_path = os.getenv("DATA_PATH")
        if data_path:
            file_path = Path(data_path)
        else:
            base_dir = Path(__file__).resolve().parents[2]
            file_path = base_dir / "data" / "Tweets.csv"
        print("Loading from:", file_path)
        self.data = pd.read_csv(file_path)
        print("Loading from:", file_path)
        self.data = pd.read_csv(file_path)

    
    def load_data(self):
        """Loads data from a CSV file."""
        self.data = pd.read_csv(self.filepath)
        return self.data

    def preprocess_parties(self):
        self.data = self.data[self.data["Party"].notna()]
        self.data = self.data[self.data["Party"].str.strip() != ""]

        self.data.Party = self.data.Party.apply(self.clean_text)
        return self.label_encoder(self.data.Party.values)

    @staticmethod
    def remove_characters(text: str) -> str:
        """Remove non-letters from a given string"""
        text = str(text)
        remove_chars = string.punctuation + string.digits
        translator = str.maketrans('', '', remove_chars)
        return text.translate(translator)

    def clean_text(self, text: str) -> str:
        """Keep only retain words in a given string"""
        text = self.remove_characters(text)
        return text.strip()

    def vectorize_text(self, tweets: list[str]):
        self.vectorizer = TfidfVectorizer(max_features=2500, min_df=1, max_df=0.8, stop_words="english")
        return self.vectorizer.fit_transform(tweets).toarray()

    def label_encoder(self, parties):
        self.encoder = LabelEncoder()
        return self.encoder.fit_transform(parties)

    def preprocess_tweets(self):
        self.data.Tweet = self.data.Tweet.apply(self.clean_text)
        return self.vectorize_text(self.data.Tweet.values)

    def preprocess_parties(self):
        self.data.Party = self.data.Party.apply(self.clean_text)
        return self.label_encoder(self.data.Party.values)
    


