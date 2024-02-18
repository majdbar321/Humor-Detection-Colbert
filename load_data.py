import pandas as pd
from sklearn.model_selection import train_test_split
from embeddings_utils import word_count
from embeddings_utils import split_sentences, pad_or_truncate_sentences

class LoadData:
    def __init__(self):
        """
        Initialize LoadData class.
        """
        return

    def load_data(self, path, min_length=30, max_length=100, min_words=5, max_words=25):
        """
        Load data from a CSV file and apply filters based on character length and word length.

        Args:
            path (str): Path to the CSV file.
            min_length (int): Minimum character length of jokes (default: 30).
            max_length (int): Maximum character length of jokes (default: 100).
            min_words (int): Minimum word count of jokes (default: 5).
            max_words (int): Maximum word count of jokes (default: 25).

        Returns:
            pandas.DataFrame: Filtered dataframe containing jokes data.
        """
        jokes_df = pd.read_csv(path)

        # Drop duplicate rows
        jokes_df.drop_duplicates(subset=['text'], inplace=True)

        # Filter rows based on character length and word length
        jokes_df = jokes_df[
            (jokes_df['text'].str.len() >= min_length) &
            (jokes_df['text'].str.len() <= max_length) &
            (jokes_df['text'].apply(word_count) >= min_words) &
            (jokes_df['text'].apply(word_count) <= max_words)
        ]
        self.jokes_df = jokes_df
        return jokes_df

    def split_data(self, split_ratio=0.2, random_state=42):
        """
        Split the data into training and validation sets.

        Args:
            split_ratio (float): Ratio of validation data (default: 0.2).
            random_state (int): Random seed for reproducibility (default: 42).

        Returns:
            tuple: X_train, X_val, y_train, y_val
                - X_train (pandas.Series): Training data.
                - X_val (pandas.Series): Validation data.
                - y_train (pandas.Series): Training labels.
                - y_val (pandas.Series): Validation labels.
        """
        # Split the data into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(
            self.jokes_df['text'], self.jokes_df['humor'], test_size=split_ratio, random_state=random_state
        )
        return X_train, X_val, y_train, y_val

    def sentence_data_prep(self, X_train, X_val):
        """
        Prepare sentence-wise data for training and validation sets.

        Args:
            X_train (pandas.Series): Training data.
            X_val (pandas.Series): Validation data.

        Returns:
            tuple: X_train_sentences, X_val_sentences
                - X_train_sentences (pandas.Series): Sentence-wise data for training data.
                - X_val_sentences (pandas.Series): Sentence-wise data for validation data.
        """
        # Get sentence-wise embeddings for training and validation sets
        X_train_sentences = X_train.apply(split_sentences)
        X_val_sentences = X_val.apply(split_sentences)

        X_train_sentences = X_train_sentences.apply(pad_or_truncate_sentences)
        X_val_sentences = X_val_sentences.apply(pad_or_truncate_sentences)

        return X_train_sentences, X_val_sentences
