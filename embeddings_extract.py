import numpy as np
from textblob import TextBlob
from transformers import BertTokenizer, TFBertModel
from sklearn.model_selection import train_test_split
from sklearn import metrics
from embeddings_utils import  split_sentences, pad_or_truncate_sentences
from load_data import LoadData

def load_bert(path, from_pt=True):
    """
    Load BERT tokenizer and model.

    Args:
        path (str): Path to the BERT model.

    Returns:
        tokenizer (BertTokenizer): BERT tokenizer.
        bert_model (TFBertModel): BERT model.
    """
    tokenizer = BertTokenizer.from_pretrained(path, from_pt=from_pt)
    bert_model = TFBertModel.from_pretrained(path, from_pt=from_pt)
    return tokenizer, bert_model
def get_bert_embeddings(texts, tokenizer, bert_model, strategy, batch_size=32):
    """
    Get BERT embeddings for a list of texts.

    Args:
        texts (list): The list of texts.
        tokenizer: The BERT tokenizer.
        bert_model: The BERT model.
        strategy: The strategy for distributing computation across GPUs.
        batch_size (int, optional): The batch size for processing texts. Defaults to 32.

    Returns:
        numpy.ndarray: The BERT embeddings for the texts.
    """
    embeddings = []

    # Process texts in smaller batches
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]

        # Ensure the input is a list of strings
        batch_texts = [str(text) for text in batch_texts]

        tokenized = tokenizer(batch_texts, padding=True, truncation=True, return_tensors='tf', max_length=5)

        # Use the MirroredStrategy scope to distribute computation across GPUs
        with strategy.scope():
            batch_embeddings = bert_model(tokenized)['last_hidden_state']
        
        embeddings.extend(batch_embeddings.numpy())

    return np.array(embeddings)

def extract_words_embeddings(X_train, X_val, strategy, tokenizer, bert_model, path_to_save=""):
    """
    Extract BERT embeddings for words in the training and validation data.

    Args:
        X_train (pd.Series): Training data.
        X_val (pd.Series): Validation data.
        strategy (str): Embedding strategy.
        tokenizer (BertTokenizer): BERT tokenizer.
        bert_model (TFBertModel): BERT model.
        path_to_save (str, optional): Path to save the embeddings. Defaults to "".
    """
    print('Extracting BERT embeddings for training data...')

    # Extract BERT embeddings for training data
    X_train_embeddings = get_bert_embeddings(X_train, tokenizer, bert_model, strategy)
    X_val_embeddings = get_bert_embeddings(X_val, tokenizer, bert_model, strategy)

    # Save the embeddings to disk
    np.save(path_to_save + '/X_train_embeddings.npy', X_train_embeddings)
    print('BERT embeddings for training data saved to disk.')

    np.save(path_to_save + '/X_val_embeddings.npy', X_val_embeddings)
    print('Extracting BERT embeddings for validation data...')


def extract_sentences_embeddings(X_train, X_val, tokenizer, bert_model, path_to_save=""):
    """
    Extract BERT embeddings for sentences in the training and validation data.

    Args:
        X_train (pd.Series): Training data.
        X_val (pd.Series): Validation data.
        tokenizer (BertTokenizer): BERT tokenizer.
        bert_model (TFBertModel): BERT model.
        path_to_save (str, optional): Path to save the embeddings. Defaults to "".
    """
    Dataload = LoadData()
    X_train_sentences, X_val_sentences = Dataload.sentence_data_prep(X_train, X_val)

    # Obtain BERT embeddings for individual sentences in training and validation sets
    X_train_sentence_embeddings = [get_bert_embeddings(X_train_sentences.apply(lambda x: x[i]), tokenizer, bert_model) for i in range(3)]
    X_val_sentence_embeddings = [get_bert_embeddings(X_val_sentences.apply(lambda x: x[i]), tokenizer, bert_model) for i in range(3)]

    # Save the embeddings to disk
    for i, emb in enumerate(X_train_sentence_embeddings):
        np.save(path_to_save + f'/X_train_sentence_embedding_{i}.npy', emb)
        print(f'X_train_sentence_embedding_{i}.npy saved to disk.')

    for i, emb in enumerate(X_val_sentence_embeddings):
        np.save(path_to_save + f'/X_val_sentence_embedding_{i}.npy', emb)
        print(f'X_val_sentence_embedding_{i}.npy saved to disk.')
