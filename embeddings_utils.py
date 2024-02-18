import numpy as np
from textblob import TextBlob
import os

def word_count(text):
    """
    Count the number of words in a text.

    Args:
        text (str): The input text.

    Returns:
        int: The number of words in the text.
    """
    return len(text.split())



def split_sentences(text):
    """
    Split a text into sentences.

    Args:
        text (str): The input text.

    Returns:
        list: The list of sentences in the text.
    """
    return [str(sentence) for sentence in TextBlob(text).sentences]


def pad_or_truncate_sentences(sentences_list, num_sentences=3):
    """
    Pad or truncate a list of sentences to a specified number of sentences.

    Args:
        sentences_list (list): The list of sentences.
        num_sentences (int, optional): The desired number of sentences. Defaults to 3.

    Returns:
        list: The padded or truncated list of sentences.
    """
    if len(sentences_list) > num_sentences:
        return sentences_list[:num_sentences]
    else:
        return sentences_list + [''] * (num_sentences - len(sentences_list))
    

# Load the embeddings from disk
import os
import numpy as np

def load_embeddings(path_to_embeddings, prefix):
    """
    Load embeddings from multiple files with a given prefix.

    Args:
        path_to_embeddings (str): The path to the directory containing the embeddings files.
        prefix (str): The prefix used in the filenames of the embeddings files.

    Returns:
        list: A list of loaded embeddings.

    """
    embeddings = []
    i = 0
    while True:
        file_path = f'{path_to_embeddings}/{prefix}_embedding_{i}.npy'
        if os.path.exists(file_path):
            emb = np.load(file_path, allow_pickle=True)
            embeddings.append(emb)
            i += 1
        else:
            break
    return embeddings