# Simplified ColBERT for Humor Detection in NLP



## Introduction

This project is an exploration into detecting humor within texts through Natural Language Processing (NLP), focusing on a simplified implementation of the [ColBERT model](https://arxiv.org/abs/2004.12832). It dives deep into linguistic and humor theories, especially the Semantic Script Theory of Humor (SSTH) and incongruity theory, to craft an effective humor detection system. By analyzing and contrasting the capabilities and limitations of several state-of-the-art models such as ColBERT, XGBoost, and XLNet, our goal is to develop advanced systems that grasp the subtle semantic nuances and contextual complexities of humor.

### Key Highlights:

- **Implementation of ColBERT:** This project presents a tailored version of the ColBERT model, modified to meet our specific objectives for humor detection. The neural network architecture has been adjusted based on well-considered decisions to align with our slightly different end goals from the original implementation.

- **Optimized Embedding Space:** To strike a balance between performance and computational efficiency, we've condensed the embedding space to a smaller size. This reduction aims to lessen the training load while still retaining the semantic integrity necessary for high-performance humor detection.

- **Comprehensive Documentation:** For a detailed account of the project's methodologies, modifications, and findings, refer to the included PDF in the repository. This document offers an in-depth exploration of the project's background, implementation details, and the rationale behind key decisions.

- **Academic Context:** This project was undertaken as part of a school project at IE University in 2023, showcasing the practical application of NLP techniques and theories in an academic setting.

The modifications to the ColBERT model and the adjustments in the embedding space were made with the dual objectives of maintaining semantic accuracy and enhancing computational efficiency. These changes reflect our commitment to achieving remarkable results in humor detection while addressing the challenges of large-scale NLP model training.

For more information and a detailed explanation of our approach, please refer to the project report PDF available in the repository. This work represents a significant step towards creating more nuanced and capable NLP systems for humor detection, contributing valuable insights and methodologies to the field.


## Installation

To install the necessary components for this project, follow these steps:
``` bash
git clone https://github.com/yalsaffar/S-COLBERT
```
``` bash
cd S-COLBERT
```

``` bash
pip install -r requirements.txt
```

## Usage

This project can be run using Jupyter notebooks for an interactive approach or via the command line for direct script execution. The data required for model training and evaluation is located in the `data` folder.

### Using Jupyter Notebooks

1. Open `embeddings.ipynb` to extract BERT embeddings from the dataset.
2. Open `training.ipynb` to train the model using the extracted embeddings.

These notebooks provide step-by-step instructions and are ideal for understanding the process and making adjustments interactively.
Paths have to be specified and set before running the code in the notebooks.



## Project Structure

- `embeddings_extract.py`: Script for extracting BERT embeddings from texts. Utilizes the BERT model and tokenizer to process the input data from the `data` folder.
- `model_training.py`: Contains the neural network architecture and training logic. It takes the extracted embeddings and trains a model to detect humor in text.
- `embeddings_utils.py`: Utility functions for text processing, including sentence splitting and embeddings loading.
- `load_data.py`: Handles loading and preprocessing of the dataset, including filtering based on text length and word count.
- `data/`: Folder containing the dataset used for training and evaluating the model. 
- `embeddings.ipynb`: Jupyter notebook for interactive embeddings extraction. Guides through the process of using BERT to convert texts into embeddings.
- `training.ipynb`: Jupyter notebook for model training. Demonstrates how to use the extracted embeddings to train and evaluate the humor detection model.

## Features

- Implementation of a simplified ColBERT model for efficient humor detection.
- Exploration of humor theories and linguistic structures to inform model design.
- Evaluation of model performance against state-of-the-art architectures.


## Configuration

Model parameters and training settings can be adjusted within the scripts and notebooks. Detailed instructions and comments are provided within each file.


## Evaluation & Results

Our simplified ColBERT model, referred to as S-ColBERT, demonstrates impressive results with a precision of 91.69%, ranking second only to the original ColBERT model. It achieves an accuracy of 91.55%, a recall of 91.42%, and an F1 score of 91.56%, showcasing its effectiveness in humor detection with reduced complexity and computational requirements.

## Future Work

Future enhancements include verifying dataset legitimacy, exploring additional preprocessing techniques, investigating other humor theories, expanding the model to longer texts, and adapting the model for different languages. Adjustments to the model architecture could also improve performance.



## License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).

## References

- Original [ColBERT model](https://arxiv.org/abs/2004.12765) paper.
- Dataset: [200K SHORT TEXTS FOR HUMOR DETECTION](https://arxiv.org/abs/2004.12765)
- [BERT](https://huggingface.co/gaunernst/bert-tiny-uncased) Model used
