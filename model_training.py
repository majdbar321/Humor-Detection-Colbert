import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import numpy as np

class ModelTraining:
    def __init__(self):
        """
        Initializes the ModelTraining class.
        """
        return

    def model_build(self):
        """
        Builds the model architecture.

        Returns:
        model (tensorflow.keras.Model): The built model.
        """
        # Define input layers
        whole_text_input = Input(shape=(5, 512), dtype='float32', name='whole_text_input')
        sentence_inputs = [Input(shape=(5, 512), dtype='float32', name=f'sentence_input_{i}') for i in range(2)]  # Change to 2 inputs

        # Sentence-specific hidden layers
        sentence_hidden_layers = []
        for sentence_input in sentence_inputs:
            hidden = Dense(20, activation='relu')(sentence_input)
            hidden = Dense(40, activation='LeakyReLU')(sentence_input)

            sentence_hidden_layers.append(hidden)

        # Concatenate sentence-specific hidden layers
        concatenated_sentences = Concatenate()(sentence_hidden_layers)

        # Whole text hidden layers
        whole_text_hidden = Dense(60, activation='relu')(whole_text_input)

        # Combine sentence-specific and whole text hidden layers
        combined = Concatenate()([concatenated_sentences, whole_text_hidden])

        # Final hidden layers and output layer
        hidden_1 = Dense(200, activation='LeakyReLU')(combined)
        hidden_2 = Dense(180, activation='relu')(hidden_1)
        hidden_3 = Dense(160, activation='LeakyReLU')(hidden_2)

        hidden_4 = Dense(140, activation='relu')(hidden_3)

        hidden_5 = Dense(120, activation='LeakyReLU')(hidden_4)
        hidden_6 = Dense(100, activation='relu')(hidden_5)
        hidden_7 = Dense(80, activation='LeakyReLU')(hidden_6)
        hidden_8 = Dense(60, activation='relu')(hidden_7)
        hidden_9 = Dense(40, activation='LeakyReLU')(hidden_8)
        hidden_10 = Dense(20, activation='relu')(hidden_9)

        output = Dense(1, activation='sigmoid')(hidden_10)
        # Build the model
        model = Model(inputs=[whole_text_input] + sentence_inputs, outputs=output)
        return model

    def model_compile(self, model, strategy, optimizer=Adam, loss="binary_crossentropy", metrics=['accuracy'], learning_rate=1e-4):
        """
        Compiles the model.

        Args:
        model (tensorflow.keras.Model): The model to be compiled.
        strategy (tensorflow.distribute.Strategy): The strategy for distributed training.
        optimizer (tensorflow.keras.optimizers.Optimizer): The optimizer to be used.
        loss (str or tensorflow.keras.losses.Loss): The loss function to be used.
        metrics (list): List of metrics to be evaluated during training.
        learning_rate (float): The learning rate for the optimizer.

        Returns:
        model (tensorflow.keras.Model): The compiled model.
        """
        with strategy.scope():
            # Compile the model
            model.compile(optimizer=optimizer(learning_rate=learning_rate), loss=loss, metrics=metrics)

        return model

    def model_fit(self, model, X_train_embeddings, X_train_sentence_embeddings, y_train, X_val_embeddings, X_val_sentence_embeddings, y_val, epochs=30, batch_size=16):
        """
        Trains the model.

        Args:
        model (tensorflow.keras.Model): The compiled model.
        X_train_embeddings (numpy.ndarray): The input embeddings for the whole text in the training set.
        X_train_sentence_embeddings (list): The input embeddings for the sentences in the training set.
        y_train (numpy.ndarray): The target labels for the training set.
        X_val_embeddings (numpy.ndarray): The input embeddings for the whole text in the validation set.
        X_val_sentence_embeddings (list): The input embeddings for the sentences in the validation set.
        y_val (numpy.ndarray): The target labels for the validation set.
        epochs (int): The number of epochs to train the model.
        batch_size (int): The batch size for training.

        Returns:
        history (tensorflow.keras.callbacks.History): The training history.
        model (tensorflow.keras.Model): The trained model.
        """
        history = model.fit(
            [X_train_embeddings] + X_train_sentence_embeddings, y_train,
            validation_data=([X_val_embeddings] + X_val_sentence_embeddings, y_val),
            epochs=epochs,
            batch_size=batch_size)
        return history, model
    
    def model_evaluate(self, model, X_val_embeddings, X_val_sentence_embeddings, y_val):
        """
        Evaluates the model on the validation set.

        Args:
        model (tensorflow.keras.Model): The trained model.
        X_val_embeddings (numpy.ndarray): The input embeddings for the whole text in the validation set.
        X_val_sentence_embeddings (list): The input embeddings for the sentences in the validation set.
        y_val (numpy.ndarray): The target labels for the validation set.
        """
        for i in range(5):
            # Make predictions on the validation set
            y_pred = model.predict([X_val_embeddings] + X_val_sentence_embeddings)

            # Convert predicted probabilities to binary predictions
            y_pred = np.where(y_pred > 0.5, 1, 0)
            y_pred = [inner_list[i] for inner_list in y_pred]
            # Calculate precision, recall, and F1-score
            from sklearn.metrics import precision_recall_fscore_support
            from sklearn.metrics import accuracy_score

            precision, recall, f1, _ = precision_recall_fscore_support(y_val, y_pred, labels=[0, 1], average='binary')

            # Calculate accuracy
            accuracy = accuracy_score(y_val, y_pred)

            print("Validation Accuracy: %.2f%%" % (accuracy * 100))
            print("Validation Precision: %.2f%%" % (precision * 100))
            print("Validation Recall: %.2f%%" % (recall * 100))
            print("Validation F1-score: %.2f%%" % (f1 * 100))
            print("-----------------")
