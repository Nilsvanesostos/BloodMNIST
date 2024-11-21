# utils/training.py

import tensorflow as tf
import matplotlib.pyplot as plt

def train_cnn(model, train_dataset, val_dataset, test_dataset, epochs=50, learning_rate=0.001):
    """
    Train a CNN model on the provided datasets.

    Parameters:
    - model: a compiled tf.keras.Model (e.g., CNN model)
    - train_dataset: tf.data.Dataset for training
    - val_dataset: tf.data.Dataset for validation
    - test_dataset: tf.data.Dataset for testing
    - epochs: int, number of epochs to train
    - learning_rate: float, learning rate for the optimizer

    Returns:
    - model: the trained model
    - history: training history object (tf.keras.callbacks.History)
    """
    # Compile the model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # Train the model
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs
    )

    # Evaluate the model on the test set
    test_loss, test_accuracy = model.evaluate(test_dataset)
    print(f"Test Loss: {test_loss}")
    print(f"Test Accuracy: {test_accuracy}")

    # Plot training & validation accuracy values
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Plot training & validation loss values
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    return model, history

def train_autoencoder(model, train_dataset, val_dataset, epochs=50, learning_rate=0.001):
    """
    Train an autoencoder model on the provided datasets.

    Parameters:
    - autoencoder: a compiled tf.keras.Model (e.g., autoencoder)
    - train_dataset: tf.data.Dataset for training
    - val_dataset: tf.data.Dataset for validation
    - epochs: int, number of epochs to train
    - learning_rate: float, learning rate for the optimizer

    Returns:
    - autoencoder: the trained model
    - history: training history object (tf.keras.callbacks.History)
    """
    # Compile the autoencoder
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='mse',  # Use MSE for reconstruction
        metrics=['mae']  # Monitor MAE
    )

    # Train the autoencoder
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs
    )

    # Plot training & validation metrics
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['mae'], label='Train MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.title('Model MAE')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Absolute Error')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    return model, history
