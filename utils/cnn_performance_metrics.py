# utils/cnn_performance_metrics.py

import subprocess
import time
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import seaborn as sns

# Function to get GPU memory usage
def get_gpu_memory():
    result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,nounits,noheader'], stdout=subprocess.PIPE)
    memory_info = result.stdout.decode('utf-8').strip().split(',')
    memory_used = int(memory_info[0])  # Memory used in MB
    memory_total = int(memory_info[1])  # Total memory in MB
    return memory_used, memory_total

# Function to estimate model memory usage
def estimate_model_memory(model):
    total_params = model.count_params()
    memory_estimate = total_params * 4 / (1024 ** 2)  # 4 bytes per parameter (float32)
    return memory_estimate

# Evaluate model function
def evaluate_model(model, train_tf_dataset, val_tf_dataset, model_name, num_epochs):
    # Number of Parameters
    total_params = model.count_params()

    # Training Time
    start_time = time.time()
    history = model.fit(train_tf_dataset, validation_data=val_tf_dataset, epochs=num_epochs)
    end_time = time.time()
    training_time = end_time - start_time

    # GPU Memory Usage
    gpu_memory_used_before, gpu_memory_total = get_gpu_memory()

    # Model Memory Usage
    model_memory_usage = estimate_model_memory(model)

    # Log the information
    model_log = {
        "Model": model_name,
        "Total Parameters": total_params,
        "Training Time": training_time,
        "GPU Memory Before Training (MB)": gpu_memory_used_before,
        "GPU Memory Total (MB)": gpu_memory_total,
        "Estimated Model Memory Usage (MB)": model_memory_usage,
    }

    # Plot training & validation accuracy values
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.legend()

    # Plot training & validation loss values
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()


    plt.show()

    return model_log

def evaluate_test_data(model, test_tf_dataset, model_log):
    # Evaluate the model on the test set
    test_loss, test_accuracy = model.evaluate(test_tf_dataset)
    print(f"Test Loss: {test_loss}")
    print(f"Test Accuracy: {test_accuracy}")

    # Get the true labels and predicted labels
    y_true = []
    y_pred = []

    for images, labels in test_tf_dataset:
        # Get the true labels
        y_true.extend(labels.numpy())

        # Get predictions
        predictions = model.predict(images)
        predicted_labels = np.argmax(predictions, axis=1)  # For multi-class classification

        # Append predicted labels
        y_pred.extend(predicted_labels)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)

    # Plot the confusion matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', xticklabels=np.arange(len(np.unique(y_true))), yticklabels=np.arange(len(np.unique(y_true))))
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.show()

    # Classification Report
    print("Classification Report:")
    print(classification_report(y_true, y_pred))

    # F1 Score
    f1 = f1_score(y_true, y_pred, average='weighted')  # 'weighted' accounts for class imbalance
    print(f"F1 Score (Weighted): {f1:.4f}")

    model_log['Test Loss'] = test_loss
    model_log['Test Accuracy'] = test_accuracy
    model_log['Test Weighted F1 Score'] = f1