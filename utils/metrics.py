# utils/metrics.py

import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def training_time(train_data, model):
    # Compile the model for training
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

    # Measure training time
    start_time = time.time()
    model.fit(train_data, epochs=1, verbose=0)
    end_time = time.time()

    print("Training Time per Epoch (seconds):", end_time - start_time)

    return end_time -start_time

def memory_occupation(model):
    """
    Calculates and prints the memory occupation of a Keras model.
    """
    # Calculate memory usage in bytes
    total_params = np.sum([np.prod(v.shape) for v in model.trainable_variables])  # Use v.shape instead of v.get_shape()
    memory_in_bytes = total_params * 4  # float32 is 4 bytes

    # Convert bytes to megabytes
    memory_in_mb = memory_in_bytes / (1024 ** 2)

    print(f"Memory Occupation: {memory_in_mb:.2f} MB")
    return memory_in_mb

# Grad-CAM Function
def get_gradcam_heatmap(model, img_tensor, target_layer_name, class_index):
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(target_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_tensor)
        loss = predictions[:, class_index[0]]

    # Compute gradients of the target class score w.r.t. the feature maps
    grads = tape.gradient(loss, conv_outputs)

    # Compute the importance of each feature map channel
    weights = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Combine weights with feature maps
    gradcam = tf.reduce_sum(weights * conv_outputs[0], axis=-1)

    # Apply ReLU to the heatmap
    gradcam = tf.nn.relu(gradcam)

    return gradcam.numpy()

# Visualize Grad-CAM
def plot_gradcam(image, heatmap):
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)  # Normalize to [0,1]
    heatmap = np.uint8(255 * heatmap)  # Scale to [0,255]

    # Apply colormap
    heatmap = tf.keras.preprocessing.image.array_to_img(heatmap[..., np.newaxis])
    heatmap = heatmap.resize((image.shape[1], image.shape[0]))
    heatmap = np.asarray(heatmap)
    heatmap = np.uint8(plt.cm.jet(heatmap)[:, :, :3] * 255)

    # Superimpose heatmap on original image
    superimposed_img = heatmap * 0.4 + image
    return superimposed_img