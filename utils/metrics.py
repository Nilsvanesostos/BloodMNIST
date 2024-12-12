# utils/metrics.py

import time
import numpy as np
import tensorflow as tf

def training_time(train_data, model):
    # Compile the model for training
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

    # Measure training time
    start_time = time.time()
    model.fit(train_data, epochs=1, verbose=0)
    end_time = time.time()

    print("Training Time per Epoch (seconds):", end_time - start_time)

    return 

def memory_occupation(model):
    # Calculate memory usage in bytes
    total_params = np.sum([np.prod(v.get_shape().as_list()) for v in model.trainable_variables])
    memory_in_bytes = total_params * 4  # float32 is 4 bytes

    # Convert to megabytes
    memory_in_megabytes = memory_in_bytes / (1024 ** 2)
    print("Model Memory Usage (MB):", memory_in_megabytes)

    return