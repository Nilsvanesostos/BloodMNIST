# utils/cnn_experiments.py

from tensorflow.keras.layers import Input
from tensorflow.keras import layers, models
from tensorflow.keras.initializers import glorot_uniform

def create_cnn(model_name):

    # Define input shape
    input_shape = (64, 64, 3)
    num_classes = 8

    model_dict= {
    # without dense layers
    'CNN_3_LAYERS_WO_MAXPOOL_IN_FINAL_LAYER_NO_DENSE_ADAM' : cnn_model_1,
    'CNN_3_LAYERS_W_MAXPOOL_IN_FINAL_LAYER_NO_DENSE_ADAM' : cnn_model_2,
    'CNN_4_LAYERS_WO_MAXPOOL_IN_FINAL_LAYER_NO_DENSE_ADAM' : cnn_model_3,
    'CNN_4_LAYERS_W_MAXPOOL_IN_FINAL_LAYER_NO_DENSE_ADAM' : cnn_model_4,
    'CNN_4_LAYERS_W_MAXPOOL_IN_FINAL_LAYER_W_BATCH_NORMALIZATION_NO_DENSE_ADAM' : cnn_model_5,

    # with dense layers
    'CNN_3_LAYERS_WO_MAXPOOL_IN_FINAL_LAYER_DENSE_128_ADAM' : cnn_model_6,
    'CNN_3_LAYERS_W_MAXPOOL_IN_FINAL_LAYER_DENSE_128_ADAM' : cnn_model_7,
    'CNN_4_LAYERS_WO_MAXPOOL_IN_FINAL_LAYER_DENSE_128_ADAM' : cnn_model_8,
    'CNN_3_LAYERS_WO_MAXPOOL_IN_FINAL_LAYER_DENSE_256_ADAM' : cnn_model_9,
    'CNN_3_LAYERS_W_MAXPOOL_IN_FINAL_LAYER_DENSE_256_ADAM' : cnn_model_10,
    'CNN_4_LAYERS_WO_MAXPOOL_IN_FINAL_LAYER_DENSE_256_ADAM' : cnn_model_11, 
    'CNN_4_LAYERS_W_MAXPOOL_IN_FINAL_LAYER_DENSE_128_ADAM' : cnn_model_12,
    'CNN_4_LAYERS_W_MAXPOOL_IN_FINAL_LAYER_DENSE_256_ADAM' : cnn_model_13,
    'CNN_4_LAYERS_WO_MAXPOOL_IN_FINAL_LAYER_DENSE_512_ADAM' : cnn_model_14,
    'CNN_4_LAYERS_W_MAXPOOL_IN_FINAL_LAYER_DENSE_512_ADAM' : cnn_model_15,
    'CNN_4_LAYERS_W_MAXPOOL_IN_FINAL_LAYER_W_BATCH_NORMALIZATION_DENSE_512_ADAM' : cnn_model_16

    }

    model_func = model_dict.get(model_name)
    if model_func:
        model = model_func(input_shape, num_classes)
    else:
        print(f"No model found for the name: {model_name}")

    return model


def cnn_model_1(input_shape, num_classes):

    # Define the input layer explicitly
    inputs = Input(shape=input_shape)

    # First convolutional block
    x = layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D(pool_size=(2, 2), strides=(1,1), padding='same')(x)

    # Second convolutional block
    x = layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), strides=(1,1), padding='same')(x)

    # Third convolutional block
    x = layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(x)

    # Flatten and fully connected layers
    x = layers.Flatten()(x)
    # x = layers.Dense(units=128, activation='relu')(x)
    outputs = layers.Dense(units=num_classes, activation='softmax')(x)  # For multi-class classification

    # Create and return the model
    model = models.Model(inputs=inputs, outputs=outputs, name='CNN')
    print('done')
    return model



def cnn_model_2(input_shape, num_classes):

    # Define the input layer explicitly
    inputs = Input(shape=input_shape)

    # First convolutional block
    x = layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D(pool_size=(2, 2), strides=(1,1), padding='same')(x)

    # Second convolutional block
    x = layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), strides=(1,1), padding='same')(x)

    # Third convolutional block
    x = layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), strides=(1,1), padding='same')(x)

    # Flatten and fully connected layers
    x = layers.Flatten()(x)
    # x = layers.Dense(units=128, activation='relu')(x)
    outputs = layers.Dense(units=num_classes, activation='softmax')(x)  # For multi-class classification

    # Create and return the model
    model = models.Model(inputs=inputs, outputs=outputs, name='CNN')
    return model



def cnn_model_3(input_shape, num_classes):

    # Define the input layer explicitly
    inputs = Input(shape=input_shape)

    # First convolutional block
    x = layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D(pool_size=(2, 2), strides=(1,1), padding='same')(x)

    # Second convolutional block
    x = layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), strides=(1,1), padding='same')(x)

    # Third convolutional block
    x = layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), strides=(1,1), padding='same')(x)

    # Fourth convolutional block
    x = layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(x)

    # Flatten and fully connected layers
    x = layers.Flatten()(x)
    # x = layers.Dense(units=128, activation='relu')(x)
    outputs = layers.Dense(units=num_classes, activation='softmax')(x)  # For multi-class classification

    # Create and return the model
    model = models.Model(inputs=inputs, outputs=outputs, name='CNN')
    return model



def cnn_model_4(input_shape, num_classes):

    # Define the input layer explicitly
    inputs = Input(shape=input_shape)

    # First convolutional block
    x = layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D(pool_size=(2, 2), strides=(1,1), padding='same')(x)

    # Second convolutional block
    x = layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), strides=(1,1), padding='same')(x)

    # Third convolutional block
    x = layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), strides=(1,1), padding='same')(x)

    # Fourth convolutional block
    x = layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), strides=(1,1), padding='same')(x)

    # Flatten and fully connected layers
    x = layers.Flatten()(x)
    # x = layers.Dense(units=256, activation='relu')(x)
    outputs = layers.Dense(units=num_classes, activation='softmax')(x)  # For multi-class classification

    # Create and return the model
    model = models.Model(inputs=inputs, outputs=outputs, name='CNN')
    return model



def cnn_model_5(input_shape, num_classes):

    # Define the input layer explicitly
    inputs = Input(shape=input_shape)

    # First convolutional block
    x = layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same')(inputs)
    x = layers.BatchNormalization()(x)  # Batch normalization after Conv2D
    x = layers.Activation('relu')(x)  # Activation after batch normalization
    x = layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same')(x)

    # Second convolutional block
    x = layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same')(x)

    # Third convolutional block
    x = layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same')(x)

    # Fourth convolutional block
    x = layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same')(x)

    # Flatten and fully connected layers
    x = layers.Flatten()(x)
    # x = layers.Dense(units=512)(x)  # No activation yet
    # x = layers.BatchNormalization()(x)
    # x = layers.Activation('relu')(x)
    outputs = layers.Dense(units=num_classes, activation='softmax')(x)  # For multi-class classification

    # Create and return the model
    model = models.Model(inputs=inputs, outputs=outputs, name='CNN_with_BatchNorm')
    return model


def cnn_model_6(input_shape, num_classes):

    # Define the input layer explicitly
    inputs = Input(shape=input_shape)

    # First convolutional block
    x = layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D(pool_size=(2, 2), strides=(1,1), padding='same')(x)

    # Second convolutional block
    x = layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), strides=(1,1), padding='same')(x)

    # Third convolutional block
    x = layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(x)

    # Flatten and fully connected layers
    x = layers.Flatten()(x)
    x = layers.Dense(units=128, activation='relu')(x)
    outputs = layers.Dense(units=num_classes, activation='softmax')(x)  # For multi-class classification

    # Create and return the model
    model = models.Model(inputs=inputs, outputs=outputs, name='CNN')
    return model



def cnn_model_7(input_shape, num_classes):

    # Define the input layer explicitly
    inputs = Input(shape=input_shape)

    # First convolutional block
    x = layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D(pool_size=(2, 2), strides=(1,1), padding='same')(x)

    # Second convolutional block
    x = layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), strides=(1,1), padding='same')(x)

    # Third convolutional block
    x = layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), strides=(1,1), padding='same')(x)

    # Flatten and fully connected layers
    x = layers.Flatten()(x)
    x = layers.Dense(units=128, activation='relu')(x)
    outputs = layers.Dense(units=num_classes, activation='softmax')(x)  # For multi-class classification

    # Create and return the model
    model = models.Model(inputs=inputs, outputs=outputs, name='CNN')
    return model



def cnn_model_8(input_shape, num_classes):

    # Define the input layer explicitly
    inputs = Input(shape=input_shape)

    # First convolutional block
    x = layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D(pool_size=(2, 2), strides=(1,1), padding='same')(x)

    # Second convolutional block
    x = layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), strides=(1,1), padding='same')(x)

    # Third convolutional block
    x = layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), strides=(1,1), padding='same')(x)

    # Fourth convolutional block
    x = layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(x)

    # Flatten and fully connected layers
    x = layers.Flatten()(x)
    x = layers.Dense(units=128, activation='relu')(x)
    outputs = layers.Dense(units=num_classes, activation='softmax')(x)  # For multi-class classification

    # Create and return the model
    model = models.Model(inputs=inputs, outputs=outputs, name='CNN')
    return model



def cnn_model_9(input_shape, num_classes):

    # Define the input layer explicitly
    inputs = Input(shape=input_shape)

    # First convolutional block
    x = layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D(pool_size=(2, 2), strides=(1,1), padding='same')(x)

    # Second convolutional block
    x = layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), strides=(1,1), padding='same')(x)

    # Third convolutional block
    x = layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(x)

    # Flatten and fully connected layers
    x = layers.Flatten()(x)
    x = layers.Dense(units=256, activation='relu')(x)
    outputs = layers.Dense(units=num_classes, activation='softmax')(x)  # For multi-class classification

    # Create and return the model
    model = models.Model(inputs=inputs, outputs=outputs, name='CNN')
    return model



def cnn_model_10(input_shape, num_classes):

    # Define the input layer explicitly
    inputs = Input(shape=input_shape)

    # First convolutional block
    x = layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D(pool_size=(2, 2), strides=(1,1), padding='same')(x)

    # Second convolutional block
    x = layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), strides=(1,1), padding='same')(x)

    # Third convolutional block
    x = layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), strides=(1,1), padding='same')(x)

    # Flatten and fully connected layers
    x = layers.Flatten()(x)
    x = layers.Dense(units=256, activation='relu')(x)
    outputs = layers.Dense(units=num_classes, activation='softmax')(x)  # For multi-class classification

    # Create and return the model
    model = models.Model(inputs=inputs, outputs=outputs, name='CNN')
    return model



def cnn_model_11(input_shape, num_classes):

    # Define the input layer explicitly
    inputs = Input(shape=input_shape)

    # First convolutional block
    x = layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D(pool_size=(2, 2), strides=(1,1), padding='same')(x)

    # Second convolutional block
    x = layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), strides=(1,1), padding='same')(x)

    # Third convolutional block
    x = layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), strides=(1,1), padding='same')(x)

    # Fourth convolutional block
    x = layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(x)

    # Flatten and fully connected layers
    x = layers.Flatten()(x)
    x = layers.Dense(units=256, activation='relu')(x)
    outputs = layers.Dense(units=num_classes, activation='softmax')(x)  # For multi-class classification

    # Create and return the model
    model = models.Model(inputs=inputs, outputs=outputs, name='CNN')
    return model



def cnn_model_12(input_shape, num_classes):

    # Define the input layer explicitly
    inputs = Input(shape=input_shape)

    # First convolutional block
    x = layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D(pool_size=(2, 2), strides=(1,1), padding='same')(x)

    # Second convolutional block
    x = layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), strides=(1,1), padding='same')(x)

    # Third convolutional block
    x = layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), strides=(1,1), padding='same')(x)

    # Fourth convolutional block
    x = layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), strides=(1,1), padding='same')(x)

    # Flatten and fully connected layers
    x = layers.Flatten()(x)
    x = layers.Dense(units=128, activation='relu')(x)
    outputs = layers.Dense(units=num_classes, activation='softmax')(x)  # For multi-class classification

    # Create and return the model
    model = models.Model(inputs=inputs, outputs=outputs, name='CNN')
    return model



def cnn_model_13(input_shape, num_classes):

    # Define the input layer explicitly
    inputs = Input(shape=input_shape)

    # First convolutional block
    x = layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D(pool_size=(2, 2), strides=(1,1), padding='same')(x)

    # Second convolutional block
    x = layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), strides=(1,1), padding='same')(x)

    # Third convolutional block
    x = layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), strides=(1,1), padding='same')(x)

    # Fourth convolutional block
    x = layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), strides=(1,1), padding='same')(x)

    # Flatten and fully connected layers
    x = layers.Flatten()(x)
    x = layers.Dense(units=256, activation='relu')(x)
    outputs = layers.Dense(units=num_classes, activation='softmax')(x)  # For multi-class classification

    # Create and return the model
    model = models.Model(inputs=inputs, outputs=outputs, name='CNN')
    return model



def cnn_model_14(input_shape, num_classes):

    # Define the input layer explicitly
    inputs = Input(shape=input_shape)

    # First convolutional block
    x = layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D(pool_size=(2, 2), strides=(1,1), padding='same')(x)

    # Second convolutional block
    x = layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), strides=(1,1), padding='same')(x)

    # Third convolutional block
    x = layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), strides=(1,1), padding='same')(x)

    # Fourth convolutional block
    x = layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(x)

    # Flatten and fully connected layers
    x = layers.Flatten()(x)
    x = layers.Dense(units=512, activation='relu')(x)
    outputs = layers.Dense(units=num_classes, activation='softmax')(x)  # For multi-class classification

    # Create and return the model
    model = models.Model(inputs=inputs, outputs=outputs, name='CNN')
    return model



def cnn_model_15(input_shape, num_classes):

    # Define the input layer explicitly
    inputs = Input(shape=input_shape)

    # First convolutional block
    x = layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D(pool_size=(2, 2), strides=(1,1), padding='same')(x)

    # Second convolutional block
    x = layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), strides=(1,1), padding='same')(x)

    # Third convolutional block
    x = layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), strides=(1,1), padding='same')(x)

    # Fourth convolutional block
    x = layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), strides=(1,1), padding='same')(x)

    # Flatten and fully connected layers
    x = layers.Flatten()(x)
    x = layers.Dense(units=512, activation='relu')(x)
    outputs = layers.Dense(units=num_classes, activation='softmax')(x)  # For multi-class classification

    # Create and return the model
    model = models.Model(inputs=inputs, outputs=outputs, name='CNN')
    return model



def cnn_model_16(input_shape, num_classes):

    # Define the input layer explicitly
    inputs = Input(shape=input_shape)

    # First convolutional block
    x = layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same')(inputs)
    x = layers.BatchNormalization()(x)  # Batch normalization after Conv2D
    x = layers.Activation('relu')(x)  # Activation after batch normalization
    x = layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same')(x)

    # Second convolutional block
    x = layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same')(x)

    # Third convolutional block
    x = layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same')(x)

    # Fourth convolutional block
    x = layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same')(x)

    # Flatten and fully connected layers
    x = layers.Flatten()(x)
    x = layers.Dense(units=512)(x)  # No activation yet
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    outputs = layers.Dense(units=num_classes, activation='softmax')(x)  # For multi-class classification

    # Create and return the model
    model = models.Model(inputs=inputs, outputs=outputs, name='CNN_with_BatchNorm')
    return model