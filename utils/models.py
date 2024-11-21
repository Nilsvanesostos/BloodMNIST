# utils/models.py

import tensorflow as tf
from tensorflow.keras import layers, models

def ConvPool(X, filters, conv_kernel=(3,3), conv_strides=(1, 1), conv_padding='same', dilation_rate=(1,1), activation='relu'):
    """
    Help function for convolutional + max pooling layers

    Arguments:
    X -- imput tensor

    Returns:
    model -- a Model() instance in TensorFlow
    """
    X = tf.keras.layers.Conv2D(filters, conv_kernel, strides=conv_strides, dilation_rate=dilation_rate, padding=conv_padding, activation=activation)(X)
    X = tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=(1,1), padding='same')(X)

    return X

def cnn(input_shape, num_classes=8):
    """
    Creates a CNN model for image classification.
    
    Parameters:
    - input_shape (tuple): The shape of the input image (height, width, channels).
    
    Returns:
    - model (tensorflow.keras.Model): The compiled CNN model.
    """

    X_input = tf.keras.Input(input_shape)

    # 1st Convolutional Layer + Pooling
    X = ConvPool(X_input, filters=32, conv_kernel=(3, 3), conv_strides=(1, 1), conv_padding='same', activation='relu')

    # 2nd Convolutional Layer + Pooling
    X = ConvPool(X, filters=64, conv_kernel=(3, 3), conv_strides=(1, 1), conv_padding='same', activation='relu')

    # 3rd Convolutional Layer + Pooling
    X = ConvPool(X, filters=128, conv_kernel=(3, 3), conv_strides=(1, 1), conv_padding='same', activation='relu')

    # 4th Convolutional Layer + Pooling
    X = ConvPool(X, filters=256, conv_kernel=(3, 3), conv_strides=(1, 1), conv_padding='same', activation='relu')

    # Flatten the output to feed into the Dense layer
    X = tf.keras.layers.Flatten()(X)

    # 1st Fully Connected Layer
    X = tf.keras.layers.Dense(256, activation='relu')(X)

    # 2nd Fully Connected Layer
    X = tf.keras.layers.Dense(256, activation='relu')(X)

    # Output Layer
    X = tf.keras.layers.Dense(num_classes, activation='softmax')(X)

    # Create model. This creates your Keras model instance, you'll use this instance to train/test the model.
    model = tf.keras.Model(inputs = X_input, outputs = X, name='cnn')


    return model

def dilated_cnn(input_shape, num_classes=8):
    """
    Creates a dilated CNN model for image classification.
    
    Parameters:
    - input_shape (tuple): The shape of the input image (height, width, channels).
    
    Returns:
    - model (tensorflow.keras.Model): The compiled CNN model.
    """

    X_input = tf.keras.Input(input_shape)

    # 1st Convolutional Layer + Pooling
    X = ConvPool(X_input, filters=32, conv_kernel=(3, 3), conv_strides=(1, 1), conv_padding='same', dilation_rate=(3, 3), activation='relu')

    # 2nd Convolutional Layer + Pooling
    X = ConvPool(X, filters=64, conv_kernel=(3, 3), conv_strides=(1, 1), conv_padding='same', dilation_rate=(3, 3), activation='relu')

    # 3rd Convolutional Layer + Pooling
    X = ConvPool(X, filters=128, conv_kernel=(3, 3), conv_strides=(1, 1), conv_padding='same', dilation_rate=(3, 3), activation='relu')

    # 4th Convolutional Layer + Pooling
    X = ConvPool(X, filters=256, conv_kernel=(3, 3), conv_strides=(1, 1), conv_padding='same', dilation_rate=(3, 3), activation='relu')

    # Flatten the output to feed into the Dense layer
    X = tf.keras.layers.Flatten()(X)

    # 1st Fully Connected Layer
    X = tf.keras.layers.Dense(256, activation='relu')(X)

    # 2nd Fully Connected Layer
    X = tf.keras.layers.Dense(256, activation='relu')(X)

    # Output Layer
    X = tf.keras.layers.Dense(num_classes, activation='softmax')(X)

    # Create model. This creates your Keras model instance, you'll use this instance to train/test the model.
    model = tf.keras.Model(inputs = X_input, outputs = X, name='dilated_cnn')


    return model

def residual_block(x, filters, kernel_size=3, stride=1):
    """
    A residual block with two convolutional layers.
    """
    # Shortcut connection
    shortcut = x

    # First convolutional layer
    x = tf.keras.layers.Conv2D(filters, kernel_size=kernel_size, strides=stride, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)

    # Second convolutional layer
    x = tf.keras.layers.Conv2D(filters, kernel_size=kernel_size, strides=1, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)

    # Adjust shortcut to match shape if needed
    if shortcut.shape[-1] != filters:  # Check if channels differ
        shortcut = tf.keras.layers.Conv2D(filters, kernel_size=1, strides=stride, padding='same')(shortcut)
        shortcut = tf.keras.layers.BatchNormalization()(shortcut)

    # Add shortcut to the main path
    x = tf.keras.layers.Add()([x, shortcut])
    x = tf.keras.layers.Activation('relu')(x)

    return x

def resnet(input_shape, num_classes=8):
    """
    Builds a ResNet-like architecture.
    """
    X_input = tf.keras.Input(input_shape)

    # Initial convolutional layer
    X = tf.keras.layers.Conv2D(32, (3, 3), strides=1, padding='same')(X_input)
    X = tf.keras.layers.BatchNormalization()(X)
    X = tf.keras.layers.Activation('relu')(X)

    # Residual block 1
    X = residual_block(X, filters=32)

    # Residual block 2
    X = residual_block(X, filters=64)

    # Residual block 3
    X = residual_block(X, filters=128)

    # Residual block 4
    X = residual_block(X, filters=256)

    # Global Average Pooling
    X = tf.keras.layers.GlobalAveragePooling2D()(X)

    # Flatten the output to feed into the Dense layer
    X = tf.keras.layers.Flatten()(X)

    # 1st Fully Connected Layer
    X = tf.keras.layers.Dense(256, activation='relu')(X)

    # 2nd Fully Connected Layer
    X = tf.keras.layers.Dense(256, activation='relu')(X)

    # Output Layer
    X = tf.keras.layers.Dense(num_classes, activation='softmax')(X)

    # Create the model
    model = tf.keras.Model(inputs = X_input, outputs = X, name='resnet')

    return model

def ConvConvPool(X, filters, conv_kernel=(3,3), conv_strides=(1, 1), conv_padding='same', dilation_rate=(1,1), activation='relu'):
    """
    Block with 2 Conv layers + Max Pooling
    """
    X = tf.keras.layers.Conv2D(filters, conv_kernel, strides=conv_strides, dilation_rate=dilation_rate, padding=conv_padding, activation=activation)(X)
    X = tf.keras.layers.Conv2D(filters, conv_kernel, strides=conv_strides, dilation_rate=dilation_rate, padding=conv_padding, activation=activation)(X)
    X = tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=(1,1), padding='same')(X)

    return X

def vgg(input_shape, num_classes=8):
    """
    Creates a VGG-like architecture
    """
    X_input = tf.keras.Input(input_shape)

    # 1st Convolution Layer + Convolution Layer + Pooling
    X = ConvConvPool(X_input, filters=32, conv_kernel=(3, 3), conv_strides=(1, 1), conv_padding='same', activation='relu')

    # 2nd Convolution Layer + Convolution Layer + Pooling
    X = ConvConvPool(X, filters=64, conv_kernel=(3, 3), conv_strides=(1, 1), conv_padding='same', activation='relu')

    # 3rd Convolution Layer + Convolution Layer + Pooling
    X = ConvConvPool(X, filters=128, conv_kernel=(3, 3), conv_strides=(1, 1), conv_padding='same', activation='relu')

    # 4th Convolution Layer + Convolution Layer + Pooling
    X = ConvConvPool(X, filters=256, conv_kernel=(3, 3), conv_strides=(1, 1), conv_padding='same', activation='relu')
    
    # Flatten the output to feed into the Dense layer
    X = tf.keras.layers.Flatten()(X)

    # 1st Fully Connected Layer
    X = tf.keras.layers.Dense(256, activation='relu')(X)

    # 2nd Fully Connected Layer
    X = tf.keras.layers.Dense(256, activation='relu')(X)

    # Output Layer
    X = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    # Create the model
    model = tf.keras.Model(inputs = X_input, outputs = X, name='vgg')

    return model

def autoencoder(input_shape, latent_dim=32):
    """
    Build an autoencoder-like architecture
    """
    # Encoder
    encoder_input = tf.keras.layers.Input(shape=input_shape, name="encoder_input")
    X = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(encoder_input)
    X = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(X)
    X = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(X)
    X = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(X)
    X = tf.keras.layers.Flatten()(X)
    latent_output = tf.keras.layers.Dense(latent_dim, activation='relu', name="latent_output")(X)

    # Create the encoder
    encoder = tf.keras.Model(encoder_input, latent_output, name="encoder")

    # Decoder
    decoder_input = tf.keras.layers.Input(shape=(latent_dim,), name="decoder_input")
    X = tf.keras.layers.Dense(16 * 16 * 64, activation='relu')(decoder_input)  # Match flattened size
    X = tf.keras.layers.Reshape((16, 16, 64))(X)
    X = tf.keras.layers.Conv2DTranspose(64, (3, 3), activation='relu', padding='same')(X)
    X = tf.keras.layers.Conv2DTranspose(32, (3, 3), activation='relu', padding='same')(X)
    decoder_output = tf.keras.layers.Conv2DTranspose(3, (3, 3), activation='sigmoid', padding='same', name="decoder_output")(X)

    # Create the decoder
    decoder = tf.keras.Model(decoder_input, decoder_output, name="decoder")

    # Create the autoencoder
    autoencoder_input = encoder_input
    autoencoder_output = decoder(encoder(autoencoder_input))
    autoencoder = tf.keras.Model(autoencoder_input, autoencoder_output, name="autoencoder")

    return encoder, decoder, autoencoder

def autoclassifier(encoder, num_classes=8):
    """
    Build a classifier using the encoder part of the autoencoders and a CNN-like architecture. We set 
    encoder.trainable = False so that we do not have to train again the encoder.
    """
    # Freeze encoder weights if needed
    encoder.trainable = False

    # Classification head
    classifier_input = encoder.input
    X = encoder.output

    # Reshape latent features to 2D 
    X = tf.keras.layers.Reshape((4, 4, 2))(X)  # Remember we choose 32 features 

    # 1st Convolution Layer + Pooling
    X = ConvPool(X, filters=64, conv_kernel=(3, 3), conv_strides=(1, 1), conv_padding='same', activation='relu')

    # 2nd Convolution Layer + Pooling
    X = ConvPool(X, filters=128, conv_kernel=(3, 3), conv_strides=(1, 1), conv_padding='same', activation='relu')

    # Fully connected layers
    X = tf.keras.layers.Flatten()(X)
    X = tf.keras.layers.Dense(128, activation='relu')(X)
    X = tf.keras.layers.Dense(num_classes, activation='softmax', name="classifier_output")(X)

    # Create the model
    classifier = tf.keras.Model(classifier_input, X, name="autoclassifier")

    return classifier
