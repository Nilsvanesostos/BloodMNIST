# utils/models.py

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.initializers import glorot_uniform

# conv2d_bn block
def conv2d_bn(X_input, filters, kernel_size, strides, padding='same', activation=None, dilation_rate=(1,1), name=None):
    """
    Implementation of a conv block as defined above

    Arguments:
    X_input -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    filters -- integer, defining the number of filters in the CONV layer
    kernel_size -- (f1, f2) tuple of integers, specifying the shape of the CONV kernel
    s -- integer, specifying the stride to be used
    padding -- padding approach to be used
    name -- name for the layers

    Returns:
    X -- output of the conv2d_bn block, tensor of shape (n_H, n_W, n_C)
    """

    # Define layer names based on the base name
    conv_name = f"{name}_conv"
    bn_name = f"{name}_bn"
    activation_name = f"{name}_activation"

    X = layers.Conv2D(filters = filters, kernel_size = kernel_size, strides = strides, padding = padding, dilation_rate = dilation_rate, name = conv_name, 
                      kernel_initializer = glorot_uniform(seed=0))(X_input)
    X = layers.BatchNormalization(axis = 3, name = bn_name)(X)
    if activation is not None:
        X = layers.Activation(activation, name=activation_name)(X)
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

    # 1st Convolution Layer + Pooling
    X = conv2d_bn(X_input, filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', name='1')
    X = layers.MaxPooling2D(pool_size=(2,2), strides=(1,1), padding='same')(X)

    # 2nd Convolutional Layer + Pooling
    X = conv2d_bn(X, filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', name='2')
    X = layers.MaxPooling2D(pool_size=(2,2), strides=(1,1), padding='same')(X)

    # 3rd Convolutional Layer + Pooling
    X = conv2d_bn(X, filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', name='3')
    X = layers.MaxPooling2D(pool_size=(2,2), strides=(1,1), padding='same')(X)

    # 4th Convolutional Layer + Pooling
    X = conv2d_bn(X, filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', name='4')
    X = layers.MaxPooling2D(pool_size=(2,2), strides=(1,1), padding='same')(X)

    # Flatten the output to feed into the Dense layer
    X = layers.Flatten()(X)

    # # Fully Connected Layer
    # X = layers.Dense(512, activation='relu')(X)

    # Output Layer
    X = layers.Dense(num_classes, activation='softmax')(X)

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

    X_input = layers.Input(input_shape)

    # 1st Convolution Layer + Pooling
    X = conv2d_bn(X_input, filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', dilation_rate=(3, 3), name='1')
    X = layers.MaxPooling2D(pool_size=(2,2), strides=(1,1), padding='same')(X)

    # 2nd Convolutional Layer + Pooling
    X = conv2d_bn(X, filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', dilation_rate=(3, 3), name='2')
    X = layers.MaxPooling2D(pool_size=(2,2), strides=(1,1), padding='same')(X)

    # 3rd Convolutional Layer + Pooling
    X = conv2d_bn(X, filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', dilation_rate=(3, 3), name='3')
    X = layers.MaxPooling2D(pool_size=(2,2), strides=(1,1), padding='same')(X)

    # 4th Convolutional Layer + Pooling
    X = conv2d_bn(X, filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', dilation_rate=(3, 3), name='4')
    X = layers.MaxPooling2D(pool_size=(2,2), strides=(1,1), padding='same')(X)

    # Flatten the output to feed into the Dense layer
    X = layers.Flatten()(X)

    # # Fully Connected Layer
    # X = layers.Dense(512, activation='relu')(X)

    # Output Layer
    X = layers.Dense(num_classes, activation='softmax')(X)

    # Create model. This creates your Keras model instance, you'll use this instance to train/test the model.
    model = models.Model(inputs = X_input, outputs = X, name='dilated_cnn')


    return model

def identity_block(x, filter):
    # copy tensor to variable called x_skip
    x_skip = x
    # Layer 1
    x = layers.Conv2D(filter, (3,3), padding = 'same')(x)
    x = layers.BatchNormalization(axis=3)(x)
    x = layers.Activation('relu')(x)
    # Layer 2
    x = layers.Conv2D(filter, (3,3), padding = 'same')(x)
    x = layers.BatchNormalization(axis=3)(x)
    # Add Residue
    x = layers.Add()([x, x_skip])     
    x = layers.Activation('relu')(x)
    return x

def convolutional_block(x, filter):
    # copy tensor to variable called x_skip
    x_skip = x
    # Layer 1
    x = layers.Conv2D(filter, (3,3), padding = 'same', strides = (2,2))(x)
    x = layers.BatchNormalization(axis=3)(x)
    x = layers.Activation('relu')(x)
    # Layer 2
    x = layers.Conv2D(filter, (3,3), padding = 'same')(x)
    x = layers.BatchNormalization(axis=3)(x)
    # Processing Residue with conv(1,1)
    x_skip = layers.Conv2D(filter, (1,1), strides = (2,2))(x_skip)
    # Add Residue
    x = layers.Add()([x, x_skip])     
    x = layers.Activation('relu')(x)
    return x

def resnet34(input_shape, classes = 8):
    # Step 1 (Setup Input Layer)
    x_input = layers.Input(input_shape)
    x = layers.ZeroPadding2D((3, 3))(x_input)
    # Step 2 (Initial Conv layer along with maxPool)
    x = layers.Conv2D(64, kernel_size=7, strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPool2D(pool_size=3, strides=2, padding='same')(x)
    # Define size of sub-blocks and initial filter size
    block_layers = [3, 4, 6, 3]
    filter_size = 64
    # Step 3 Add the Resnet Blocks
    for i in range(4):
        if i == 0:
            # For sub-block 1 Residual/Convolutional block not needed
            for j in range(block_layers[i]):
                x = identity_block(x, filter_size)
        else:
            # One Residual/Convolutional Block followed by Identity blocks
            # The filter size will go on increasing by a factor of 2
            filter_size = filter_size*2
            x = convolutional_block(x, filter_size)
            for j in range(block_layers[i] - 1):
                x = identity_block(x, filter_size)
    # Step 4 End Dense Network
    x = layers.AveragePooling2D((2,2), padding = 'same')(x)
    x = layers.Flatten()(x)
    # x = layers.Dense(512, activation = 'relu')(x)
    x = layers.Dense(classes, activation = 'softmax')(x)
    model = models.Model(inputs = x_input, outputs = x, name = "resnet34")

    return model

def vgg(input_shape, num_classes=8):
    """
    Creates a VGG-like architecture
    """
    X_input = layers.Input(input_shape)

    # 1st Convolution Layer + Convolution Layer + Pooling
    X = conv2d_bn(X_input, filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', name='1_1')
    X = conv2d_bn(X, filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', name='1_2')
    X = layers.MaxPooling2D(pool_size=(2,2), strides=(1,1), padding='same')(X)
    
    # 2st Convolution Layer + Convolution Layer + Pooling
    X = conv2d_bn(X, filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', name='2_1')
    X = conv2d_bn(X, filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', name='2_2')
    X = layers.MaxPooling2D(pool_size=(2,2), strides=(1,1), padding='same')(X)

    # 3st Convolution Layer + Convolution Layer + Pooling
    X = conv2d_bn(X, filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', name='3_1')
    X = conv2d_bn(X, filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', name='3_2')
    X = layers.MaxPooling2D(pool_size=(2,2), strides=(1,1), padding='same')(X)

    # 4st Convolution Layer + Convolution Layer + Pooling
    X = conv2d_bn(X, filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', name='4_1')
    X = conv2d_bn(X, filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', name='4_2')
    X = layers.MaxPooling2D(pool_size=(2,2), strides=(1,1), padding='same')(X)

    # Flatten the output to feed into the Dense layer
    X = layers.Flatten()(X)

    # # Fully Connected Layer
    # X = layers.Dense(512, activation='relu')(X)

    # Output Layer
    X = layers.Dense(num_classes, activation='softmax')(X)

    # Create the model
    model = models.Model(inputs = X_input, outputs = X, name='vgg')

    return model

def autoencoder(input_shape, latent_dim=32):
    """
    Build an autoencoder-like architecture
    """
    # Encoder
    encoder_input = layers.Input(shape=input_shape, name="encoder_input")

    X = conv2d_bn(encoder_input, filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='elu', name='encoder_1')
    X = layers.MaxPooling2D(pool_size=(2,2), strides=(1,1), padding='same')(X)
    X = conv2d_bn(X, filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='elu', name='encoder_2')
    X = layers.MaxPooling2D(pool_size=(2,2), strides=(1,1), padding='same')(X)
    X = conv2d_bn(X, filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='elu', name='encoder_3')
    X = layers.MaxPooling2D(pool_size=(2,2), strides=(1,1), padding='same')(X)
    X = conv2d_bn(X, filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='elu', name='encoder_4')
    X = layers.MaxPooling2D(pool_size=(2,2), strides=(1,1), padding='same')(X)
    
    X = layers.Flatten()(X)
    latent_output = layers.Dense(latent_dim, name="latent_output")(X)

    # Create the encoder
    encoder = models.Model(encoder_input, latent_output, name="encoder")

    # Decoder
    decoder_input = layers.Input(shape=(latent_dim,), name="decoder_input")

    X = layers.Dense(4 * 4 * 256, activation='elu')(decoder_input)  # Match the encoder output size
    X = layers.Reshape((4, 4, 256))(X)
    X = layers.Conv2DTranspose(128, (3, 3), strides=2, activation='elu', padding='same')(X)  # (8, 8, 128)
    X = layers.Conv2DTranspose(64, (3, 3), strides=2, activation='elu', padding='same')(X)   # (16, 16, 64)
    X = layers.Conv2DTranspose(32, (3, 3), strides=2, activation='elu', padding='same')(X)   # (32, 32, 32)
    decoder_output = layers.Conv2DTranspose(3, (3, 3), strides=2, activation=None, padding='same', name="decoder_output")(X)  # (64, 64, 3)

    # Create the decoder
    decoder = models.Model(decoder_input, decoder_output, name="decoder")

    # Create the autoencoder
    autoencoder_input = encoder_input
    autoencoder_output = decoder(encoder(autoencoder_input))
    autoencoder = models.Model(autoencoder_input, autoencoder_output, name="autoencoder")

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
    X = layers.Reshape((4, 4, 2))(X)  # Remember we choose 32 features 

    # 1st Convolution Layer + Pooling
    X = conv2d_bn(X, filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', name='class_1')
    X = layers.MaxPooling2D(pool_size=(2,2), strides=(1,1), padding='same')(X)

    # 2nd Convolution Layer + Pooling
    X = conv2d_bn(X, filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', name='class_2')
    X = layers.MaxPooling2D(pool_size=(2,2), strides=(1,1), padding='same')(X)

    # Fully connected layers
    X = layers.Flatten()(X)
    # X = layers.Dense(512, activation='relu')(X)
    X = layers.Dense(num_classes, activation='softmax', name="classifier_output")(X)

    # Create the model
    classifier = models.Model(classifier_input, X, name="autoclassifier")

    return classifier

# stem_block
def stem_block(X_input):
    """
    Implementation of the stem block as defined above

    Arguments:
    X_input -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)

    Returns:
    X -- output of the stem block, tensor of shape (n_H, n_W, n_C)
    """

    # 1st Conv 
    X = conv2d_bn(X_input, filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', name='stem_1')

    # 2nd Conv
    X = conv2d_bn(X, filters=64, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu', name='stem_2')

    # 3rd Conv
    X = conv2d_bn(X, filters=96, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', name='stem_3')

    return X

# Inception-A block
def inception_a_block(X_input, base_name):
    """
    Implementation of the Inception-A block

    Arguments:
    X_input -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)

    Returns:
    X -- output of the block, tensor of shape (n_H, n_W, n_C)
    """

    # Branch 1
    branch1 = layers.AveragePooling2D(pool_size = (3, 3), strides = (1, 1),
                           padding = 'same', name = base_name + 'ia_branch_1_1')(X_input)
    branch1 = conv2d_bn(branch1, filters = 96, kernel_size = (1, 1),
                        strides = (1, 1), padding = 'same', activation='relu',
                        name = base_name + 'ia_branch_1_2')

    # Branch 2
    branch2 = conv2d_bn(X_input, filters = 96, kernel_size = (1, 1),
                        strides = (1, 1), padding = 'same', activation='relu',
                        name = base_name + 'ia_branch_2_1')

    # Branch 3
    branch3 = conv2d_bn(X_input, filters = 64, kernel_size = (1, 1),
                        strides = (1, 1), padding = 'same', activation='relu',
                        name = base_name + 'ia_branch_3_1')
    branch3 = conv2d_bn(branch3, filters = 96, kernel_size = (3, 3),
                        strides = (1, 1), padding = 'same', activation='relu',
                        name = base_name + 'ia_branch_3_2')

    # Branch 4
    branch4 = conv2d_bn(X_input, filters = 64, kernel_size = (1, 1),
                        strides = (1, 1), padding = 'same', activation='relu',
                        name = base_name + 'ia_branch_4_1')
    branch4 = conv2d_bn(branch4, filters = 96, kernel_size = (3, 3),
                        strides = (1, 1), padding = 'same', activation='relu',
                        name = base_name + 'ia_branch_4_2')
    branch4 = conv2d_bn(branch4, filters = 96, kernel_size = (3, 3),
                        strides = (1, 1), padding = 'same', activation='relu',
                        name = base_name + 'ia_branch_4_3')

    # Concatenate branch1, branch2, branch3 and branch4 along the channel axis
    X = layers.Concatenate(axis=3)([branch1, branch2, branch3, branch4])

    return X

# Inception-B block
def inception_b_block(X_input, base_name):
    """
    Implementation of the Inception-B block

    Arguments:
    X_input -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)

    Returns:
    X -- output of the block, tensor of shape (n_H, n_W, n_C)
    """

    # Branch 1
    branch1 = layers.AveragePooling2D(pool_size = (3, 3), strides = (1, 1),
                           padding = 'same', name = base_name + 'ib_branch_1_1')(X_input)
    branch1 = conv2d_bn(branch1, filters = 128, kernel_size = (1, 1),
                        strides = (1, 1), padding = 'same', activation='relu',
                        name = base_name + 'ib_branch_1_2')

    # Branch 2
    branch2 = conv2d_bn(X_input, filters = 384, kernel_size = (1, 1),
                        strides = (1, 1), padding = 'same', activation='relu',
                        name = base_name + 'ib_branch_2_1')

    # Branch 3
    branch3 = conv2d_bn(X_input, filters = 192, kernel_size = (1, 1),
                        strides = (1, 1), padding = 'same', activation='relu',
                        name = base_name + 'ib_branch_3_1')
    branch3 = conv2d_bn(branch3, filters = 224, kernel_size = (1, 7),
                        strides = (1, 1), padding = 'same', activation='relu',
                        name = base_name + 'ib_branch_3_2')
    branch3 = conv2d_bn(branch3, filters = 256, kernel_size = (7, 1),
                        strides = (1, 1), padding = 'same', activation='relu',
                        name = base_name + 'ib_branch_3_3')

    # Branch 4
    branch4 = conv2d_bn(X_input, filters = 192, kernel_size = (1, 1),
                        strides = (1, 1), padding = 'same', activation='relu',
                        name = base_name + 'ib_branch_4_1')
    branch4 = conv2d_bn(branch4, filters = 192, kernel_size = (1, 7),
                        strides = (1, 1), padding = 'same', activation='relu',
                        name = base_name + 'ib_branch_4_2')
    branch4 = conv2d_bn(branch4, filters = 224, kernel_size = (7, 1),
                        strides = (1, 1), padding = 'same', activation='relu',
                        name = base_name + 'ib_branch_4_3')
    branch4 = conv2d_bn(branch4, filters = 224, kernel_size = (1, 7),
                        strides = (1, 1), padding = 'same', activation='relu',
                        name = base_name + 'ib_branch_4_4')
    branch4 = conv2d_bn(branch4, filters = 256, kernel_size = (7, 1),
                        strides = (1, 1), padding = 'same', activation='relu',
                        name = base_name + 'ib_branch_4_5')

    # Concatenate branch1, branch2, branch3 and branch4 along the channel axis
    X = layers.Concatenate(axis=3)([branch1, branch2, branch3, branch4])

    return X

# Inception-C block
def inception_c_block(X_input, base_name):
    """
    Implementation of the Inception-C block

    Arguments:
    X_input -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)

    Returns:
    X -- output of the block, tensor of shape (n_H, n_W, n_C)
    """

    # Branch 1
    branch1 = layers.AveragePooling2D(pool_size = (3, 3), strides = (1, 1),
                           padding = 'same', name = base_name + 'ic_branch_1_1')(X_input)
    branch1 = conv2d_bn(branch1, filters = 256, kernel_size = (1, 1),
                        strides = (1, 1), padding = 'same', activation='relu',
                        name = base_name + 'ic_branch_1_2')

    # Branch 2
    branch2 = conv2d_bn(X_input, filters = 256, kernel_size = (1, 1),
                        strides = (1, 1), padding = 'same', activation='relu',
                        name = base_name + 'ic_branch_2_1')

    # Branch 3
    branch3 = conv2d_bn(X_input, filters = 384, kernel_size = (1, 1),
                        strides = (1, 1), padding = 'same', activation='relu',
                        name = base_name + 'ic_branch_3_1')
    branch3_1 = conv2d_bn(branch3, filters = 256, kernel_size = (1, 3),
                        strides = (1, 1), padding = 'same', activation='relu',
                        name = base_name + 'ic_branch_3_2')
    branch3_2 = conv2d_bn(branch3, filters = 256, kernel_size = (3, 1),
                        strides = (1, 1), padding = 'same', activation='relu',
                        name = base_name + 'ic_branch_3_3')

    # Branch 4
    branch4 = conv2d_bn(X_input, filters = 384, kernel_size = (1, 1),
                        strides = (1, 1), padding = 'same', activation='relu',
                        name = base_name + 'ic_branch_4_1')
    branch4 = conv2d_bn(branch4, filters = 448, kernel_size = (1, 3),
                        strides = (1, 1), padding = 'same', activation='relu',
                        name = base_name + 'ic_branch_4_2')
    branch4 = conv2d_bn(branch4, filters = 512, kernel_size = (3, 1),
                        strides = (1, 1), padding = 'same', activation='relu',
                        name = base_name + 'ic_branch_4_3')
    branch4_1 = conv2d_bn(branch4, filters = 256, kernel_size = (3, 1),
                        strides = (1, 1), padding = 'same', activation='relu',
                        name = base_name + 'ic_branch_4_4')
    branch4_2 = conv2d_bn(branch4, filters = 256, kernel_size = (1, 3),
                        strides = (1, 1), padding = 'same', activation='relu',
                        name = base_name + 'ic_branch_4_5')

    # Concatenate branch1, branch2, branch3_1, branch3_2, branch4_1 and branch4_2 along the channel axis
    X = layers.Concatenate(axis=3)([branch1, branch2, branch3_1, branch3_2, branch4_1, branch4_2])

    return X

# Reduction-A block
def reduction_a_block(X_input):
    """
    Implementation of the Inception-A block

    Arguments:
    X_input -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)

    Returns:
    X -- output of the block, tensor of shape (n_H, n_W, n_C)
    """

    # Branch 1
    branch1 = layers.AveragePooling2D(pool_size = (3, 3), strides = (2, 2), padding = 'same', name = 'ra_branch_1_1')(X_input)

    # Branch 2
    branch2 = conv2d_bn(X_input, filters = 96, kernel_size = (3, 3), strides = (2, 2), padding = 'same', activation='relu', name = 'ra_branch_2_1')

    # Branch 3
    branch3 = conv2d_bn(X_input, filters = 64, kernel_size = (1, 1), strides = (1, 1), padding = 'same', activation='relu', name = 'ra_branch_3_1')
    branch3 = conv2d_bn(branch3, filters = 96, kernel_size = (3, 3), strides = (2, 2), padding = 'same', activation='relu', name = 'ra_branch_3_2')

    # Concatenate branch1, branch2, branch3 and branch4 along the channel axis
    X = layers.Concatenate(axis=3)([branch1, branch2, branch3])

    return X

# Reduction-B block
def reduction_b_block(X_input):
    """
    Implementation of the Inception-B block

    Arguments:
    X_input -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)

    Returns:
    X -- output of the block, tensor of shape (n_H, n_W, n_C)
    """

    # Branch 1
    branch1 = layers.AveragePooling2D(pool_size = (3, 3), strides = (2, 2), padding = 'same', name = 'rb_branch_1_1')(X_input)

    # Branch 2
    branch2 = conv2d_bn(X_input, filters = 128, kernel_size = (1, 1), strides = (1, 1), padding = 'same', activation='relu', name = 'rb_branch_2_1')
    branch2 = conv2d_bn(branch2, filters=192, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu', name = 'rb_branch_2_2')

    # Branch 3
    branch3 = conv2d_bn(X_input, filters = 128, kernel_size = (1, 1), strides = (1, 1), padding = 'same', activation='relu', name = 'rb_branch_3_1')
    branch3 = conv2d_bn(branch3, filters = 128, kernel_size = (1, 7), strides = (1, 1), padding = 'same', activation='relu', name = 'rb_branch_3_2')
    branch3 = conv2d_bn(branch3, filters = 128, kernel_size = (7, 1), strides = (1, 1), padding = 'same', activation='relu', name = 'rb_branch_3_3')
    branch3 = conv2d_bn(branch3, filters=192, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu', name = 'rb_branch_3_4')

    # Concatenate branch1, branch2, branch3 and branch4 along the channel axis
    X = layers.Concatenate(axis=3)([branch1, branch2, branch3])

    return X

# Inception
def inception(input_shape, num_classes=8):
    """
    Implementation of the Inception architecture

    Arguments:
    input_shape -- shape of the images of the dataset
    classes -- integer, number of classes

    Returns:
    model -- a Model() instance in Keras
    """

   # Define the input as a tensor with shape input_shape
    X_input = layers.Input(input_shape)

    # Call the above functions for the stem, inception-a, reduction-a, inception-b, reduction-b and inception-c blocks
    X = stem_block(X_input)

    # Four Inception A blocks
    X = inception_a_block(X, 'a1')
    X = inception_a_block(X, 'a2')

    # Reduction A block
    X = reduction_a_block(X)

    # Seven Inception B blocks
    X = inception_b_block(X, 'b1')
    X = inception_b_block(X, 'b2')
    X = inception_b_block(X, 'b3')

    # Reduction B block
    X = reduction_b_block(X)

    # Three Inception C blocks
    X = inception_c_block(X, 'c1')

    # AVGPOOL
    kernel_pooling = (1,1) # check it in the model.summary() list of layers and dimensions
    X = layers.AveragePooling2D(kernel_pooling, name='avg_pool')(X)
    X = layers.Flatten()(X)

    # # Fully Connected Layer
    # X = layers.Dense(512, activation='relu')(X)

    # Dropout
    X = layers.Dropout(rate = 0.2)(X)

    # Output layer
    X = layers.Dense(num_classes, activation='softmax', name='fc')(X)

    # Create model
    model = models.Model(inputs = X_input, outputs = X, name='inception')

    return model

def vgg12(input_shape, num_classes=8):
    """
    Creates a VGG-like architecture
    """
    X_input = layers.Input(input_shape)

    # 1st layer
    X = conv2d_bn(X_input, filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', name='1_1')
    X = conv2d_bn(X, filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', name='1_2')
    X = layers.MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same')(X)
    
    # 2nd layer
    X = conv2d_bn(X, filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', name='2_1')
    X = conv2d_bn(X, filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', name='2_2')
    X = layers.MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same')(X)

    # 3rd layer
    X = conv2d_bn(X, filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', name='3_1')
    X = conv2d_bn(X, filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', name='3_2')
    X = layers.MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same')(X)

    # 4th layer
    X = conv2d_bn(X, filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', name='4_1')
    X = conv2d_bn(X, filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', name='4_2')
    X = conv2d_bn(X, filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', name='4_3')
    X = layers.MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same')(X)

    # Flatten the output to feed into the Dense layer
    X = layers.Flatten()(X)

    # Fully Connected Layer
    X = layers.Dense(1024, activation='relu')(X)
    X = layers.Dropout(0.3)(X)
    X = layers.Dense(1024, activation='relu')(X)
    X = layers.Dropout(0.3)(X)
    X = layers.Dense(256, activation='relu')(X)

    # Output Layer
    X = layers.Dense(num_classes, activation='softmax')(X)

    # Create the model
    model = models.Model(inputs = X_input, outputs = X, name='vgg')

    return model

def vgg15(input_shape, num_classes=8):
    """
    Creates a VGG-like architecture
    """
    X_input = layers.Input(input_shape)

    # 1st layer
    X = conv2d_bn(X_input, filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', name='1_1')
    X = conv2d_bn(X, filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', name='1_2')
    X = layers.MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same')(X)
    
    # 2nd layer
    X = conv2d_bn(X, filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', name='2_1')
    X = conv2d_bn(X, filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', name='2_2')
    X = layers.MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same')(X)

    # 3rd layer
    X = conv2d_bn(X, filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', name='3_1')
    X = conv2d_bn(X, filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', name='3_2')
    X = layers.MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same')(X)

    # 4th layer
    X = conv2d_bn(X, filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', name='4_1')
    X = conv2d_bn(X, filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', name='4_2')
    X = conv2d_bn(X, filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', name='4_3')
    X = layers.MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same')(X)

    # 5th layer
    X = conv2d_bn(X, filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', name='5_1')
    X = conv2d_bn(X, filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', name='5_2')
    X = conv2d_bn(X, filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', name='5_3')
    X = layers.MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same')(X)

    # Flatten the output to feed into the Dense layer
    X = layers.Flatten()(X)

    # Fully Connected Layer
    X = layers.Dense(1024, activation='relu')(X)
    X = layers.Dropout(0.5)(X)
    X = layers.Dense(1024, activation='relu')(X)
    X = layers.Dropout(0.5)(X)
    X = layers.Dense(256, activation='relu')(X)

    # Output Layer
    X = layers.Dense(num_classes, activation='softmax')(X)

    # Create the model
    model = models.Model(inputs = X_input, outputs = X, name='vgg')

    return model

def vgg18(input_shape, num_classes=8):
    """
    Creates a VGG-like architecture
    """
    X_input = layers.Input(input_shape)

    # 1st layer
    X = conv2d_bn(X_input, filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', name='1_1')
    X = conv2d_bn(X, filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', name='1_2')
    X = layers.MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same')(X)
    
    # 2nd layer
    X = conv2d_bn(X, filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', name='2_1')
    X = conv2d_bn(X, filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', name='2_2')
    X = layers.MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same')(X)

    # 3rd layer
    X = conv2d_bn(X, filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', name='3_1')
    X = conv2d_bn(X, filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', name='3_2')
    X = layers.MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same')(X)

    # 4th layer
    X = conv2d_bn(X, filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', name='4_1')
    X = conv2d_bn(X, filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', name='4_2')
    X = conv2d_bn(X, filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', name='4_3')
    X = layers.MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same')(X)

    # 5th layer
    X = conv2d_bn(X, filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', name='5_1')
    X = conv2d_bn(X, filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', name='5_2')
    X = conv2d_bn(X, filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', name='5_3')
    X = layers.MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same')(X)

    # 6th layer
    X = conv2d_bn(X, filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', name='6_1')
    X = conv2d_bn(X, filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', name='6_2')
    X = conv2d_bn(X, filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', name='6_3')
    X = layers.MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same')(X)

    # Flatten the output to feed into the Dense layer
    X = layers.Flatten()(X)

    # Fully Connected Layer
    X = layers.Dense(1024, activation='relu')(X)
    X = layers.Dropout(0.3)(X)
    X = layers.Dense(1024, activation='relu')(X)
    X = layers.Dropout(0.3)(X)
    X = layers.Dense(256, activation='relu')(X)

    # Output Layer
    X = layers.Dense(num_classes, activation='softmax')(X)

    # Create the model
    model = models.Model(inputs = X_input, outputs = X, name='vgg')

    return model


