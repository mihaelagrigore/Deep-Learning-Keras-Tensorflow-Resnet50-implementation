# import packages
from typing import List, Tuple

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from tensorflow.keras.initializers import glorot_uniform


class ResNet50:

    @staticmethod
    def identity_block(X: tf.Tensor, level: int, block: int, filters: List[int]) -> tf.Tensor:
        """
        Creates an identity block (see figure 3.1 from readme)

        Input:
            X - input tensor of shape (m, height_prev, width_prev, chan_prev)
            level - integer, one of the 5 levels that our networks is conceptually divided into (see figure 3.1 in the readme file)
                  - level names have the form: conv2_x, conv3_x ... conv5_x
            block - each conceptual level has multiple blocks (1 identity and several convolutional blocks)
                    block is the number of this block within its conceptual layer
                    i.e. first block from level 2 will be named conv2_1
            filters - a list on integers, each of them defining the number of filters in each convolutional layer

        Output:
            X - tensor (m, height, width, chan)
        """

        # layers will be called conv{level}_iden{block}_{convlayer_number_within_block}'
        conv_name = f'conv{level}_{block}' + '_{layer}_{type}'

        # unpack number of filters to be used for each conv layer
        f1, f2, f3 = filters

        # the shortcut branch of the identity block
        # takes the value of the block input
        X_shortcut = X

        # first convolutional layer (plus batch norm & relu activation, of course)
        X = Conv2D(filters=f1, kernel_size=(1, 1), strides=(1, 1),
                   padding='valid', name=conv_name.format(layer=1, type='conv'),
                   kernel_initializer=glorot_uniform(seed=0))(X)
        X = BatchNormalization(axis=3, name=conv_name.format(layer=1, type='bn'))(X)
        X = Activation('relu', name=conv_name.format(layer=1, type='relu'))(X)

        # second convolutional layer
        X = Conv2D(filters=f2, kernel_size=(3, 3), strides=(1, 1),
                   padding='same', name=conv_name.format(layer=2, type='conv'),
                   kernel_initializer=glorot_uniform(seed=0))(X)
        X = BatchNormalization(axis=3, name=conv_name.format(layer=2, type='bn'))(X)
        X = Activation('relu')(X)

        # third convolutional layer
        X = Conv2D(filters=f3, kernel_size=(1, 1), strides=(1, 1),
                   padding='valid', name=conv_name.format(layer=3, type='conv'),
                   kernel_initializer=glorot_uniform(seed=0))(X)
        X = BatchNormalization(axis=3, name=conv_name.format(layer=3, type='bn'))(X)

        # add shortcut branch to main path
        X = Add()([X, X_shortcut])

        # relu activation at the end of the block
        X = Activation('relu', name=conv_name.format(layer=3, type='relu'))(X)

        return X

    @staticmethod
    def convolutional_block(X: tf.Tensor, level: int, block: int, filters: List[int], s: Tuple[int,int,int]=(2, 2)) -> tf.Tensor:
        """
        Creates a convolutional block (see figure 3.1 from readme)

        Input:
            X - input tensor of shape (m, height_prev, width_prev, chan_prev)
            level - integer, one of the 5 levels that our networks is conceptually divided into (see figure 3.1 in the readme file)
                  - level names have the form: conv2_x, conv3_x ... conv5_x
            block - each conceptual level has multiple blocks (1 identity and several convolutional blocks)
                    block is the number of this block within its conceptual layer
                    i.e. first block from level 2 will be named conv2_1
            filters - a list on integers, each of them defining the number of filters in each convolutional layer
            s   - stride of the first layer;
                - a conv layer with a filter that has a stride of 2 will reduce the width and height of its input by half

        Output:
            X - tensor (m, height, width, chan)
        """

        # layers will be called conv{level}_{block}_{convlayer_number_within_block}'
        conv_name = f'conv{level}_{block}' + '_{layer}_{type}'

        # unpack number of filters to be used for each conv layer
        f1, f2, f3 = filters

        # the shortcut branch of the convolutional block
        X_shortcut = X

        # first convolutional layer
        X = Conv2D(filters=f1, kernel_size=(1, 1), strides=s, padding='valid',
                   name=conv_name.format(layer=1, type='conv'),
                   kernel_initializer=glorot_uniform(seed=0))(X)
        X = BatchNormalization(axis=3, name=conv_name.format(layer=1, type='bn'))(X)
        X = Activation('relu', name=conv_name.format(layer=1, type='relu'))(X)

        # second convolutional layer
        X = Conv2D(filters=f2, kernel_size=(3, 3), strides=(1, 1), padding='same',
                   name=conv_name.format(layer=2, type='conv'),
                   kernel_initializer=glorot_uniform(seed=0))(X)
        X = BatchNormalization(axis=3, name=conv_name.format(layer=2, type='bn'))(X)
        X = Activation('relu', name=conv_name.format(layer=2, type='relu'))(X)

        # third convolutional layer
        X = Conv2D(filters=f3, kernel_size=(1, 1), strides=(1, 1), padding='valid',
                   name=conv_name.format(layer=3, type='conv'),
                   kernel_initializer=glorot_uniform(seed=0))(X)
        X = BatchNormalization(axis=3, name=conv_name.format(layer=3, type='bn'))(X)

        # shortcut path
        X_shortcut = Conv2D(filters=f3, kernel_size=(1, 1), strides=s, padding='valid',
                            name=conv_name.format(layer='short', type='conv'),
                            kernel_initializer=glorot_uniform(seed=0))(X_shortcut)
        X_shortcut = BatchNormalization(axis=3, name=conv_name.format(layer='short', type='bn'))(X_shortcut)

        # add shortcut branch to main path
        X = Add()([X, X_shortcut])

        # nonlinearity
        X = Activation('relu', name=conv_name.format(layer=3, type='relu'))(X)

        return X

    @staticmethod
    def build(input_size: Tuple[int, int, int], classes: int) -> Model:
        """
            Builds the ResNet50 model (see figure 4.2 from readme)
    
            Input:
                - input_size - a (height, width, chan) tuple, the shape of the input images
                - classes - number of classes the model must learn
    
            Output:
                model - a Keras Model() instance
        """

        # tensor placeholder for the model's input
        X_input = Input(input_size)

        ### Level 1 ###

        # padding
        X = ZeroPadding2D((3, 3))(X_input)

        # convolutional layer, followed by batch normalization and relu activation
        X = Conv2D(filters=64, kernel_size=(7, 7), strides=(2, 2),
                   name='conv1_1_1_conv',
                   kernel_initializer=glorot_uniform(seed=0))(X)
        X = BatchNormalization(axis=3, name='conv1_1_1_nb')(X)
        X = Activation('relu')(X)

        ### Level 2 ###

        # max pooling layer to halve the size coming from the previous layer
        X = MaxPooling2D((3, 3), strides=(2, 2))(X)

        # 1x convolutional block
        X = ResNet50.convolutional_block(X, level=2, block=1, filters=[64, 64, 256], s=(1, 1))

        # 2x identity blocks
        X = ResNet50.identity_block(X, level=2, block=2, filters=[64, 64, 256])
        X = ResNet50.identity_block(X, level=2, block=3, filters=[64, 64, 256])

        ### Level 3 ###

        # 1x convolutional block
        X = ResNet50.convolutional_block(X, level=3, block=1, filters=[128, 128, 512], s=(2, 2))

        # 3x identity blocks
        X = ResNet50.identity_block(X, level=3, block=2, filters=[128, 128, 512])
        X = ResNet50.identity_block(X, level=3, block=3, filters=[128, 128, 512])
        X = ResNet50.identity_block(X, level=3, block=4, filters=[128, 128, 512])

        ### Level 4 ###
        # 1x convolutional block
        X = ResNet50.convolutional_block(X, level=4, block=1, filters=[256, 256, 1024], s=(2, 2))
        # 5x identity blocks
        X = ResNet50.identity_block(X, level=4, block=2, filters=[256, 256, 1024])
        X = ResNet50.identity_block(X, level=4, block=3, filters=[256, 256, 1024])
        X = ResNet50.identity_block(X, level=4, block=4, filters=[256, 256, 1024])
        X = ResNet50.identity_block(X, level=4, block=5, filters=[256, 256, 1024])
        X = ResNet50.identity_block(X, level=4, block=6, filters=[256, 256, 1024])

        ### Level 5 ###
        # 1x convolutional block
        X = ResNet50.convolutional_block(X, level=5, block=1, filters=[512, 512, 2048], s=(2, 2))
        # 2x identity blocks
        X = ResNet50.identity_block(X, level=5, block=2, filters=[512, 512, 2048])
        X = ResNet50.identity_block(X, level=5, block=3, filters=[512, 512, 2048])

        # Pooling layers
        X = AveragePooling2D(pool_size=(2, 2), name='avg_pool')(X)

        # Output layer
        X = Flatten()(X)
        X = Dense(classes, activation='softmax', name='fc_' + str(classes),
                  kernel_initializer=glorot_uniform(seed=0))(X)

        # Create model
        model = Model(inputs=X_input, outputs=X, name='ResNet50')

        return model
