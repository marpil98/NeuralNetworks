import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D,Conv1D, BatchNormalization, ReLU, GlobalAveragePooling2D,GlobalAveragePooling1D
from tensorflow.keras.layers import  Dense, ELU,Dropout,AveragePooling2D,LeakyReLU,Flatten, MaxPooling2D, PReLU,Activation
from tensorflow.keras.models import Model
from tensorflow.keras.activations import gelu

class ResNet():

    def __init__(self, filters,activation_in,activation_out=None, kernel_size=5,dropout_rate = .45, stride=2, output_units=1):
        self.filters = filters
        self.kernel_size = kernel_size
        self.dropout_rate = dropout_rate
        self.stride = stride
        self.activation = activation_in
        self.output_units = output_units
        if activation_out==None:
            self.activation_out = self.activation
        else:
            self.activation_out = activation_out
    def residual_block(self,x):
        # Convolutional layer 1
        x_shortcut = x
        x = Conv2D(self.filters, kernel_size=self.kernel_size, strides=self.stride, padding='same')(x)
        x = BatchNormalization()(x)
        x = self.activation(x)
        
        x = Dropout(self.dropout_rate)(x)
        # Convolutional layer 2
        x = Conv2D(self.filters, kernel_size=self.kernel_size, padding='same')(x)
        x = BatchNormalization()(x)
        
        # Residual connection
        if self.stride != 1 or x_shortcut.shape[-1] != self.filters:
            x_shortcut = Conv2D(self.filters, kernel_size=1, strides=self.stride, padding='same')(x_shortcut)
        x = tf.keras.layers.add([x, x_shortcut])
        x = self.activation_out(x)
        return x

    def build_resnet(self,x, num_blocks=3,pooling = GlobalAveragePooling2D()):
        #input_layer = Input(shape=input_shape)
        #x = input_layer
        num_filters = self.filters
        # Initial convolutional layer
        x = Conv2D(num_filters, kernel_size=7, padding='same', strides=2)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        if pooling == GlobalAveragePooling2D() or pooling == GlobalAveragePooling2D():
            for _ in range(num_blocks):
                x = self.residual_block(x)
                if x.shape[1] >= 2 and x.shape[2] >= 2:
                    num_filters*=2
            # Global average pooling
            x = pooling(x)
        else:
            # Residual blocks
            for _ in range(num_blocks):
                x = self.residual_block(x)
                if x.shape[1] >= 2 and x.shape[2] >= 2:
                    # Pooling
                    x = pooling(x)
                    num_filters*=2
            # Flatten
        x = Flatten()(x)
        x = Dense(units = 1, activation = 'linear')(x)
        return x
    def stworz_i_wyucz_Model(self,input_shape,X_train, y_train, num_blocks=3,pooling = GlobalAveragePooling2D(),epochs = 30,batch_size = 32,loss_function = tf.keras.losses.MeanAbsoluteError(),optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)):
        # Defining input shape and building the model
        # Flatten
        input = Input(input_shape)
        x = Flatten()(x)
        # Fully connected layers for regression
        x = Dense(self.output_units, activation=self.activation_out)(x)
        x = self.build_resnet(x, num_blocks,pooling)
        model = Model(input,x)
        model.summary()

        model.compile(optimizer=optimizer, loss=loss_function)
        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)

        return model
    
class ResNet1D():
    def __init__(self, filters,activation_in,activation_out=None, kernel_size=5,dropout_rate = .45, stride=2, output_units=1):
        self.filters = filters
        self.kernel_size = kernel_size
        self.dropout_rate = dropout_rate
        self.stride = stride
        self.activation = activation_in
        self.output_units = output_units
        if activation_out==None:
            self.activation_out = self.activation
        else:
            self.activation_out = activation_out
    def residual_block(self, x):
        # Convolutional layer 1
        x_shortcut = x
        x = Conv1D(self.filters, kernel_size=self.kernel_size, strides=self.stride, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation(self.activation)(x)  # Użyj nazwy funkcji aktywacji
        x = Dropout(self.dropout_rate)(x)
        # Convolutional layer 2
        x = Conv1D(self.filters, kernel_size=self.kernel_size, padding='same')(x)
        x = BatchNormalization()(x)
        
        # Residual connection
        if self.stride != 1 or x_shortcut.shape[-1] != self.filters:
            x_shortcut = Conv1D(self.filters, kernel_size=1, strides=self.stride, padding='same')(x_shortcut)
        x = tf.keras.layers.add([x, x_shortcut])
        x = Activation(self.activation_out)(x)  # Użyj nazwy funkcji aktywacji
        return x

    def build_resnet(self,x, num_blocks=3,pooling = None):
        #input_layer = Input(shape=input_shape)
        #x = input_layer
        num_filters = self.filters
        # Initial convolutional layer
        x = Conv1D(num_filters, kernel_size=7, padding='same', strides=2)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        if pooling != None:
            if pooling == GlobalAveragePooling1D():
                for _ in range(num_blocks):
                    x = self.residual_block(x)
                    if x.shape[1] >= 2 and x.shape[2] >= 2:
                        num_filters*=2
                # Global average pooling
                x = pooling(x)
            else:
                # Residual blocks
                for _ in range(num_blocks):
                    x = self.residual_block(x)
                    if x.shape[1] >= 2 and x.shape[2] >= 2:
                        # Pooling
                        x = pooling(x)
                        num_filters*=2
            
        return x
    
    def stworz_i_wyucz_Model(self,input_shape,X_train, y_train, num_blocks=3,pooling = None,epochs = 30,batch_size = 32,loss_function = tf.keras.losses.MeanAbsoluteError(),optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)):
        # Defining input shape and building the model
        input = Input(input_shape)
        x = self.build_resnet(input, num_blocks,pooling)
        # Flatten
        x = Flatten()(x)
        # Fully connected layers for regression
        x = Dense(self.output_units, activation=self.activation_out)(x)
        
        model = Model(input, outputs=x)
        model.summary()

        model.compile(optimizer=optimizer, loss=loss_function)
        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)

        return model

class ResNetMLP():
    def __init__(self, activation_in,activation_out=None,dropout_rate = .45, output_units=2):

        self.dropout_rate = dropout_rate
        self.activation = activation_in
        self.output_units = output_units
        if activation_out==None:
            self.activation_out = self.activation
        else:
            self.activation_out = activation_out
    def residual_block(self,x):
        # Convolutional layer 1
        x_shortcut = x
        x = Dense(units = x.shape[1])(x)
        x = BatchNormalization()(x)
        x = self.activation(x)
        
        x = Dropout(self.dropout_rate)(x)
        # Convolutional layer 2
        x = Dense(units = x.shape[-1])(x)
        x = BatchNormalization()(x)
        
       
        x = tf.keras.layers.add([x, x_shortcut])
        x = self.activation_out(x)
        return x

    def build_resnet(self,x, num_blocks=3):
        #input_layer = Input(shape=input_shape)
        #x = input_layer
        # Initial convolutional layer
        x = Dense(units = x.shape[1])(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)

        for _ in range(num_blocks):
            x = self.residual_block(x)
            
        return x
                
    
    def stworz_i_wyucz_Model(self,input_shape,X_train, y_train, num_blocks=3,epochs = 30,batch_size = 32,loss_function = tf.keras.losses.MeanAbsoluteError(),optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)):
        # Defining input shape and building the model
        input = Input(input_shape)
        x = self.build_resnet(input, num_blocks)
        # Flatten
        x = Flatten()(x)
        # Fully connected layers for regression
        x = Dense(self.output_units, activation=self.activation_out)(x)
        
        model = Model(input, outputs=x)
        model.summary()

        model.compile(optimizer=optimizer, loss=loss_function)
        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)

        return model
