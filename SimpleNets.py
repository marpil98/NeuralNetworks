
from tensorflow.keras.layers import LSTM, Dense, Conv2D, Flatten, BatchNormalization, Dropout,Activation
from tensorflow.keras.regularizers import l2

# Function generative basic architecture, which can be used
# in functional API tensorflow

def recursive_net(
    x, layers, outputs, LSTM_activation, recursive_activation,
    drop, l2_ratio, out_act='linear', batch_normalization=True
    ):
    """
    Function is generating LSTM neural network
    
    Parameters
    ----------
    x : tf.keras.layers
        Input
    layers : int
        Number of layers
    outputs : int
        Number of outputs - determines problem's type - 1 = regression,
        more = classification/multiple regression
    LSTM_activation : str
        Activation function for LSTM part
    recursive_activation : str
        Activation function for recursive part
    drop : float, [0.,1.]
        Part of neurons which will be dropped. 
        One of the ways to counteract overtraining
    l2_ratio : _type_
        Coefficient determining strength of l2 regularization. 
        One of the ways to counteract overtraining
    out_act : str, optional
        Activation function the last one layer, by default 'linear'
    batch_normalization : bool, optional
        Determines if in each layer data will be
        batch-normalized, by default True

    Returns
    -------
    tf.keras.layers
        Output layer. Can be used in another layer as input
    """
    if type(outputs) == int:
        
        outputs = [outputs for _ in range(layers)]
        
    if type(LSTM_activation)==str:
        
        LSTM_activation = [LSTM_activation for _ in range(layers)]
        
    if type(recursive_activation)==str:
        
        recursive_activation = [recursive_activation for i in range(layers)]
    
    for i in range(layers):
        
        if batch_normalization:
            
            x=BatchNormalization()(x)
            
        if i == (layers - 1):
            
            x = LSTM(
                units=outputs[i], activation=LSTM_activation[i], 
                recurrent_activation=recursive_activation[i],
                return_sequences=False, dropout=drop, 
                kernel_regularizer=l2(l2_ratio)
                )(x)
            
        else:
            
            print(LSTM_activation[i])
            x = LSTM(
                units=outputs[i], activation=LSTM_activation[i],
                recurrent_activation=recursive_activation[i],
                return_sequences=True, dropout=drop,
                kernel_regularizer=l2(l2_ratio)
                )(x)
            
    out = Dense(units = 1,activation = out_act)(x)
    
    return out

def convolutional_net(
    x, layers, filters, strides, kernel_size, activation, pooling, 
    drop, l2_ratio, pooling_type='global', out_act='linear', batch_normalization=True
    ):
    """
    Function is generating convolutional neural network

    Parameters
    ----------
    x : tf.keras.layers
        Input
    layers : int 
        Number of layers
    filters : int or list
        Number of filters per layer using in convolution.
        If an int is passed, there is the same number 
        of filters for each layer
    strides : int or list
        Number of strides determines one filter's step
        in each layer. If an int is passed, there is the 
        same number of strides for each layer
    kernel_size : int or tuple
        Kernel size
    activation : str or list
        Activation function between layers.
        For each layer can be passed another function
    pooling : tf.keras.layers or list
        Specifies a type of pooling, such as MaxPooling2D.
    drop : float, [0.,1.]
        Part of neurons which will be dropped. 
        One of the ways to counteract overtraining
    l2_ratio : _type_
        Coefficient determining strength of l2 regularization. 
        One of the ways to counteract overtraining
    pooling_type : str, optional
        Determines if pooling is global or local, by default 'global'
    out_act : str, optional
        Activation function the last one layer, by default 'linear'
    batch_normalization : bool, optional
        Determines if in each layer data will be
        batch-normalized, by default True

    Returns
    -------
    tf.keras.layers
        Output layer. Can be used in another layer as input
    """
    if type(filters) == int:
        
        filters = [filters for _ in range(layers)]
        
    if type(strides) == int:
        
        strides = [strides for _ in range(layers)]
        
    if type(activation) != list:
        
        activation = [activation for _ in range(layers)]
        
    if (pooling_type != 'global' and type(pooling) != list):
        
        pooling = [pooling for _ in range(layers)]
        
    for i in range(layers):
        
        if batch_normalization:
            
            x = BatchNormalization()(x)
            
        x = Conv2D(
            filters=filters[i], kernel_size=kernel_size, strides=strides[i],
            kernel_regularizer=l2(l2_ratio), padding='same'
            )(x)
        
        x = Activation(activation[i])(x)
        
        if pooling_type != 'global':
            
            x = pooling[i](x)
            
        x = Dropout(drop)(x)
        
    if pooling_type=='global':
        
        x = pooling(x)
        
    x = Flatten()(x)
    out = Dense(units=1, activation=out_act)(x)
    
    return out

def MLP(x, layers, neurons, activation):
    """
    Function is generating multi-layer perceptron

    Parameters
    ----------
    x : tf.keras.layers
        Input
    layers : int 
        Number of layers
    neurons : int or list
        Number of neurons in each layer.If an int is passed, 
        there is the same number of neurons for each layer
    activation : str or list
        Activation function between layers.
        For each layer can be passed another function

    Returns
    -------
    _type_
        _description_
    """
    if type(neurons) == int:
        
        neurons = [neurons for _ in range(neurons)]
        
    if type(activation) != list:
        
        activation = [activation for _ in range(activation)]

    for i in range(layers):
        
        x = Dense(units=neurons[i])(x)
        
    out = Activation(activation[i])(x)
    
    return out
