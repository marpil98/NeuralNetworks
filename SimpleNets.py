from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin, ClassifierMixin
from sklearn.utils.validation import check_is_fitted

from tensorflow.keras.layers import LSTM, Dense, Conv2D
from tensorflow.keras.layers import Flatten, BatchNormalization, Dropout, Activation, Input
from tensorflow.keras.regularizers import l2
from tensorflow.keras import Model 
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, TerminateOnNaN
from tensorflow.keras.optimizers import Adam

import plotly.graph_objects as go 

# Function generative basic architecture, which can be used
# in functional API tensorflow
callbacs = [
    ReduceLROnPlateau(patience=5, min_delta=.001), 
    TerminateOnNaN(), 
    EarlyStopping(patience=10, min_delta=.1, restore_best_weights=True)
    ]

def lstm(
    x, layers, outputs, activation, recursive_activation,
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
    activation : str
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
        
    if type(activation)==str:
        
        activation = [activation for _ in range(layers)]
        
    if type(recursive_activation)==str:
        
        recursive_activation = [recursive_activation for i in range(layers)]
    
    for i in range(layers):
        
        if batch_normalization:
            
            x=BatchNormalization()(x)
            
        if i == (layers - 1):
            
            x = LSTM(
                units=outputs[i], activation=activation[i], 
                recurrent_activation=recursive_activation[i],
                return_sequences=False, dropout=drop, 
                kernel_regularizer=l2(l2_ratio)
                )(x)
            
        else:
            
            print(activation[i])
            x = LSTM(
                units=outputs[i], activation=activation[i],
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


class LSTMRegressor(BaseEstimator, RegressorMixin):
    
    def __init__(
    self, input_shape, layers, activation="tanh", 
    recursive_activation="sigmoid", drop=0.0, l2_ratio=0.0, 
    out_act='linear', batch_normalization=True, callbacks=None
    ):
        super().__init__()

        # Zachowanie zgodności ze stylem scikit-learn
        self.input_shape = input_shape
        self.layers = layers
        self.activation = activation
        self.recursive_activation = recursive_activation
        self.drop = drop
        self.l2_ratio = l2_ratio
        self.out_act = out_act
        self.batch_normalization = batch_normalization
        self.callbacks = callbacks

        # Tworzenie modelu
        self.input = Input(input_shape)
        self._fun_model = lstm(
            x=self.input, layers=layers, outputs=1, activation=activation,
            recursive_activation=recursive_activation, drop=drop,
            l2_ratio=l2_ratio, out_act=out_act, 
            batch_normalization=batch_normalization
        )
        self.model = Model(inputs=self.input, outputs=self._fun_model)
        
    def fit(self, X, y, epochs=10, optimizer=Adam(learning_rate=.1), val_ratio=.1,
            loss="mse", batch_size=None):
        
        self.model.compile(
            optimizer=optimizer,
            loss=loss
        )
        print(X.shape)
        self._history = self.model.fit(
            x=X, 
            y=y,
            validation_split=val_ratio,
            callbacks=callbacs,
            batch_size=batch_size,
            epochs=epochs
        )
        
        self._is_fitted = True
        
        return self
    
    def predict(self, X):
        
        check_is_fitted(self)
        return self.model.predict(X)
        
    def plot_learning_curve(self):
        
        check_is_fitted(self)
        
        l = self._history.history['loss']
        v = self._history.history['val_loss']
        ep = self._history.epoch
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ep, y=l, name="training"))
        fig.add_trace(go.Scatter(x=ep, y=v, name="validation"))
        fig.update_layout(
            xaxis_title="Epochs",
            yaxis_title="Loss value",
            title="Learning curve",
            hovermode='x'
        )
        fig.show()
        return fig
    
    def __sklearn_is_fitted__(self):
        """
        Check fitted status and return a Boolean value.
        """
        return hasattr(self, "_is_fitted") and self._is_fitted