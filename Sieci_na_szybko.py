
from tensorflow.keras.layers import LSTM,Dense,Conv2D,Flatten
# Funkcje do stworzenia różnych rodzajów sieci "na szybko" ograniczone do kilku najważniejszych parametrów 
def siec_rekurencyjna(x,warstwy,wyjscia,aktywacja_LSTM,aktywacja_rekurencji,drop,l2_ratio,out_act = 'linear'):

    if type(wyjscia)==int:
        wyjscia = [wyjscia for _ in range(warstwy)]
    if type(aktywacja_LSTM)==str:
        aktywacja_LSTM = [aktywacja_LSTM for _ in range(warstwy)]
    if type(aktywacja_rekurencji)==str:
        aktywacja_rekurencji = [aktywacja_rekurencji for i in range(warstwy)]
    
    for i in range(warstwy):
        x=BatchNormalization()(x)
        if i == warstwy-1:
            x = LSTM(units=wyjscia[i],activation=aktywacja_LSTM[i],recurrent_activation=aktywacja_rekurencji[i],return_sequences=False,dropout=drop,kernel_regularizer = l2(l2_ratio))(x)
        else:
            print(aktywacja_LSTM[i])
            x = LSTM(units=wyjscia[i],activation=aktywacja_LSTM[i],recurrent_activation=aktywacja_rekurencji[i],return_sequences=True,dropout=drop,kernel_regularizer = l2(l2_ratio))(x)
    out = Dense(units = 1,activation = out_act)(x)
    
    return out

def siec_konwolucyjna(x,warstwy,filtry,strides,kernel_size,aktywacja,pooling,drop,l2_ratio,pooling_type = 'global',out_act = 'linear'):

    if type(filtry)==int:
        filtry = [filtry for _ in range(warstwy)]
    if type(strides)==int:
        strides = [strides for _ in range(warstwy)]
    if type(aktywacja)!=list:
        aktywacja = [aktywacja for _ in range(warstwy)]
    if pooling_type!='global' and type(pooling)!=list:
        pooling = [pooling for _ in range(warstwy)]
    for i in range(warstwy):
        x = BatchNormalization()(x)
        x = Conv2D(filters=filtry[i],kernel_size = kernel_size,strides = strides[i],kernel_regularizer=l2(l2_ratio), padding = 'same')(x)
        x = aktywacja(x)
        if pooling_type!='global':
            x = pooling[i](x)
        x = Dropout(drop)(x)
    if pooling_type=='global':
        x = pooling(x)
    x = Flatten()(x)
    out = Dense(units = 1,activation = out_act)(x)
    
    return out

def MLP(x,warstwy,neurony,aktywacja):

    if type(neurony)==int:
        neurony = [neurony for _ in range(neurony)]
    if type(aktywacja)!=list:
        aktywacja = [aktywacja for _ in range(aktywacja)]

    for i in range(warstwy):
        
        x = Dense(units=neurony[i])(x)
    out = aktywacja(x)
    
    return out
