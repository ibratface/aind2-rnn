import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import keras


# TODO: fill out the function below that transforms the input series
# and window-size into a set of input/output pairs for use with our RNN model
def window_transform_series(series,window_size):
    # containers for input/output pairs
    X = []
    y = []

    for i in range(window_size, len(series)):
        X.append(series[i-window_size:i])
        y.append(series[i])

    # reshape each
    X = np.asarray(X)
    X.shape = (np.shape(X)[0:2])
    y = np.asarray(y)
    y.shape = (len(y),1)

    return X,y

# TODO: build an RNN to perform regression on our time series input/output data
def build_part1_RNN(step_size, window_size):
    model = Sequential([
        LSTM(5, input_shape=(window_size, 1)),
        Dense(1),
    ])

    # build model using keras documentation recommended optimizer initialization
    optimizer = keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

    # compile the model
    model.compile(loss='mean_squared_error', optimizer=optimizer)


### TODO: list all unique characters in the text and remove any non-english ones
def clean_text(text):
    # find all unique characters in the text
    uniques = set()
    for c in text:
        uniques.add(c)
    # print (uniques)

    # remove as many non-english characters and character sequences as you can
    alphas = [ chr(c) for c in range(ord('a'), ord('z') + 1) ]
    digits = [ chr(c) for c in range(ord('0'), ord('9') + 1) ]
    puncts = "'-,.? &\"/%$!:();"
    noneng = uniques - set(alphas) - set(digits) - set(puncts)
    # print (noneng)
    for c in noneng:
        text = text.replace(c, ' ')

    # shorten any extra dead space created above
    text = text.replace('  ',' ')


### TODO: fill out the function below that transforms the input text and window-size into a set of input/output pairs for use with our RNN model
def window_transform_text(text,window_size,step_size):
    # containers for input/output pairs
    inputs = []
    outputs = []

    for i in range(window_size, len(text), step_size):
        inputs.append(text[i-window_size:i])
        outputs.append(text[i])

    return inputs,outputs
