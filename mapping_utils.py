"""
Utility methods for alternative mapping methods.

Currently implemented:
- MLP: using a multi-layer perceptron (one linear layer works best)
"""

from cupy_utils import get_cupy

import tensorflow as tf

from keras.models import Sequential
from keras.layers import Dense
from keras import backend as K


def mlp(x, z, src_indices, trg_indices, cuda):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    K.set_session(session)
    model = Sequential([
        Dense(z.shape[1], activation=None, input_dim=x.shape[1])
    ])
    model.compile(optimizer='adam', loss='mse')
    if cuda:
        xp = get_cupy()
        x = xp.asnumpy(x)
        z = xp.asnumpy(z)
        src_indices = xp.asnumpy(src_indices)
        trg_indices = xp.asnumpy(trg_indices)
    model.fit(x[src_indices], z[trg_indices], epochs=10, batch_size=128)
    xw = model.predict(x)
    zw = z
    if cuda:
        xp = get_cupy()
        xw = xp.asarray(xw)
        zw = xp.asarray(zw)
    return xw, zw
