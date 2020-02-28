import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Concatenate, Input
from tensorflow.keras.models import Model

from interact.layers import V


def test_v():
    i1 = Input(shape=1, dtype=tf.int32)
    i2 = Input(shape=1, dtype=tf.int32)
    i3 = Input(shape=1, dtype=tf.int32)
    i = Concatenate()([i1, i2, i3])
    k = 20
    weights = np.random.uniform(size=(3, k))

    v = V(k)
    o = v(i)
    o.set_weights([weights])
    m = Model(i, o)

    v1, v2, v3 = 1.3, 7.2, 9.9
    i = [
        [0, 0, 0],
        [0, 0, v3],
        [0, v2, 0],
        [0, v2, v3],
        [v1, 0, 0],
        [v1, 0, v3],
        [v1, v2, 0],
        [v1, v2, v3]
    ]
    expected_o = [
        [0],
        [0],
        [0],
        [v2 * v3 * np.sum(weights[1] * weights[2])],
        [0],
        [v1 * v3 * np.sum(weights[0] * weights[2])],
        [v1 * v2 * np.sum(weights[0] * weights[1])],
        [
            v1 * v2 * np.sum(weights[0] * weights[1]) +
            v2 * v3 * np.sum(weights[1] * weights[2]) +
            v1 * v3 * np.sum(weights[0] * weights[2])
        ],
    ]
    o = m.predict(i)
    assert np.all(np.isclose(o, expected_o))
