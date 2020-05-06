import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Concatenate, Input
from tensorflow.keras.models import Model

from interact.features import DenseFeature, Interaction
from interact.layers import FM, V


def test_v():
    i1 = Input(shape=1, dtype=tf.float32)
    i2 = Input(shape=1, dtype=tf.float32)
    i3 = Input(shape=1, dtype=tf.float32)
    k = 20
    weights = np.random.uniform(size=(3, k))
    interaction = Interaction([DenseFeature('df1'), DenseFeature('df2'), DenseFeature('df3')], k=k)
    v = V(interaction)
    fm = FM(interaction)

    e = v([i1, i2, i3])
    products = fm([e])

    m = Model([i1, i2, i3], products)
    _, eye = m.get_weights()
    m.set_weights([weights, eye])

    v1, v2, v3 = 1.3, 7.2, 9.9
    i = np.array([
        [0, 0, 0],
        [0, 0, v3],
        [0, v2, 0],
        [0, v2, v3],
        [v1, 0, 0],
        [v1, 0, v3],
        [v1, v2, 0],
        [v1, v2, v3]
    ])
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
    o = m.predict([i[:, 0], i[:, 1], i[:, 2]])
    assert np.all(np.isclose(o, expected_o, atol=1e-04))
