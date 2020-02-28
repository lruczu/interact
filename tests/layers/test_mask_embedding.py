import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

from interact.layers import MaskEmbedding


def test_mask_embedding():
    i = Input(shape=3, dtype=tf.int32)
    o = MaskEmbedding(input_dim=10, output_dim=5)(i)
    m = Model(i, o)

    i1 = [[1, 2, 10]]
    expected_o1 = [
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1]
    ]
    o1 = m.predict(i1)
    assert o1 == expected_o1

    i2 = [[1, 0, 1]]
    expected_o2 = [
        [1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1]
    ]
    o2 = m.predict(i2)
    assert o2 == expected_o2

    i3 = [[0, 0, 0]]
    expected_o3 = [
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0]
    ]
    o3 = m.predict(i3)
    assert o3 == expected_o3
