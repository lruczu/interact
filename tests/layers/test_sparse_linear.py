import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

from interact.layers import SparseLinear


def test_sparse_linear():
    i = Input(shape=3, dtype=tf.int32)
    sparse_linear = SparseLinear(vocabulary_size=100)
    o = sparse_linear(i)
    ws = np.random.uniform(size=101)
    _, mask = sparse_linear.get_weights()
    sparse_linear.set_weights([ws.reshape(-1, 1), mask])
    m = Model(i, o)

    i = [
        [1, 2, 10],
        [2, 2, 10],  # valid row but makes no sense
        [1, 50, 100]
    ]
    expected_o = [
        [ws[1] + ws[2] + ws[10]],
        [ws[2] + ws[2] + ws[10]],
        [ws[1] + ws[50] + ws[100]]
    ]
    o = m.predict(i)
    assert np.all(np.isclose(o, expected_o))
