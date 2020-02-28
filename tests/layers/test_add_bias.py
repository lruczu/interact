import numpy as np
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

from interact.layers import AddBias


def test_no_fail():
    BIAS = 18.9
    i = Input(shape=1)
    add_bias = AddBias()
    o = add_bias(i)
    add_bias.set_weights([np.array([BIAS])])
    model = Model(i, o)

    i = [
        [1],
        [2],
        [1.1]
    ]
    expected_o = [
        [1 + BIAS],
        [2 + BIAS],
        [1.1 + BIAS]
    ]
    o = model.predict(i)
    assert np.all(np.isclose(o, expected_o))
