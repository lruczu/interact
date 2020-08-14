import itertools

import tensorflow as tf
from tensorflow.python.keras import activations
from tensorflow.python.keras import initializers
from tensorflow.python.keras import layers


class AttentionAFM(layers.Layer):
    """
        (None, m, k) -> (None, m, t)
        where each observation, of shape (m, k) is treated as a dataset
        which is transformed by a 1-layer neural net whose output is of dimension t, 
        called attention factor.
    """

    def __init__(
        self, 
        t: int,
        activation = activations.relu,
        **kwargs,
    ):
        self._t = t
        self._activation = activation

        self._w = None
        self._b = None

        super(AttentionAFM, self).__init__(**kwargs)


    def build(self, input_shape: tf.TensorShape):
        if input_shape.ndims != 3:
            raise ValueError('Input shape must be equal to 3.')
        
        self._w = self.add_weight('w',
            shape=[int(input_shape[2]), self._t], # W = (k, t)
            initializer=initializers.glorot_uniform(),
        )
        self._b = self.add_weight('b',
            shape=[1, self._t],
            initializer=initializers.zeros(),
        )

        super(AttentionAFM, self).build(input_shape)

    def call(self, inputs):
        return self._activation(tf.matmul(inputs, self._w) + self._b)
