import itertools

import tensorflow as tf
from tensorflow.python.keras import activations
from tensorflow.python.keras import initializers
from tensorflow.python.keras import layers

from interact.layers import Product


class AttentionAutoInt(layers.Layer):
    """
    
    """
    def __init__(
        self,
        d: int,
        d_prime: int,
        **kwargs,
    ):
        self._d = d
        self._d_prime = d_prime
        
        self._W_query = None
        self._W_key = None
        self._W_value = None

        super(AttentionAutoInt, self).__init__(**kwargs)

    def build(self, input_shapes: tf.TensorShape):
        self.x = input_shapes
        for input_shape in input_shapes:
            if input_shape.ndims != 2:
                raise ValueError('Input shape must be equal to 2.')
            if int(input_shape[-1]) != self._d:
                    raise ValueError(f'Last dimension of input shape must be equal to {self._d}.')

        self._W_query = Product(self._d, self._d_prime)
        self._W_key = Product(self._d, self._d_prime)
        self._W_value = Product(self._d, self._d_prime)
        self._W_res = Product(self._d, self._d_prime)

        super(AttentionAutoInt, self).build(input_shapes)

    def call(self, inputs):
        e_query = self._W_query(inputs)
        e_key = self._W_key(inputs)
        e_value = self._W_value(inputs)
        e_res = self._W_res(inputs)

        # (None, m, d_prime)
        e_query_concat = layers.Concatenate(axis=1)([tf.expand_dims(eq, axis=1) for eq in e_query])

        # (None, m, d_prime)
        e_key_concat = layers.Concatenate(axis=1)([tf.expand_dims(ek, axis=1) for ek in e_key])

        # (None, m, d_prime)
        e_value_concat = layers.Concatenate(axis=1)([tf.expand_dims(ev, axis=1) for ev in e_value])

        # (None, m, d_prime)
        e_res_concat = layers.Concatenate(axis=1)([tf.expand_dims(er, axis=1) for er in e_res])
        
        # (None, m, m)
        alphas = tf.nn.softmax(tf.matmul(e_query_concat, e_key_concat, transpose_b=True), axis=2)

        # (None, m, d_prime)
        new_e  = tf.matmul(alphas, e_value_concat)

        return tf.nn.relu(e_res_concat + new_e)
