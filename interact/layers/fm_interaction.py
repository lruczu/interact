from typing import List

import tensorflow as tf
from tensorflow.python.keras import layers


class FMInteraction(layers.Layer):
    """
    List of embeddings -> Two-order interactions

    In terms of dimensions:
    [
        (None, 1, d), (None, 1, d), (None, m, d), ...
    ] -> (None, 1)
    """

    def __init__(self, **kwargs):
        super(FMInteraction, self).__init__(**kwargs)

    def build(self, input_shapes: List[tf.TensorShape]):
        d = None
        for input_shape in input_shapes:
            if input_shape.ndims != 3:
                raise ValueError('Embeddings must have 3 dimensions.')
            if d is None:
                d = int(input_shape[2])
            else:
                if d != int(input_shape[2]):
                    raise ValueError('All embeddings must have the same last dimension.')

        super(FMInteraction, self).build(input_shapes)

    def call(self, inputs: List[tf.Tensor]):
        if len(inputs) > 1:
            i = layers.Concatenate(axis=1)(inputs)
        else:
            i = inputs[0]
            
        rows_sum = tf.reduce_sum(i, axis=1)
        squared_rows_sum = tf.reduce_sum(i * i, axis=1)

        return 0.5 * tf.reduce_sum(rows_sum ** 2 - squared_rows_sum, axis=1, keepdims=True)
