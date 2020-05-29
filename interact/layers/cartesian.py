import itertools

import tensorflow as tf
from tensorflow.python.keras import initializers
from tensorflow.python.keras import layers


class Cartesian(layers.Layer):
    """
        List of Tensors [(None, d), (None, d), .., (None, d)] of length n
        -> Tensor of shape (None, m, d) where m = (n 2), i.e. all possible combinations.
        Each observation is element-wise multiplication of two vectors.
    """

    def __init__(self, **kwargs):
        super(Cartesian, self).__init__(**kwargs)


    def build(self, input_shapes: tf.TensorShape):
        d = None
        for input_shape in input_shapes:
            if input_shape.ndims != 2:
                raise ValueError('Embeddings must have 2 dimensions.')
            if d is None:
                d = int(input_shape[1])
            else:
                if d != int(input_shape[1]):
                    raise ValueError('All embeddings must have the same last dimension.')

        super(Cartesian, self).build(input_shapes)

    def call(self, inputs):
        combinations = []
        for v1, v2 in itertools.combinations(inputs, 2):
            combinations.append(v1 * v2)
        return tf.stack(combinations, axis=1)
