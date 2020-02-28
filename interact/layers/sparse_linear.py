import tensorflow as tf
from tensorflow.python.keras import layers


class SparseLinear(layers.Layer):
    """Linear combination of sparse columns.

        input: [[1 2 100]]
        output: [w1 + w2 + w100]
    """
    def __init__(self, vocabulary_size: int, **kwargs):
        self.vocabulary_size = vocabulary_size
        self.linear_weights = None
        super(SparseLinear, self).__init__(**kwargs)

    def build(self, input_shape: tf.TensorShape):
        if input_shape.ndims != 2:
            raise ValueError('Unexpected input shape.')

        m = int(input_shape[1])
        self.linear_weights = layers.Embedding(
            self.vocabulary_size,
            1,
            input_length=m
        )

    def call(self, inputs):
        if inputs.shape.ndims != 2:
            raise ValueError('Unexpected input shape.')
        return tf.reduce_sum(self.linear_weights(inputs - 1), axis=1)
