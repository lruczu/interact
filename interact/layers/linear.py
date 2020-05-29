import tensorflow as tf
from tensorflow.python.keras import initializers
from tensorflow.python.keras import layers


class Linear(layers.Layer):
    """Linear combination of dense columns.

        input: [[x1, x2, .., xn]]
        output: [[w1 * x1 + w2 * x2 + .. wn * xn]],
        where ws are trainable weights of the layer (bias term is not included).

        In terms of shapes: (None, N) -> (None, 1)
    """
    def __init__(self, **kwargs):
        self._linear_weights = None
        super(Linear, self).__init__(**kwargs)

    def build(self, input_shape: tf.TensorShape):
        if input_shape.ndims != 2:
            raise ValueError('Input shape must be equal to 2.')

        n_features = int(input_shape[1])
        self._linear_weights = self.add_weight('linear_weights',
                                              shape=[1, n_features],
                                              initializer=initializers.glorot_uniform())

    def call(self, inputs):
        if inputs.shape.ndims != 2:
            raise ValueError('Unexpected input shape.')
        return tf.reduce_sum(inputs * self._linear_weights, axis=1, keepdims=True)
