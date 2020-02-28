import tensorflow as tf
from tensorflow.python.keras import layers
from tensorflow.python.keras import initializers


class V(layers.Layer):
    def __init__(self, k: int, **kwargs):
        self.k = k
        self.v = None
        super(V, self).__init__(**kwargs)

    def build(self, input_shapes):
        if len(input_shapes) <= 1:
            raise ValueError('At least two dense features are required.')
        for input_shape in input_shapes:
            if int(input_shape[1]) != 1:
                raise ValueError('Dense features must be represented by one column.')
        n_features = len(input_shapes)
        self.v = self.add_weight('v',
                                 shape=[n_features, self.k],
                                 initializer=initializers.glorot_uniform())

    def call(self, inputs):
        rows_sum = tf.matmul(inputs, self.v) ** 2
        squaured_rows_sum = tf.matmul(inputs * inputs, self.v * self.v)

        return 0.5 * tf.reduce_sum(rows_sum - squaured_rows_sum, axis=1, keepdims=True)
