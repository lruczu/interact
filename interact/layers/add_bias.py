import tensorflow as tf
from tensorflow.python.keras import initializers
from tensorflow.python.keras import layers


class AddBias(layers.Layer):
    """
    Add single number to tensor.
    """
    def __init__(self, **kwargs):
        self.bias = None
        super(AddBias, self).__init__(**kwargs)

    def build(self, input_shape: tf.TensorShape):
        if input_shape.ndims != 2 or int(input_shape[1]) != 1:
            raise ValueError('Unexpected input shape.')
        self.bias = self.add_weight('bias',
                                    shape=1,
                                    initializer=initializers.zeros())

    def call(self, inputs):
        if inputs.shape.ndims != 2 or int(inputs.shape[1]) != 1:
            raise ValueError('Unexpected input shape.')
        return inputs + self.bias
