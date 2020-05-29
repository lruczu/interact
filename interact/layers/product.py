import tensorflow as tf
from tensorflow.python.keras import initializers
from tensorflow.python.keras import layers


class Product(layers.Layer):
    """
        (None, m, k) -> (None, m, 1)
        where each observation, of shape (m, k) is treated as a dataset
        which is transformed by applying dot product on each row using vector 'h'.
    """

    def __init__(
        self, 
        dim1: int, 
        dim2: int = 1,
        **kwargs,
    ):
        self._dim1 = dim1
        self._dim2 = dim2

        self._W = None
        super(Product, self).__init__(**kwargs)

    def build(self, input_shape: tf.TensorShape):
        if not isinstance(input_shape, list):
            input_shapes = [input_shape]
        else:
            input_shapes = input_shape

        for shape in input_shapes:
            if int(shape[-1]) != self._dim1:
                raise ValueError(f'Last dimension of input shape must be equal to {self._dim1}.')

        self._W = self.add_weight('W',
            shape=[self._dim1, self._dim2],
            initializer=initializers.glorot_uniform(),
        )

    def call(self, inputs):
        if isinstance(inputs, list):
            return [tf.matmul(i, self._W) for i in inputs]
        return tf.matmul(inputs, self._W)
