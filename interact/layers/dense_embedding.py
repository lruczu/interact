import tensorflow as tf
from tensorflow.python.keras import initializers, layers
from tensorflow.keras import regularizers

from interact.fields import DenseField


class DenseEmbedding(layers.Layer):
    """
    DenseField -> Tensor of shape (None, 1, d)
    """
    def __init__(
        self,
        dense_field: DenseField,
        l2_penalty: float = 0,
        **kwargs,
    ):
        self._dense_field = dense_field
        self._l2_penalty = l2_penalty

        self._v = None

        super(DenseEmbedding, self).__init__(**kwargs)

    def build(self, input_shape):
        if int(input_shape[1]) != 1:
            raise ValueError(f'Only one column is allowed for the dense field.')

        self._v = self.add_weight('v',
                                  shape=[1, self._dense_field.d],
                                  initializer=initializers.glorot_uniform(),
                                  regularizer=regularizers.l2(self._l2_penalty))
        super(DenseEmbedding, self).build(input_shape)

    def call(self, input):
       return tf.expand_dims(input * self._v, axis=1)
