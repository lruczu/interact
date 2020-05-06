import tensorflow as tf
from tensorflow.python.keras import layers
from tensorflow.python.keras import initializers

from interact.features import Interaction, InteractionType


class V(layers.Layer):
    def __init__(
        self,
        interaction: Interaction,
        **kwargs,
    ):
        self._interaction = interaction
        if self._interaction.interaction_type != InteractionType.DENSE:
            raise ValueError('Only dense interaction type is allowed.')
        self._v = None
        super(V, self).__init__(**kwargs)

    def build(self, input_shapes):
        if self._interaction.n_features != len(input_shapes):
            raise ValueError(f'Number of features must be equal to {self._interaction.n_features}')

        for input_shape, df in zip(input_shapes, self._interaction.features):
            if int(input_shape[1]) != 1:
                raise ValueError(f'Wrong number of columns for {df.get_name()}')

        self._v = self.add_weight('v',
                                  shape=[len(input_shapes), self._interaction.k],
                                  initializer=initializers.glorot_uniform())

    def call(self, inputs):
        i = layers.Concatenate()(inputs)
        rows_sum = tf.matmul(i, self._v) ** 2
        squared_rows_sum = tf.matmul(i * i, self._v * self._v)

        return 0.5 * tf.reduce_sum(rows_sum - squared_rows_sum, axis=1, keepdims=True)
