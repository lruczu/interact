import numpy as np
import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.python.keras import layers
from tensorflow.python.keras import initializers

from interact.features import Interaction, InteractionType


class V(layers.Layer):
    """
    Expects a list of embeddings from dense features.
    """

    def __init__(
        self,
        interaction: Interaction,
        l2_penalty: int,
        **kwargs,
    ):
        self._interaction = interaction
        if self._interaction.interaction_type != InteractionType.DENSE:
            raise ValueError('Only dense interaction type is allowed.')
        self._v = None
        self._eye = None
        super(V, self).__init__(**kwargs)

    def build(self, input_shapes):
        if self._interaction.n_features != len(input_shapes):
            raise ValueError(f'Number of features must be equal to {self._interaction.n_features}')

        for input_shape, df in zip(input_shapes, self._interaction.features):
            if int(input_shape[1]) != 1:
                raise ValueError(f'Wrong number of columns for {df.get_name()}')

        self._v = self.add_weight('v',
                                  shape=[len(input_shapes), self._interaction.k],
                                  initializer=initializers.glorot_uniform(),
                                  regularizer=regularizers.l2(self._l2_penalty))
        self._eye = self.add_weight('eye',
                                    shape=[len(input_shapes), len(input_shapes)],
                                    initializer='zeros',
                                    trainable=False)
        self._update_weights()

    def call(self, es):
        if len(inputs) > 1:
            dense_inputs = layers.Concatenate()(es)
        else:
            dense_inputs = inputs[0]
        scaling_factor = tf.expand_dims(dense_inputs, axis=-1) * tf.expand_dims(self._eye, axis=0)
        return tf.matmul(scaling_factor, self._v)

    def _update_weights(self):
        ws = self.get_weights()
        ws[-1] = np.eye(ws[-1].shape[0])
        self.set_weights(ws)
