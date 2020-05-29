import tensorflow as tf
from tensorflow.python.keras import layers
from tensorflow.python.keras import initializers

from interact.features import DenseFeature, Interaction, InteractionType, SparseFeature


class FM(layers.Layer):
    """
        Takes a list of embeddings 
    """
    def __init__(
        self,
        interaction: Interaction,
        **kwargs,
    ):
        self._interaction = interaction
        super(FM, self).__init__(**kwargs)

    def build(self, input_shapes):
        k = None
        for input_shape in input_shapes:
            if k is None:
                k = int(input_shape[2])
            else:
                if k != int(input_shape[2]):
                    raise ValueError("Embedding dimensions must be the same.")

    def call(self, vs):
        if len(vs) > 1:
            v = layers.Concatenate(axis=1)(vs)
        else:
            v = vs[0]
        rows_sum = tf.reduce_sum(v, axis=1) ** 2
        squared_rows_sum = tf.reduce_sum(v * v, axis=1)
        all_interactions = 0.5 * tf.reduce_sum(rows_sum - squared_rows_sum, axis=1, keepdims=True)
        return all_interactions
