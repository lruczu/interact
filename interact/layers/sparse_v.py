import tensorflow as tf
from tensorflow.python.keras import layers
from tensorflow.keras import regularizers

from interact.features import Interaction, InteractionType
from interact.layers import MaskEmbedding


class SparseV(layers.Layer):
    """
    Expects a list of embeddings from sparse features.
    """

    def __init__(
        self,
        interaction: Interaction,
        l2_penalty: float = 0,
        **kwargs,
    ):
        self._interaction = interaction
        self._l2_penalty = l2_penalty
        if self._interaction.interaction_type != InteractionType.SPARSE:
            raise ValueError('Only sparse interaction type is allowed.')

        self._masks = []
        self._vs = []
        super(SparseV, self).__init__(**kwargs)

    def build(self, input_shapes):
        if self._interaction.n_features != len(input_shapes):
            raise ValueError(f'Number of features must be equal to {self._interaction.n_features}')
        for input_shape, sf in zip(input_shapes, self._interaction.features):
            if int(input_shape[1]) != sf.m:
                raise ValueError(f'Wrong number of columns for {sf.get_name()}')

        for sf in self._interaction.features:
            self._vs.append(
                layers.Embedding(
                    input_dim=sf.vocabulary_size + 1,
                    output_dim=self._interaction.k,
                    input_length=sf.m,
                    embeddings_regularizer=regularizers.l2(self._l2_penalty)
                )
            )
            self._masks.append(
                MaskEmbedding(
                    input_dim=sf.vocabulary_size,
                    output_dim=self._interaction.k,
                    input_length=sf.m,
                )
            )

    def call(self, inputs):
        es = []
        for index, i in enumerate(inputs):
            e = self._vs[index](i) * self._masks[index](i)
            es.append(e)
        return es
        if len(es) > 1:
            v = layers.Concatenate(axis=1)(es)
        else:
            v = es[0]
        rows_sum = tf.reduce_sum(v, axis=1) ** 2
        squared_rows_sum = tf.reduce_sum(v * v, axis=1)
        return 0.5 * tf.reduce_sum(rows_sum - squared_rows_sum, axis=1)
