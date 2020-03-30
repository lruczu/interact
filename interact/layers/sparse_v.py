import tensorflow as tf
from tensorflow.python.keras import layers

from interact.features import Interaction, InteractionType
from interact.layers import MaskEmbedding


class SparseV(layers.Layer):
    def __init__(
        self,
        interaction: Interaction,
        within: bool = True,
        **kwargs,
    ):
        self._interaction = interaction
        self._within = within
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
        within_interactions = []
        for index, i in enumerate(inputs):
            e = self._vs[index](i) * self._masks[index](i)
            if not self._within:
                rows_sum = tf.reduce_sum(e, axis=1) ** 2
                squared_rows_sum = tf.reduce_sum(e * e, axis=1)
                within_interactions.append(0.5 * tf.reduce_sum(rows_sum - squared_rows_sum, axis=1))
            es.append(e)
        if len(es) > 1:
            v = layers.Concatenate(axis=1)(es)
        else:
            v = es[0]
        rows_sum = tf.reduce_sum(v, axis=1) ** 2
        squared_rows_sum = tf.reduce_sum(v * v, axis=1)
        all_interactions = 0.5 * tf.reduce_sum(rows_sum - squared_rows_sum, axis=1)
        if self._within:
            return all_interactions

        return all_interactions - tf.math.add_n(within_interactions)
