import numpy as np
import tensorflow as tf
from tensorflow.python.keras import layers, initializers

from interact.features import DenseFeature, Interaction, InteractionType, SparseFeature
from interact.layers import MaskEmbedding


class MixedV(layers.Layer):
    def __init__(
        self,
        interaction: Interaction,
        within: bool = True,
        **kwargs,
    ):
        self._interaction = interaction
        self._within = within
        if self._interaction.interaction_type != InteractionType.MIXED:
            raise ValueError('Only mixed interaction type is allowed.')
        self._sparse_vs = []
        self._masks = []
        self._v = None
        self._eye = None
        super(MixedV, self).__init__(**kwargs)

    def build(self, input_shapes):
        if self._interaction.n_features != len(input_shapes):
            raise ValueError(f'Number of features must be equal to {self._interaction.n_features}')

        for f, input_shape in zip(self._interaction.features, input_shapes):
            if isinstance(f, DenseFeature):
                if int(input_shape[1]) != 1:
                    raise ValueError(f'Wrong number of columns for {f.get_name()}')
            else:
                if int(input_shape[1]) != f.m:
                    raise ValueError(f'Wrong number of columns for {f.get_name()}')

        dense_features = 0
        for f in self._interaction.features:
            if isinstance(f, DenseFeature):
                dense_features += 1
            elif isinstance(f, SparseFeature):
                self._sparse_vs.append(
                    layers.Embedding(
                        input_dim=f.vocabulary_size + 1,
                        output_dim=self._interaction.k,
                        input_length=f.m,
                    )
                )
                self._masks.append(
                    MaskEmbedding(
                        input_dim=f.vocabulary_size,
                        output_dim=self._interaction.k,
                        input_length=f.m,
                    )
                )
            else:
                raise ValueError('Wrong type of feature.')

        self._v = self.add_weight('v',
                                  shape=[dense_features, self._interaction.k],
                                  initializer=initializers.glorot_uniform())
        self._eye = self.add_weight('eye',
                                    shape=[dense_features, dense_features],
                                    initializer='zeros',
                                    trainable=False)

        self._update_weights()

    def call(self, inputs):
        es = []
        dense_inputs = []
        within_interactions = []
        sparse_index = 0
        for i, f in zip(inputs, self._interaction.features):
            if isinstance(f, DenseFeature):
                dense_inputs.append(i)
            elif isinstance(f, SparseFeature):
                e = self._sparse_vs[sparse_index](i) * self._masks[sparse_index](i)
                es.append(e)
                sparse_index += 1
                if not self._within:
                    rows_sum = tf.reduce_sum(e, axis=1) ** 2
                    squared_rows_sum = tf.reduce_sum(e * e, axis=1)
                    within_interactions.append(0.5 * tf.reduce_sum(rows_sum - squared_rows_sum, axis=1))
            else:
                raise ValueError('Wrong type of input.')

        if len(es) > 1:
            v_sparse = layers.Concatenate(axis=1)(es)
        else:
            v_sparse = es[0]

        if len(dense_inputs) > 1:
            dense_inputs = layers.Concatenate()(dense_inputs)
        else:
            dense_inputs = dense_inputs[0]

        dense_inputs = tf.expand_dims(dense_inputs, axis=-1) * tf.expand_dims(self._eye, axis=0)

        v_dense = tf.matmul(dense_inputs, self._v)
        global_v = layers.Concatenate(axis=1)([v_sparse, v_dense])
        rows_sum = tf.reduce_sum(global_v, axis=1) ** 2
        squared_rows_sum = tf.reduce_sum(global_v * global_v, axis=1)

        all_interactions = 0.5 * tf.reduce_sum(rows_sum - squared_rows_sum, axis=1)
        if self._within:
            return all_interactions
        return all_interactions - tf.math.add_n(within_interactions)

    def _update_weights(self):
        ws = self.get_weights()
        ws[-1] = np.eye(ws[-1].shape[0])
        self.set_weights(ws)
