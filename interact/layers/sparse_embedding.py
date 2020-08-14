import tensorflow as tf
from tensorflow.python.keras import layers
from tensorflow.keras import regularizers

#from interact.fields import SparseField
from interact.layers import MaskEmbedding


class SparseEmbedding(layers.Layer):
    """
    It operates in two modes:
        if averaged:
            SparseField -> Tensor of shape (None, 1, d). Each row is equal to the average of embedding vectors
        else:
            SparseField -> Tensor of shape (None, m, d). Each row is a matrix of m embedding vectors, some
            of which might be zeros.
    """
    def __init__(
        self,
        sparse_field,
        averaged: bool = False,
        l1_penalty: float = 0,
        l2_penalty: float = 0,
        flatten: bool = False,
        **kwargs,
    ):
        self._sparse_field = sparse_field
        self._averaged = averaged
        self._l1_penalty = l1_penalty
        self._l2_penalty = l2_penalty
        self._flatten = flatten

        self._v = None
        self._mask = None

        super(SparseEmbedding, self).__init__(**kwargs)

    def build(self, input_shape):
        if int(input_shape[1]) != self._sparse_field.m:
            raise ValueError(f'Wrong number of columns for {self._sparse_field.get_name()}')

        self._v = layers.Embedding(
                input_dim=self._sparse_field.vocabulary_size + 1,
                output_dim=self._sparse_field.d,
                input_length=self._sparse_field.m,
                embeddings_regularizer=regularizers.l1_l2(l1=self._l1_penalty, l2=self._l2_penalty)
            )
        self._mask = MaskEmbedding(
                input_dim=self._sparse_field.vocabulary_size,
                output_dim=self._sparse_field.d,
                input_length=self._sparse_field.m,
            )

        super(SparseEmbedding, self).build(input_shape)

    def call(self, input):
        e = self._v(input) *  self._mask(input)
        if self._averaged:
            # (None, # of embedding vectors, 1)
            # zero embeddings will have 0 number of nonzero components
            zero_embedding_flag = tf.math.reduce_any(e > 0, axis=2, keepdims=True)

            n_nonzero_embeddings = tf.math.count_nonzero(zero_embedding_flag, axis=1, keepdims=True)
            n_nonzero_embeddings_with_default_for_zero = tf.nn.relu(n_nonzero_embeddings - 1) + 1

            return tf.reduce_sum(e, axis=1, keepdims=True) / tf.cast(n_nonzero_embeddings_with_default_for_zero, tf.float32)

            # n_nonzero_embeddings = tf.math.reduce_sum(
            #     n_nonzeros_per_embedding_vector > 0, axis=1, keepdims=True)


            # #n_nonzeros = tf.math.count_nonzero(input, axis=1, keepdims=True)
            # n_nonzeros_with_default_for_zero = tf.nn.relu(n_nonzero_embeddings - 1) + 1
            # return tf.reduce_sum(e, axis=1, keepdims=True) / n_nonzeros_with_default_for_zero
        return e
