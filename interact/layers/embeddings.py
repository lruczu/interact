import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.eager import context
from tensorflow.python.keras import layers
from tensorflow.python.keras.utils import tf_utils


"""
class MaskV(layers.Embedding):
    @tf_utils.shape_type_conversion
    def build(self, input_shape):
        if context.executing_eagerly() and context.context().num_gpus():
            with ops.device('cpu:0'):
                self.embeddings = self.add_weight(
                    shape=(self.input_dim + 1, self.output_dim),
                    initializer='ones',
                    name='embeddings',
                    trainable=False)
                self._update_embeddings()
        else:
            self.embeddings = self.add_weight(
                  shape=(self.input_dim + 1, self.output_dim),
                  initializer='ones',
                  name='embeddings',
                  trainable=False)
            self._update_embeddings()
            
        self.built = True
        
    def _update_embeddings(self):
        e = np.ones(shape=(self.input_dim + 1, self.output_dim))
        e[0] = 0
        self.set_weights([e])


class CustomEmbedding(layers.Layer):
    def build(self, input_shape):
        assert True
        self.mask_embedding = MaskEmbedding(input_shape)
        self.embedding = layers.Embedding(input_shape)

    def call(self, inputs):
        return self.mask_embedding(inputs) * self.embedding(inputs)
"""


class V(layers.Layer):
    """
    Returns matrix V for sparse/categorical features.

    Suppose we have a 100-dimensional sparse/categorical feature
    and we want to encode these features as 50-dimensional vectors.
    To simplify notation let by ei denote embedding of a i-th feature
    and by z a vector filled with zeros.
    Consider two cases:

    1. Input is a categorical feature, which means only at one place we have 1, the rest of
    the vector is filled with zeros. In this case we should have the following input:
        [[1],
        [45],
        [100]]
        The output will be
            [[z, e1]]
            [[z, e45]]
            [[z, e100]]
    2. Input is a categorical feature, but this time the number of nonzero elements is
     (can be bigger than one). It is user's responsibility to determine maximum number of
     nonzero features. Suppose we set this number to 5. Then the expected input is the following:
        [1, 45, 0, 0, 0]
        [0, 100, 0, 0, 0]
        [0, 0, 0, 0, 0],
        [10, 20, 98, 99, 100]
        The output will be
            [[e1, e45, z, z, z]]
            [[z, e100, z, z, z]]
            [[z, z, z, z, z]]
            [[e10, e20, e98, e99, e100]]
    """
    def __init__(
        self,
        input_dim,
        output_dim,
        input_length,
        **kwargs,
    ):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_length = input_length

        self.mask_v = None
        self.v = None
        super(V, self).__init__(**kwargs)

    def build(self, input_shape):
        self.mask_v = MaskV(input_dim=self.input_dim,
                            output_dim=self.output_dim,
                            input_length=self.input_length)
        self.v = layers.Embedding(
            self.input_dim,
            self.output_dim,
            input_length=self.input_length)
        
        #super(CustomEmbedding, self).build(input_shape)  # Be sure to call this somewhere!
    
    #def build(self, input_shape):
    #    self.build = True
    #    assert True
    #    self.mask_embedding = MaskEmbedding(input_shape)
    #    self.embedding = layers.Embedding(input_shape)

    def call(self, inputs):
        return self.mask_v(inputs) * self.v(inputs)
