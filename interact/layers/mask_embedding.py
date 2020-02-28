import numpy as np
from tensorflow.python.framework import ops
from tensorflow.python.eager import context
from tensorflow.python.keras import layers
from tensorflow.python.keras.utils import tf_utils


class MaskEmbedding(layers.Embedding):
    """Mask positions corresponding to zero vector.

    Suppose that we have a number of levels 3 (sequence length) and
    a vocabulary size is equal to 10. Let embedding dimension be 5.
    Then,

    input: [[1, 2, 3]]
    output: [
        [1, 1, 1, 1, 1],
        [1, 1, 1 ,1 ,1],
        [1, 1, 1, 1, 1]
    ]

    input: [[1, 0, 1]]
    output: [
        [1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1]
    ]

    input: [[0, 0, 0]]
    output: [
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0]
    ]
    """
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

    def _update_embeddings(self):
        e = np.ones(shape=(self.input_dim + 1, self.output_dim))
        e[0] = 0
        self.set_weights([e])
