from typing import List

from tensorflow.python.keras import layers

from interact.features import SparseFeature
from interact.layers import MaskEmbedding


class SparseV(layers.Layer):
    def __init__(
        self,
        sfs: List[SparseFeature],
        k: int,
        **kwargs,
    ):
        self._sfs = sfs
        self.k = k

        self._masks = []
        self._vs = []
        super(SparseV, self).__init__(**kwargs)

    def build(self, input_shapes):
        if len(self._sfs) != len(input_shapes):
            raise ValueError(f'Number of features must be equal to {len(self._sfs)}')
        for input_shape, sf in zip(input_shapes, self._sfs):
            if int(input_shape[1]) != sf.m:
                raise ValueError(f'Number of columns in '
                                 f'the input must be the following: {[sf.m for sf in self._sfs]}')

        for sf in self._sfs:
            self._vs.append(
                layers.Embedding(
                    input_dim=sf.vocabulary_size + 1,
                    output_dim=self.k,
                    input_length=sf.m,
                )
            )
            self._masks.append(
                MaskEmbedding(
                    input_dim=sf.vocabulary_size,
                    output_dim=self.k,
                    input_length=sf.m,
                )
            )

    def call(self, inputs):
        es = []
        for index, i in enumerate(inputs):
            e = self._vs[index](i) * self._masks[index](i)
            es.append(e)
        return layers.Concatenate(axis=1)(es)
