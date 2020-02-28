from typing import List

from tensorflow.keras.layers import Input

from interact.features import DenseFeature, SparseFeature


def prepare_dense_inputs(dense_fs: List[DenseFeature]) -> List[Input]:
    return [Input(shape=1) for d_f in dense_fs]


def prepare_sparse_inputs(sparse_fs: List[SparseFeature]) -> List[Input]:
    return [Input(shape=s_f.n_levels) for s_f in sparse_fs]
