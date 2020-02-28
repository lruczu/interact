from tensorflow.keras.layers import Add, Concatenate
from tensorflow.keras.models import Model

from interact.features import FeatureCollection
from interact.layers import AddBias, Linear
from interact.utils import prepare_dense_inputs, prepare_sparse_inputs


def FactorizationMachine(
    fc: FeatureCollection,
) -> Model:

    dense_fs, sparse_fs = fc.get_input_features()
    dense_inputs = prepare_dense_inputs(dense_fs)
    sparse_inputs = prepare_sparse_inputs(sparse_fs)

    dense_linear_combination = Linear()(Concatenate(dense_inputs))
    sparse_linear_combination = Linear()(Concatenate(sparse_inputs))  # to implement

    linear_dense_part = None
    linear_sparse_part = None

    linear_part = AddBias()(Concatenate()([linear_dense_part, linear_sparse_part]))
