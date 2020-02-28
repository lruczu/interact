from tensorflow.keras.layers import Add, Concatenate
from tensorflow.keras.models import Model

from interact.features import FeatureCollection
from interact.layers import AddBias, Linear, SparseLinear, V
from interact.utils import prepare_dense_inputs, prepare_sparse_inputs


def FactorizationMachine(
    fc: FeatureCollection,
    k: int,
) -> Model:
    dense_fs = fc.get_dense_features()
    sparse_fs = fc.get_sparse_features()

    dense_inputs = prepare_dense_inputs(dense_fs)
    sparse_inputs = prepare_sparse_inputs(sparse_fs)

    dense_linear_combination = Linear()(
        Concatenate()(
            dense_inputs
        )
    )

    linear_output = AddBias()(dense_linear_combination)
    two_way_interactions = V(k)(dense_inputs)

    o = linear_output + two_way_interactions

    return Model(dense_inputs, o)

# each feature in interaction brings some V matrix
# feature can occur a couple of times in feature, each such occurrence brings new V!!!
# (all dense features are concatenated and V is produced)
#
#
#