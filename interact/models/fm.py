from typing import List, Tuple

from tensorflow.keras.layers import Add, Concatenate
from tensorflow.keras.models import Model

from interact.features import Feature, FeatureCollection, InteractionType
from interact.layers import AddBias, Linear, SparseLinear, SparseV, V


def FactorizationMachine(
    fs: List[Feature],
    interactions: List[Tuple[Feature,...]],
    k: int,
) -> Model:
    fc = FeatureCollection(
        features=fs,
        interactions=interactions,
    )
    dense_inputs = fc.get_dense_inputs()
    sparse_inputs = fc.get_sparse_inputs()

    dense_linear_output = []
    sparse_linear_output = []

    if len(dense_inputs):
        dense_linear_output = Linear()(
                Concatenate()(
                    dense_inputs
                )
            )
    if len(sparse_inputs):
        sparse_linear_output = Add()(
                [SparseLinear()(i) for i in sparse_inputs]
            )
    linear_output = AddBias()(
        Add()([dense_linear_output, sparse_linear_output])
    )

    interaction_outputs = []
    for interation_type, inputs in fc.get_interactions():
        if interation_type == InteractionType.DENSE:
            interaction_outputs.append(
                V(k=k)(Concatenate(inputs))
            )
        elif interation_type == InteractionType.SPARSE:
            interaction_outputs.append(
                SparseV(inputs, k=k)(inputs)
            )
        elif interation_type == InteractionType.MIXED:
            pass

    o = Add([linear_output] + interaction_outputs)

    return Model(fc.get_inputs(), o)

# each feature in interaction brings some V matrix
# feature can occur a couple of times in feature, each such occurrence brings new V!!!
# (all dense features are concatenated and V is produced)
#
#
#
