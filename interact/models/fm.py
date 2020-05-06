from typing import List, Tuple

from tensorflow.keras.layers import Add, Concatenate
from tensorflow.keras.models import Model

from interact.features import Feature, FeatureCollection, Interaction, InteractionType
from interact.layers import AddBias, Linear, MixedV, SparseLinear, SparseV, V


def FM(
    fs: List[Feature],
    interactions: List[Interaction],
) -> Model:
    fc = FeatureCollection(
        features=fs,
        interactions=interactions,
    )

    assert len(interactions) >= 1
    assert len(fs) >= 2

    products = []
    for interaction, inputs in fc.get_interactions():
        if interaction.interaction_type == InteractionType.DENSE:
            products.append(V(interaction)(inputs))
        elif interaction.interaction_type == InteractionType.SPARSE:
            products.append(SparseV(interaction)(inputs))
        else:
            products.append(MixedV(interaction)(inputs))

    if len(products) > 1:
        interaction_sum = Add()(products)
    else:
        interaction_sum = products[0]

    dense_inputs = fc.get_dense_inputs()
    sparse_inputs = fc.get_sparse_inputs()

    linear_part = []
    if len(dense_inputs) == 1:
        linear_part.append(Linear()(dense_inputs[0]))
    elif len(dense_inputs) > 1:
        linear_part.append(Linear()(Concatenate()(dense_inputs)))

    for sparse_input, sparse_f in zip(sparse_inputs, fc.get_sparse_features()):
        linear_part.append(SparseLinear(sparse_f.vocabulary_size)(sparse_input))

    if len(linear_part) > 1:
        linear_sum = Add()(linear_part)
    else:
        linear_sum = linear_part[0]

    o = AddBias()(Add()([interaction_sum, linear_sum]))
    return Model(fc.get_inputs(), o)
