from typing import List

import tensorflow as tf
from tensorflow.keras.layers import Add
from tensorflow.keras.models import Model

from interact.fields import Field, FieldsManager
from interact.layers import AddBias, AttentionAFM, Cartesian, Product


def AFM(
    fields: List[Field],
    l2_penalty: float = 0,
    t: int = 10
):
    """
    Args:
        fields:
        l2_penalty:
        t: attention factor
    """
    FieldsManager.validate_fields(fields)
    inputs = FieldsManager.fields2inputs(fields)

    # [(None, d), (None, d), ..]
    embeddings = [
        tf.squeeze(FieldsManager.input2embedding(i, field, l2_penalty=l2_penalty, averaged=True), axis=1)
        for i, field in zip(inputs, fields)
    ]

    # [(None, 1), (None, 1), ..]
    linear_terms = [
        FieldsManager.input2linear(i, field)
        for i, field in zip(inputs, fields)
    ]

    # (None, 1)
    linear_part = AddBias()(Add()(linear_terms))

    # (None, m, d)
    all_interactions = Cartesian()(embeddings)

    # (None, m, 1)
    weights = Product(t)(AttentionAFM(t)(all_interactions))

    # (None, m, 1)
    weighted_interactions = Product(fields[0].d)(all_interactions * weights)

    # (None, 1)
    interaction_part = tf.reduce_sum(tf.squeeze(weighted_interactions, axis=2), axis=1, keepdims=True)

    y_afm = linear_part + interaction_part

    return Model(inputs, y_afm)
