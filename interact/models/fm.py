from typing import List

from tensorflow.keras.layers import Add
from tensorflow.keras.models import Model

from interact.fields import Field, FieldsManager
from interact.layers import AddBias, FMInteraction


def FM(
    fields: List[Field], 
    l2_penalty: float = 0, 
    averaged: bool = False,
):
    """
    Args:
        fields:
        l2_penalty:
        averaged:
    """
    FieldsManager.validate_fields(fields)
    inputs = FieldsManager.fields2inputs(fields)

    embeddings = [
        FieldsManager.input2embedding(i, field, l2_penalty=l2_penalty, averaged=averaged) 
        for i, field in zip(inputs, fields)
    ]

    linear_terms = [
        FieldsManager.input2linear(i, field)
         for i, field in zip(inputs, fields)
    ]

    fm_interaction = FMInteraction()

    interactions_part = fm_interaction(embeddings)
    linear_part = Add()(linear_terms)

    ouput = AddBias()(interactions_part + linear_part)

    return Model(inputs, ouput)
