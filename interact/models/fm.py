from typing import List

from tensorflow.keras.layers import Activation, Add
from tensorflow.keras.models import Model

from interact.fields import Field, FieldsManager
from interact.layers import AddBias, FMInteraction


def FM(
    fields: List[Field],
    l1_penalty: float = 0,
    l2_penalty: float = 0, 
    averaged: bool = False,
    activation: Activation = Activation('relu'),
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
        FieldsManager.input2embedding(i, field, l1_penalty=l1_penalty, l2_penalty=l2_penalty, averaged=averaged) 
        for i, field in zip(inputs, fields)
    ]

    linear_terms = [
        FieldsManager.input2linear(i, field)
         for i, field in zip(inputs, fields)
    ]

    fm_interaction = FMInteraction()

    interactions_part = fm_interaction(embeddings)
    linear_part = Add()(linear_terms) if len(linear_terms) > 1 else linear_terms[0]

    output = AddBias()(interactions_part + linear_part)

    output = activation(output)

    return Model(inputs, output)
