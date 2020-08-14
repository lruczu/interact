from typing import List

import tensorflow as tf
from tensorflow.keras.layers import Activation, Add, Concatenate, Dense
from tensorflow.keras.models import Model

from interact.fields import Field, FieldsManager
from interact.layers import AddBias, FMInteraction


def DeepFM(
    fields: List[Field], 
    l2_penalty: float = 0,
    activation: Activation = Activation('relu'),
):
    """
    Args:
        fields:
        l2_penalty:
    """
    FieldsManager.validate_fields(fields)
    inputs = FieldsManager.fields2inputs(fields)

    embeddings = [
        FieldsManager.input2embedding(i, field, l2_penalty=l2_penalty, averaged=True) 
        for i, field in zip(inputs, fields)
    ]

    linear_terms = [
        FieldsManager.input2linear(i, field)
         for i, field in zip(inputs, fields)
    ]

    fm_interaction = FMInteraction()

    interactions_part = fm_interaction(embeddings)
    linear_part = Add()(linear_terms)

    y_fm = AddBias()(interactions_part + linear_part)

    dnn_input = Concatenate()([tf.squeeze(e, axis=1) for e in embeddings])

    y_dnn = Dense(1)(Dense(10)(dnn_input))
    output = activation(y_fm + y_dnn)

    return Model(inputs, output)
