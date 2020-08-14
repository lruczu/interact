from typing import List

import tensorflow as tf
from tensorflow.keras.layers import Activation, Add, Flatten
from tensorflow.keras.models import Model

from interact.fields import Field, FieldsManager
from interact.layers import AddBias, AttentionAutoInt, Cartesian, Product


def AutoInt(
    fields: List[Field],
    d_prime: int,
    h: int,
    l2_penalty: float = 0,
    activation: Activation = Activation('relu'),
):
    FieldsManager.validate_fields(fields)
    inputs = FieldsManager.fields2inputs(fields)

    # [(None, d), (None, d), ..]
    embeddings = [
        tf.squeeze(FieldsManager.input2embedding(i, field, l2_penalty=l2_penalty, averaged=True), axis=1)
        for i, field in zip(inputs, fields)
    ]

    y_autoint = AddBias(
        Product()(
            Flatten()(
                AttentionAutoInt(embeddings)
            )
        )
    )

    output = activation(y_autoint)

    return Model(inputs, output)
