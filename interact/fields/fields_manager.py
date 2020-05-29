from typing import List

import tensorflow as tf
from tensorflow.keras.layers import Input

from interact.fields import DenseField, Field, SparseField
from interact.layers import DenseEmbedding, Linear, SparseEmbedding, SparseLinear


class FieldsManager:
    @staticmethod
    def validate_fields(fields: List[Fields]) -> None:
        if set([field.d for field in fields]) > 1:
            raise ValueError('Fields must have the same embedding dimension.')

    @staticmethod
    def fields2inputs(fields: List[Field]) -> List[Input]:
        inputs = []
        for field in fields:
            if isinstance(field, DenseField):
                inputs.append(Input(shape=1, dtype=field.dtype))
            elif isinstance(field, SparseField):
                inputs.append(Input(shape=field.m, dtype=field.dtype))
            else:
                raise ValueError('Wrong type of the field')
        return inputs

    @staticmethod
    def input2embedding(
        i: Input, 
        field: Field, 
        l2_penalty: float = 0, 
        averaged: bool = True,
    ) -> tf.Tensor:
        if i.dtype != field.dtype:
            raise ValueError('Input and field must have the same type.')

        if isinstance(field, DenseField):
            return DenseEmbedding(field, l2_penalty=l2_penalty)(i)
        elif isinstance(field, SparseField):
            return SparseEmbedding(field, l2_penalty=l2_penalty, averaged=averaged)(i)
        else:
            raise ValueError('Wrong type of field.')

    @staticmethod
    def input2linear(
        i: Input, 
        field: Field,
    ) -> tf.Tensor:
        if i.dtype != field.dtype:
            raise ValueError('Input and field must have the same type.')

        if isinstance(field, DenseField):
            return Linear()(i)
        elif isinstance(field, SparseField):
            return SparseLinear(vocabulary_size=field.vocabulary_size)(i)
        else:
            raise ValueError('Wrong type of field.')
