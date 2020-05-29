import itertools
from typing import List

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Concatenate, Input
from tensorflow.keras.models import Model

from interact.features import DenseFeature, SparseFeature, Interaction
from interact.fields import DenseField, FieldsManager, SparseField
from interact.layers import DenseEmbedding, FMInteraction, SparseEmbedding


def _get_seq_vector(indices: List[int], m: int) -> np.ndarray:
    seq_v = np.zeros(m)
    if len(indices) == 0:
        return seq_v
    seq_v[:len(indices)] = np.array(indices)
    return seq_v


def _get_two_order_interactions(indices: List[int], weights: np.ndarray) -> float:
    sum_ = 0
    for i1, i2 in itertools.combinations(indices, 2):
        sum_ += np.sum(weights[i1] * weights[i2])
    return sum_


def test_3_dense_fields():
    fields = [DenseField('field1', d=20), DenseField('field2', d=20), DenseField('field3', d=20)]
    inputs = FieldsManager.fields2inputs(fields)

    e1 = DenseEmbedding(fields[0])(inputs[0])
    e2 = DenseEmbedding(fields[1])(inputs[1])
    e3 = DenseEmbedding(fields[2])(inputs[2])

    fm_interaction = FMInteraction()

    o = fm_interaction([e1, e2, e3])

    model = Model(inputs, o)

    w1, w2, w3 = model.get_weights()
    v1, v2, v3 = 1.3, 7.2, 9.9
    i = np.array([
        [0, 0, 0],
        [0, 0, v3],
        [0, v2, 0],
        [0, v2, v3],
        [v1, 0, 0],
        [v1, 0, v3],
        [v1, v2, 0],
        [v1, v2, v3]
    ])
    expected_o = [
        [0],
        [0],
        [0],
        [v2 * v3 * np.sum(w2 * w3)],
        [0],
        [v1 * v3 * np.sum(w1 * w3)],
        [v1 * v2 * np.sum(w1 * w2)],
        [
            v1 * v2 * np.sum(w1 * w2) +
            v2 * v3 * np.sum(w2 * w3) +
            v1 * v3 * np.sum(w1 * w3)
        ],
    ]
    o = model.predict([i[:, 0], i[:, 1], i[:, 2]])
    assert np.all(np.isclose(o, expected_o, atol=1e-04))


def test_1_sparse_fields():
    fields = [SparseField('field1', vocabulary_size=150, m=40, d=30)]
    inputs = FieldsManager.fields2inputs(fields)
    
    e1 = SparseEmbedding(fields[0])(inputs[0])

    fm_interaction = FMInteraction()

    o = fm_interaction([e1])

    model = Model(inputs, o)
    w, _ = model.get_weights()

    i = np.vstack([
        _get_seq_vector([], m=40),
        _get_seq_vector([1], m=40),
        _get_seq_vector([40], m=40),
        _get_seq_vector([1, 40], m=40),
        _get_seq_vector([1, 20, 40], m=40),
        _get_seq_vector([1, 10, 20, 40], m=40),
    ])
    expected_o = [
        [_get_two_order_interactions([], weights=w)],
        [_get_two_order_interactions([1], weights=w)],
        [_get_two_order_interactions([40], weights=w)],
        [_get_two_order_interactions([1, 40], weights=w)],
        [_get_two_order_interactions([1, 20, 40], weights=w)],
        [_get_two_order_interactions([1, 10, 20, 40], weights=w)],
    ]
    o = model.predict([i])
    assert np.all(np.isclose(o, expected_o, atol=1e-04))


def test_1_dense_1_sparse_field():
    fields = [DenseField('field1', d=30), SparseField('field2',  vocabulary_size=150, m=40, d=30)]
    inputs = FieldsManager.fields2inputs(fields)

    e1 = DenseEmbedding(fields[0])(inputs[0])
    e2 = SparseEmbedding(fields[1])(inputs[1])

    fm_interaction = FMInteraction()

    o = fm_interaction([e1, e2])

    model = Model(inputs, o)

    dense_w, sparse_w, _ = model.get_weights()

    i_dense = np.array([13.4, 5, 3, 9])
    i_sparse = np.vstack([
        _get_seq_vector([], m=40),
        _get_seq_vector([1], m=40),
        _get_seq_vector([40], m=40),
        _get_seq_vector([1, 40], m=40),
    ])
    expected_o = [
        [0],
        [5 * (dense_w[0] * sparse_w[1]).sum()],
        [3 * (dense_w[0] * sparse_w[40]).sum()],
        [
            9 * (dense_w[0] * sparse_w[1]).sum()
            + 9 * (dense_w[0] * sparse_w[40]).sum()
            + (sparse_w[1] * sparse_w[40]).sum()
        ]
    ]
    o = model.predict([i_dense, i_sparse])
    assert np.all(np.isclose(o, expected_o))
