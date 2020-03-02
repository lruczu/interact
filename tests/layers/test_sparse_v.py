import itertools
from typing import List

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Concatenate, Input
from tensorflow.keras.models import Model

from interact.features import SparseFeature
from interact.layers import SparseV


def _get_seq_vector(indices: List[int], m: int) -> np.ndarray:
    assert min(indices) >= 1
    seq_v = np.zeros(m)
    seq_v[:len(indices)] = np.array(indices) + 1
    return seq_v


def _get_two_order_interactions(indices: List[int], weights: np.ndarray) -> float:
    sum_ = 0
    for i1, i2 in itertools.combinations(indices, 2):
        sum_ = np.sum(weights[i1] * weights[i2])
    return sum_


def _get_two_order_interactions_from_two_inputs(
        indices1: List[int],
        indices2: List[int],
        weights1: np.ndarray,
        weights2: np.ndarray,
):
    weights = np.vstack([weights1, weights2])
    indices = np.hstack([
        indices1, np.array(indices2) + weights1.shape[0] - 1  # add m1 to indices
    ]).tolist()
    return _get_two_order_interactions(indices, weights)


def test_sparse_v_one_feature():
    k = 30
    m = 40
    i = Input(shape=m, dtype=tf.int32)
    sf = SparseFeature('sf', 150, m=m)
    sparse_v = SparseV([sf], k=k)
    o = sparse_v([i])([i])
    m = Model([i], o)
    e, mask = m.get_weights()
    w = np.random.uniform(size=(m+1, k))
    m.set_weights([w, mask])

    i = [
        _get_seq_vector([], m=40),
        _get_seq_vector([1], m=40),
        _get_seq_vector([40], m=40),
        _get_seq_vector([1, 40], m=40),
        _get_seq_vector([1, 20, 40], m=40),
        _get_seq_vector([1, 10, 20, 40], m=40),
    ]
    expected_o = [
        _get_two_order_interactions([], weights=w),
        _get_two_order_interactions([1], weights=w),
        _get_two_order_interactions([40], weights=w),
        _get_two_order_interactions([1, 40], weights=w),
        _get_two_order_interactions([1, 20, 40], weights=w),
        _get_two_order_interactions([1, 10, 20, 40], weights=w),
    ]
    o = m.predict(i)
    assert np.all(np.isclose(o, expected_o))


def test_sparse_v_two_features():
    k = 30
    m1 = 40
    m2 = 20

    i1 = Input(shape=m1, dtype=tf.int32)
    i2 = Input(shape=m2, dtype=tf.int32)

    sf1 = SparseFeature('sf1', 150, m=m1)
    sf2 = SparseFeature('sf2', 10 ** 4, m=m2)

    sparse_v = SparseV([sf1, sf2], k=k)
    o = sparse_v([i1, i2])
    m = Model([i1, i2], o)
    e1, mask1, e2, mask2 = m.get_weights()
    w1 = np.random.uniform(size=(m1 + 1, k))
    w2 = np.random.uniform(size=(m2 + 1, k))
    m.set_weights([w1, mask1, w2, mask2])

    i1 = [
        _get_seq_vector([], m=m1),  # zero in both
        _get_seq_vector([m1], m=m1),  # only one in sf1
        _get_seq_vector([], m=m1),  # only one in sf2
        _get_seq_vector([m1], m=m1),  # one in sf1 and one in sf2
        _get_seq_vector([1, m1], m=m1),  # two in sf1 and one in sf2
        _get_seq_vector([m1], m=m1),  # one in sf1 and two in sf2
        _get_seq_vector([1, m1], m=m1),  # two in sf1 and in sf2
        _get_seq_vector([1, 2, 3, m1], m=m1)  # more complex example
    ]

    i2 = [
        _get_seq_vector([], m=m2),
        _get_seq_vector([], m=m2),
        _get_seq_vector([m2], m=m2),
        _get_seq_vector([m2], m=m2),
        _get_seq_vector([m2], m=m2),
        _get_seq_vector([1, m2], m=m2),
        _get_seq_vector([1, m2], m=m2),
        _get_seq_vector([1, 2, 3, m2], m=m2),
    ]

    expected_o = [
        _get_two_order_interactions_from_two_inputs([], [], w1, w2),
        _get_two_order_interactions_from_two_inputs([m1], [], w1, w2),
        _get_two_order_interactions_from_two_inputs([], [m2], w1, w2),
        _get_two_order_interactions_from_two_inputs([m1], [m2], w1, w2),
        _get_two_order_interactions_from_two_inputs([1, m1], [m2], w1, w2),
        _get_two_order_interactions_from_two_inputs([m1], [1, m2], w1, w2),
        _get_two_order_interactions_from_two_inputs([1, m1], [1, m2], w1, w2),
        _get_two_order_interactions_from_two_inputs([1, 2, 3, m1], [1, 2, 3, m2], w1, w2),
    ]
    o = m.predict([i1, i2])
    assert np.all(np.isclose(o, expected_o))
