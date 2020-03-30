import itertools
from typing import List

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Concatenate, Input
from tensorflow.keras.models import Model

from interact.features import Interaction, SparseFeature
from interact.layers import SparseV


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


def _get_two_order_interactions_from_two_inputs(
        indices1: List[int],
        indices2: List[int],
        weights1: np.ndarray,
        weights2: np.ndarray,
        within: bool = True
):

    weights = np.vstack([weights1, weights2])
    indices = np.hstack([
        indices1, np.array(indices2) + weights1.shape[0]  # add (m1 + 1) to indices
    ]).tolist()
    all_interactions = _get_two_order_interactions(indices, weights)
    if within:
        return all_interactions
    return all_interactions - (
            _get_two_order_interactions(indices1, weights1) +
            _get_two_order_interactions(indices2, weights2)
    )


def test_sparse_v_one_feature():
    k = 30
    m = 40
    vocab_size = 150
    w = np.random.uniform(size=(vocab_size + 1, k))
    i = Input(shape=m, dtype=tf.int32)
    sf = SparseFeature('sf', vocab_size, m=m)
    sparse_v = SparseV(Interaction([sf], k=k))
    o = sparse_v([i])
    m = Model([i], o)
    e, mask = m.get_weights()
    m.set_weights([w, mask])

    i = np.vstack([
        _get_seq_vector([], m=40),
        _get_seq_vector([1], m=40),
        _get_seq_vector([40], m=40),
        _get_seq_vector([1, 40], m=40),
        _get_seq_vector([1, 20, 40], m=40),
        _get_seq_vector([1, 10, 20, 40], m=40),
    ])
    expected_o = [
        _get_two_order_interactions([], weights=w),
        _get_two_order_interactions([1], weights=w),
        _get_two_order_interactions([40], weights=w),
        _get_two_order_interactions([1, 40], weights=w),
        _get_two_order_interactions([1, 20, 40], weights=w),
        _get_two_order_interactions([1, 10, 20, 40], weights=w),
    ]
    o = m.predict([i])
    assert np.all(np.isclose(o, expected_o, atol=1e-04))


def test_sparse_v_two_features():
    k = 30
    m1 = 40
    m2 = 20
    vocab_size1 = 150
    vocab_size2 = 10000

    i1 = Input(shape=m1, dtype=tf.int32)
    i2 = Input(shape=m2, dtype=tf.int32)

    sf1 = SparseFeature('sf1', vocabulary_size=vocab_size1, m=m1)
    sf2 = SparseFeature('sf2', vocabulary_size=vocab_size2, m=m2)

    sparse_v = SparseV(Interaction([sf1, sf2], k=k))
    o = sparse_v([i1, i2])
    m = Model([i1, i2], o)
    e1, e2, mask1, mask2 = m.get_weights()
    w1 = np.random.uniform(size=(vocab_size1 + 1, k))
    w2 = np.random.uniform(size=(vocab_size2 + 1, k))
    m.set_weights([w1, w2, mask1, mask2])

    i1 = np.vstack([
        _get_seq_vector([], m=m1),  # zero in both
        _get_seq_vector([m1], m=m1),  # only one in sf1
        _get_seq_vector([], m=m1),  # only one in sf2
        _get_seq_vector([m1], m=m1),  # one in sf1 and one in sf2
        _get_seq_vector([1, m1], m=m1),  # two in sf1 and one in sf2
        _get_seq_vector([m1], m=m1),  # one in sf1 and two in sf2
        _get_seq_vector([1, m1], m=m1),  # two in sf1 and in sf2
        _get_seq_vector([1, 2, 3, m1], m=m1)  # more complex example
    ])

    i2 = np.vstack([
        _get_seq_vector([], m=m2),
        _get_seq_vector([], m=m2),
        _get_seq_vector([m2], m=m2),
        _get_seq_vector([m2], m=m2),
        _get_seq_vector([m2], m=m2),
        _get_seq_vector([1, m2], m=m2),
        _get_seq_vector([1, m2], m=m2),
        _get_seq_vector([1, 2, 3, m2], m=m2),
    ])

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


def test_sparse_v_two_features_without_interaction_within_field():
    k = 30
    m1 = 40
    m2 = 20
    vocab_size1 = 150
    vocab_size2 = 10000

    i1 = Input(shape=m1, dtype=tf.int32)
    i2 = Input(shape=m2, dtype=tf.int32)

    sf1 = SparseFeature('sf1', vocabulary_size=vocab_size1, m=m1)
    sf2 = SparseFeature('sf2', vocabulary_size=vocab_size2, m=m2)

    sparse_v = SparseV(Interaction([sf1, sf2], k=k), within=False)
    o = sparse_v([i1, i2])
    m = Model([i1, i2], o)
    e1, e2, mask1, mask2 = m.get_weights()
    w1 = np.random.uniform(size=(vocab_size1 + 1, k))
    w2 = np.random.uniform(size=(vocab_size2 + 1, k))
    m.set_weights([w1, w2, mask1, mask2])

    i1 = np.vstack([
        _get_seq_vector([], m=m1),  # zero in both
        _get_seq_vector([m1], m=m1),  # only one in sf1
        _get_seq_vector([], m=m1),  # only one in sf2
        _get_seq_vector([m1], m=m1),  # one in sf1 and one in sf2
        _get_seq_vector([1, m1], m=m1),  # two in sf1 and one in sf2
        _get_seq_vector([m1], m=m1),  # one in sf1 and two in sf2
        _get_seq_vector([1, m1], m=m1),  # two in sf1 and in sf2
        _get_seq_vector([1, 2, 3, m1], m=m1)  # more complex example
    ])

    i2 = np.vstack([
        _get_seq_vector([], m=m2),
        _get_seq_vector([], m=m2),
        _get_seq_vector([m2], m=m2),
        _get_seq_vector([m2], m=m2),
        _get_seq_vector([m2], m=m2),
        _get_seq_vector([1, m2], m=m2),
        _get_seq_vector([1, m2], m=m2),
        _get_seq_vector([1, 2, 3, m2], m=m2),
    ])

    expected_o = [
        _get_two_order_interactions_from_two_inputs([], [], w1, w2, within=False),
        _get_two_order_interactions_from_two_inputs([m1], [], w1, w2, within=False),
        _get_two_order_interactions_from_two_inputs([], [m2], w1, w2, within=False),
        _get_two_order_interactions_from_two_inputs([m1], [m2], w1, w2, within=False),
        _get_two_order_interactions_from_two_inputs([1, m1], [m2], w1, w2, within=False),
        _get_two_order_interactions_from_two_inputs([m1], [1, m2], w1, w2, within=False),
        _get_two_order_interactions_from_two_inputs([1, m1], [1, m2], w1, w2, within=False),
        _get_two_order_interactions_from_two_inputs([1, 2, 3, m1], [1, 2, 3, m2], w1, w2, within=False),
    ]
    o = m.predict([i1, i2])
    assert np.all(np.isclose(o, expected_o))
