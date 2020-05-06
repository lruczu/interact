from typing import List

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Concatenate, Input
from tensorflow.keras.models import Model

from interact.features import DenseFeature, SparseFeature, Interaction
from interact.layers import FM, MixedV


def _get_seq_vector(indices: List[int], m: int) -> np.ndarray:
    seq_v = np.zeros(m)
    if len(indices) == 0:
        return seq_v
    seq_v[:len(indices)] = np.array(indices)
    return seq_v


def test_one_dense_one_sparse():
    k = 30
    m = 40
    vocab_size = 150
    dense_i = Input(shape=1, dtype=tf.float32)
    sparse_i = Input(shape=m, dtype=tf.int32)
    df = DenseFeature('df')
    sf = SparseFeature('sf', vocab_size, m=m)
    interaction = Interaction([df, sf], k=k)

    mixed_v = MixedV(interaction)
    fm = FM(interaction)

    e = mixed_v([dense_i, sparse_i])
    products = fm(e)
    m = Model([dense_i, sparse_i], products)
    dense_v, sparse_v, c1, c2 = m.get_weights()
    dense_w = np.random.uniform(size=dense_v.shape)
    sparse_w = np.random.uniform(size=sparse_v.shape)
    m.set_weights([dense_w, sparse_w, c1, c2])

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
    o = m.predict([i_dense, i_sparse])
    assert np.all(np.isclose(o, expected_o))


def test_one_dense_two_sparse():
    k = 30
    m1 = 20
    m2 = 40
    vocab_size1 = 150
    vocab_size2 = 200
    dense_i = Input(shape=1, dtype=tf.float32)
    sparse1_i = Input(shape=m1, dtype=tf.int32)
    sparse2_i = Input(shape=m2, dtype=tf.int32)
    df = DenseFeature('df')
    sf1 = SparseFeature('sf1', vocab_size1, m=m1)
    sf2 = SparseFeature('sf2', vocab_size2, m=m2)
    interaction = Interaction([df, sf1, sf2], k=k)

    mixed_v = MixedV(interaction)
    fm = FM(interaction)

    e = mixed_v([dense_i, sparse1_i, sparse2_i])
    products = fm(e)
    m = Model([dense_i, sparse1_i, sparse2_i], products)
    dense_v, sparse1_v, sparse2_v, c1, c2, c3 = m.get_weights()
    dense_w = np.random.uniform(size=dense_v.shape)
    sparse1_w = np.random.uniform(size=sparse1_v.shape)
    sparse2_w = np.random.uniform(size=sparse2_v.shape)
    m.set_weights([dense_w, sparse1_w, sparse2_w, c1, c2, c3])
    i_dense = np.array([13.4, 5])
    i_sparse1 = np.vstack([
        _get_seq_vector([1], m=m1),
        _get_seq_vector([1], m=m1),
    ])
    i_sparse2 = np.vstack([
        _get_seq_vector([], m=m2),
        _get_seq_vector([1], m=m2),
    ])
    expected_o = [
        [13.4 * (dense_w[0] * sparse1_w[1]).sum()],
        [
            5 * (dense_w[0] * sparse1_w[1]).sum()
            + 5 * (dense_w[0] * sparse2_w[1]).sum()
            + (sparse1_w[1] * sparse2_w[1]).sum()
        ]
    ]
    o = m.predict([i_dense, i_sparse1, i_sparse2])
    assert np.all(np.isclose(o, expected_o))


def test_two_dense_two_sparse():
    k = 30
    m1 = 40
    m2 = 20
    vocab_size1 = 150
    vocab_size2 = 200
    dense1_i = Input(shape=1, dtype=tf.float32)
    dense2_i = Input(shape=1, dtype=tf.float32)
    sparse1_i = Input(shape=m1, dtype=tf.int32)
    sparse2_i = Input(shape=m2, dtype=tf.int32)
    df1 = DenseFeature('df1')
    df2 = DenseFeature('df2')
    sf1 = SparseFeature('sf1', vocab_size1, m=m1)
    sf2 = SparseFeature('sf2', vocab_size2, m=m2)
    interaction = Interaction([df1, df2, sf1, sf2], k=k)
    mixed_v = MixedV(interaction)
    fm = FM(interaction)

    e = mixed_v([dense1_i, dense2_i, sparse1_i, sparse2_i])
    products = fm(e)

    m = Model([dense1_i, dense2_i, sparse1_i, sparse2_i], products)
    dense_v, sparse1_v, sparse2_v, c1, c2, c3 = m.get_weights()
    dense_w = np.random.uniform(size=dense_v.shape)
    sparse1_w = np.random.uniform(size=sparse1_v.shape)
    sparse2_w = np.random.uniform(size=sparse2_v.shape)
    m.set_weights([dense_w, sparse1_w, sparse2_w, c1, c2, c3])
    i_dense1 = np.array([13.4, 5])
    i_dense2 = np.array([8.1, 0])
    i_sparse1 = np.vstack([
        _get_seq_vector([1], m=m1),
        _get_seq_vector([1, m1], m=m1),
    ])
    i_sparse2 = np.vstack([
        _get_seq_vector([], m=m2),
        _get_seq_vector([1], m=m2),
    ])
    expected_o = [
        [
            13.4 * 8.1 * (dense_w[0] * dense_w[1]).sum()
            + 13.4 * (dense_w[0] * sparse1_w[1]).sum()
            + 8.1 * (dense_w[1] * sparse1_w[1]).sum()
        ],
        [
            5 * (dense_w[0] * sparse1_w[1]).sum()
            + 5 * (dense_w[0] * sparse1_w[m1]).sum()
            + 5 * (dense_w[0] * sparse2_w[1]).sum()
            + (sparse1_w[1] * sparse2_w[1]).sum()
            + (sparse1_w[m1] * sparse2_w[1]).sum()
            + (sparse1_w[1] * sparse1_w[m1]).sum()
        ]
    ]
    o = m.predict([i_dense1, i_dense2, i_sparse1, i_sparse2])
    assert np.all(np.isclose(o, expected_o))


def exclude_for_now_test_two_dense_two_sparse_without_interaction_within_field():
    k = 30
    m1 = 40
    m2 = 20
    vocab_size1 = 150
    vocab_size2 = 200
    dense1_i = Input(shape=1, dtype=tf.float32)
    dense2_i = Input(shape=1, dtype=tf.float32)
    sparse1_i = Input(shape=m1, dtype=tf.int32)
    sparse2_i = Input(shape=m2, dtype=tf.int32)
    df1 = DenseFeature('df1')
    df2 = DenseFeature('df2')
    sf1 = SparseFeature('sf1', vocab_size1, m=m1)
    sf2 = SparseFeature('sf2', vocab_size2, m=m2)
    interaction = Interaction([df1, df2, sf1, sf2], k=k)

    mixed_v = MixedV(interaction)
    fm = FM(interaction)

    e = mixed_v([dense1_i, dense2_i, sparse1_i, sparse2_i])
    products = fm(e)

    m = Model([dense1_i, dense2_i, sparse1_i, sparse2_i], products)
    dense_v, sparse1_v, sparse2_v, c1, c2, c3 = m.get_weights()
    dense_w = np.random.uniform(size=dense_v.shape)
    sparse1_w = np.random.uniform(size=sparse1_v.shape)
    sparse2_w = np.random.uniform(size=sparse2_v.shape)
    m.set_weights([dense_w, sparse1_w, sparse2_w, c1, c2, c3])
    i_dense1 = np.array([13.4, 5])
    i_dense2 = np.array([8.1, 0])
    i_sparse1 = np.vstack([
        _get_seq_vector([1], m=m1),
        _get_seq_vector([1, m1], m=m1),
    ])
    i_sparse2 = np.vstack([
        _get_seq_vector([], m=m2),
        _get_seq_vector([1], m=m2),
    ])
    expected_o = [
        [
            13.4 * 8.1 * (dense_w[0] * dense_w[1]).sum()
            + 13.4 * (dense_w[0] * sparse1_w[1]).sum()
            + 8.1 * (dense_w[1] * sparse1_w[1]).sum()
        ],
        [
            5 * (dense_w[0] * sparse1_w[1]).sum()
            + 5 * (dense_w[0] * sparse1_w[m1]).sum()
            + 5 * (dense_w[0] * sparse2_w[1]).sum()
            + (sparse1_w[1] * sparse2_w[1]).sum()
            + (sparse1_w[m1] * sparse2_w[1]).sum()
        ]
    ]
    o = m.predict([i_dense1, i_dense2, i_sparse1, i_sparse2])
    assert np.all(np.isclose(o, expected_o))
