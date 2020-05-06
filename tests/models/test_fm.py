import numpy as np

from interact.features import DenseFeature, SparseFeature, Interaction
from interact.models import FM


def create_dense_feature(n: int = 100) -> np.ndarray:
    return np.random.uniform(size=n)


def create_sparse_feature(m: int, vocab_size: int, n: int = 100) -> np.ndarray:
    return 1 + np.random.choice(vocab_size, size=(n, m)).astype(np.int32)


def test_run_two_dense_features():
    k = 10
    df1 = DenseFeature('df1')
    df2 = DenseFeature('df2')
    features = [df1, df2]
    interaction = Interaction([df1, df2], k=k)
    fm = FM(features, [interaction])
    fm.compile(optimizer='sgd', loss='mse')

    xs = [create_dense_feature(), create_dense_feature()]
    ys = [create_dense_feature()]

    fm.fit(xs, ys, batch_size=1)


def test_run_two_sparse_features():
    m1 = 10
    m2 = 15
    vocab_size1 = 10 ** 4
    vocab_size2 = 150
    k = 10
    sf1 = SparseFeature('sf1', vocabulary_size=vocab_size1, m=m1)
    sf2 = SparseFeature('sf2', vocabulary_size=vocab_size2, m=m2)
    features = [sf1, sf2]
    interaction = Interaction([sf1, sf2], k=k)
    fm = FM(features, [interaction])
    fm.compile(optimizer='sgd', loss='mse')

    xs = [create_sparse_feature(m=m1, vocab_size=vocab_size1),
          create_sparse_feature(m=m2, vocab_size=vocab_size2)]
    ys = [create_dense_feature()]

    fm.fit(xs, ys, batch_size=1)


def test_run_one_dense_one_sparse_feature():
    m = 15
    vocab_size = 10 ** 4
    k = 10
    df = DenseFeature('df')
    sf = SparseFeature('sf', vocabulary_size=vocab_size, m=m)
    features = [df, sf]
    interaction = Interaction([df, sf], k=k)
    fm = FM(features, [interaction])
    fm.compile(optimizer='sgd', loss='mse')

    xs = [create_dense_feature(), create_sparse_feature(m=m, vocab_size=vocab_size)]
    ys = [create_dense_feature()]

    fm.fit(xs, ys, batch_size=1)
