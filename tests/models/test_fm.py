import numpy as np

from interact.fields import DenseField, SparseField
from interact.models import FM


def create_dense_feature(n: int = 100) -> np.ndarray:
    return np.random.uniform(size=n)


def create_sparse_feature(m: int, vocab_size: int, n: int = 100) -> np.ndarray:
    return 1 + np.random.choice(vocab_size, size=(n, m)).astype(np.int32)


def test_run_two_dense_features():
    df1 = DenseField('df1', d=10)
    df2 = DenseField('df2', d=10)
    fields = [df1, df2]
    fm = FM(fields)

    fm.compile(optimizer='sgd', loss='mse')

    xs = [create_dense_feature(), create_dense_feature()]
    ys = [create_dense_feature()]

    fm.fit(xs, ys, batch_size=1)


def test_run_two_sparse_features():
    sf1 = SparseField('sf1', vocabulary_size=150, m=10, d=10)
    sf2 = SparseField('sf2', vocabulary_size=10**4, m=15, d=10)
    fields = [sf1, sf2]
    fm = FM(fields)
    fm.compile(optimizer='sgd', loss='mse')

    xs = [create_sparse_feature(m=10, vocab_size=150),
          create_sparse_feature(m=15, vocab_size=10**4)]
    ys = [create_dense_feature()]

    fm.fit(xs, ys, batch_size=1)


def test_run_one_dense_one_sparse_feature():
    df = DenseField('df', d=10)
    sf = SparseField('sf', vocabulary_size=10**4, m=15, d=10)
    fields = [df, sf]
    fm = FM(fields)
    fm.compile(optimizer='sgd', loss='mse')

    xs = [create_dense_feature(), create_sparse_feature(m=15, vocab_size=10**4)]
    ys = [create_dense_feature()]

    fm.fit(xs, ys, batch_size=1)
