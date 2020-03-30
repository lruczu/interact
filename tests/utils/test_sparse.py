import numpy as np
import pytest
from scipy import sparse

from interact.utils.sparse import to_sequences


X = np.zeros(shape=(4, 10000))
X[1, [2999, 9999]] = 1
X[2, [3]] = 1
X = sparse.csr_matrix(X)


def test_to_sequence_too_short_length_raise_warnings():
	with pytest.warns(UserWarning):
		X_seq = to_sequences(X, 1)


def test_to_sequence_too_short_length():
	X_seq = to_sequences(X, 1)
	assert X_seq.shape == (4, 1)
	
	obs1 = X_seq[0]
	obs2 = X_seq[1]
	obs3 = X_seq[2]
	obs4 = X_seq[3]

	assert obs1.sum() == 0
	assert 3000 in obs2 or 10000 in obs2
	assert 4 in obs3
	assert obs4.sum() == 0


def test_to_sequence_exact_length():
	X_seq = to_sequences(X, 2)
	assert X_seq.shape == (4, 2)

	obs1 = X_seq[0]
	obs2 = X_seq[1]
	obs3 = X_seq[2]
	obs4 = X_seq[3]

	assert obs1.sum() == 0
	assert 3000 in obs2 and 10000 in obs2
	assert 4 in obs3 and 0 in obs3
	assert obs4.sum() == 0


def test_to_sequence_too_long_length():
	X_seq = to_sequences(X, 10)
	assert X_seq.shape == (4, 10)
	
	obs1 = X_seq[0]
	obs2 = X_seq[1]
	obs3 = X_seq[2]
	obs4 = X_seq[3]

	assert obs1.sum() == 0
	assert (obs2 == 3000).sum() == 1 and (obs2 == 10000).sum() == 1 and (obs2 == 0).sum() == 8
	assert (obs3 == 4).sum() == 1 and (obs3 == 0).sum() == 9
	assert obs4.sum() == 0
