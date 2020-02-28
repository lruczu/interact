from collections import Counter
import warnings

import numpy as np
from scipy import sparse


def to_sequences(X: sparse.csr_matrix, seq_len: int=None) -> np.ndarray:
	"""

	Returns:
		Suppose that X has the shape 3 x 10000 and n_categories=3
		[
			[0, 0, 0],
			[1, 3000, 10000],
			[4, 9, 0]
		]
	"""
	n_rows, _ = X.shape
	X_seq = np.zeros(shape=(n_rows, seq_len))
	row_indices, col_indices = X.nonzero()
	
	non_empty_counter = Counter(row_indices)
	if non_empty_counter.most_common(1)[0][1] > seq_len:
		warnings.warn('Number of nonempty features is bigger than seq_len')

	#col_indices = col_indices + 1
	n = np.arange(len(row_indices))
	#k = row_indices % n_categories
	X_seq[row_indices, n % seq_len] = col_indices + 1
	
	#X_seq = X_seq * np.arange(1, n_categories + 1).reshape((1, -1))
	return X_seq
