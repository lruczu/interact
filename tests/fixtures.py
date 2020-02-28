from interact.features import DenseFeature, SparseFeature

import numpy as np
import pytest
from sklearn.feature_extraction.text import CountVectorizer

N = 10
data = {
    'price': [1.12, 10, 0.99, 56, 85, 88, 100, 1.10, 14.99, 99.99],
    'age': [1, 11, 21, 31, 41, 51, 61, 71, 81, 91],
    'description':
        [
            'word1 word2 word3 word4.',
            'word5',
            'word1 word11 word101 word5',
            '',
            'word5 word2 word13 word4.',
            'word11 word2 word4.',
            'word1 word6 word8 word4.',
            'word2 word2 word3 word4.',
            'word1 word1 word3 word4.',
            'word1 word2.'
        ],
    'category': [1, 12, 12, 14, 15, 19, 55, 89, 99, 100]
}
vectorizer = CountVectorizer()
vectorizer.fit(data['description'])


@pytest.fixture()
def binary_target():
    return np.random.choice([0, 1], size=N)


@pytest.fixture()
def f_price():
    return DenseFeature('price')


@pytest.fixture()
def f_price_data():
    return np.array(data['price'])


@pytest.fixture()
def f_age():
    return DenseFeature('age')


@pytest.fixture()
def f_age_data():
    return np.array(data['age'])


@pytest.fixture()
def f_description():
    return SparseFeature('description', vocabulary_size=len(vectorizer.vocabulary_), n_levels=4)


@pytest.fixture()
def f_category():
    return SparseFeature('category', vocabulary_size=100, n_levels=1)
