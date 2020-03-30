import pytest

from interact.exceptions import (DuplicateFeature,
                                 NoFeatureProvided,
                                 UnexpectedFeatureInInteractions,
                                 NoFeatureProvidedInInteraction,
                                 SingleDenseFeatureInInteraction
                                 )
from interact.features import DenseFeature, FeatureCollection, Interaction, SparseFeature


f_price = DenseFeature('price')
f_description = SparseFeature('description', vocabulary_size=10 ** 5, m=10)
f_category = SparseFeature('category', vocabulary_size=150, m=1)


def test_no_features_in_feature_collection():
    with pytest.raises(NoFeatureProvided):
        _ = FeatureCollection([], [])


def test_no_feautres_in_interaction():
    with pytest.raises(NoFeatureProvidedInInteraction):
        _ = Interaction([], k=1)


def test_duplicate_in_feature_list():
    with pytest.raises(DuplicateFeature):
        _ = FeatureCollection(
            [f_price, f_description, f_price], []
        )


def test_duplicate_in_interaction():
    with pytest.raises(DuplicateFeature):
        _ = FeatureCollection(
            [f_price, f_description, f_category],
            [Interaction([f_price, f_price], k=10)]
        )


def test_unexpected_feature_in_interactions():
    with pytest.raises(UnexpectedFeatureInInteractions):
        _ = FeatureCollection(
            [f_price, f_description],
            [Interaction([f_price, f_description, f_category], k=10)],
        )


def test_interaction_with_only_one_dense_variable():
    with pytest.raises(SingleDenseFeatureInInteraction):
        _ = FeatureCollection(
            [f_price, f_description],
            [Interaction([f_price], k=10)]
        )


def test_interaction_with_only_one_sparse_variable():
        _ = FeatureCollection(
            [f_price, f_description],
            [Interaction([f_description], k=10)]
        )
