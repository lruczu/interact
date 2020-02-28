import pytest

from interact.exceptions import DuplicateFeature, InteractionWithOnlyOneVariable, UnexpectedFeatureInInteractions
from interact.features import DenseFeature, SparseFeature, FeatureCollection


f_price = DenseFeature('price')
f_description = SparseFeature('description', vocabulary_size=10 ** 5)
f_category = SparseFeature('category', vocabulary_size=150)


def test_duplicate_in_feature_list():
    with pytest.raises(DuplicateFeature):
        _ = FeatureCollection(
            [f_price, f_description, f_price], []
        )


def test_unexpected_feature_in_interactions():
    with pytest.raises(UnexpectedFeatureInInteractions):
        _ = FeatureCollection(
            [f_price, f_description],
            [(f_price, f_description, f_category)],
        )


def test_interaction_with_only_one_variable():
    with pytest.raises(InteractionWithOnlyOneVariable):
        _ = FeatureCollection(
            [f_price, f_description], [(f_description, )]
        )
