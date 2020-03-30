import enum
from functools import reduce
from typing import Generator, List, Sequence, Tuple

import tensorflow as tf
from tensorflow.keras.layers import Input

from interact.exceptions import (
	DuplicateFeature,
	NoFeatureProvided,
	NoFeatureProvidedInInteraction,
	UnexpectedFeatureInInteractions,
	SingleDenseFeatureInInteraction,
)


class Feature:
	def get_name(self) -> str:
		raise NotImplementedError

	def get_full_name(self) -> str:
		raise NotImplementedError


class DenseFeature(Feature):
	def __init__(
		self, 
		name: str, 
		dtype: str = 'float32',
	):
		self.name = name
		self.dtype = dtype

	def get_name(self) -> str:
		return self.name

	def get_full_name(self) -> str:
		return f'{self.name}: {self.dtype} of shape (None, 1)'

	def __eq__(self, other):
		if not isinstance(other, Feature):
			return False
		return self.get_full_name() == other.get_full_name()

	def __repr__(self):
		return str(self.__dict__)


class SparseFeature(Feature):
	def __init__(
		self, 
		name: str, 
		vocabulary_size: int,
		m: int,
		dtype: str = 'int32',
	):
		if vocabulary_size < m:
			raise ValueError('m cannot be bigger than vocabulary size.')
		self.name = name
		self.vocabulary_size = vocabulary_size
		self.m = m
		self.dtype = dtype

	def get_name(self) -> str:
		return self.name

	def get_full_name(self) -> str:
		return f'{self.name}: {self.dtype} of shape (None, {self.m})'

	def __eq__(self, other):
		if not isinstance(other, Feature):
			return False
		return self.get_full_name() == other.get_full_name()

	def __repr__(self):
		return str(self.__dict__)


class InteractionType(enum.Enum):
	DENSE = "Dense"
	SPARSE = "Sparse"
	MIXED = "Mixed"


class Interaction:
	def __init__(self, features: Sequence[Feature], k: int):
		self.features = features
		self.k = k

		self._check_input()

		self.n_features = len(features)
		self.interaction_type = self._get_interaction_type(self.features)

	def get_feature_names(self) -> List[str]:
		return [f.get_name() for f in self.features]

	def _check_input(self):
		if len(self.features) == 0:
			raise NoFeatureProvidedInInteraction
		if len(self.features) == 1 and isinstance(self.features[0], DenseFeature):
			raise SingleDenseFeatureInInteraction

		feature_names = [f.get_name() for f in self.features]
		if len(feature_names) != len(set(feature_names)):
			raise DuplicateFeature

	def _get_interaction_type(self, features: Sequence[Feature]) -> InteractionType:
		types = [
			InteractionType.DENSE if isinstance(f, DenseFeature)
			else InteractionType.SPARSE
			for f in features
		]
		types = list(set(types))
		if len(types) > 1:
			return InteractionType.MIXED
		return types[0]


class FeatureCollection:
	def __init__(
		self,
		features: List[Feature],
		interactions: List[Interaction],
	):
		self._features = features
		self._interactions = interactions

		self._check_input()

		self._feature_names = [f.get_name() for f in self._features]
		self._dense_indices = [index for index, f in enumerate(self._features) if isinstance(f, DenseFeature)]
		self._sparse_indices = [index for index, f in enumerate(self._features) if isinstance(f, SparseFeature)]
		self._interactions_indices = [
			tuple([
				self._feature_names.index(f.get_name()) for f in i.features
			]) for i in self._interactions
		]

		self._inputs = [
			Input(shape=1)
			if isinstance(f, DenseFeature)
			else Input(shape=f.m, dtype=tf.int32)
			for f in self._features
		]

	def _check_input(self):
		if not len(self._features):
			raise NoFeatureProvided

		all_names_from_features = [f.get_name() for f in self._features]
		all_names_from_interactions = reduce(
			lambda arg1, arg2: arg1 + arg2,
			[interaction.get_feature_names() for interaction in self._interactions]) \
			if len(self._interactions) else []

		if len(all_names_from_features) != len(set(all_names_from_features)):
			raise DuplicateFeature

		unexpected_features = set(all_names_from_interactions).difference(all_names_from_features)
		if unexpected_features:
			raise UnexpectedFeatureInInteractions

	def model_inputs_schema(self) -> List[str]:
		return [f.get_full_name() for f in self._features]

	def get_dense_features(self) -> List[DenseFeature]:
		return [f for f in self._features if isinstance(f, DenseFeature)]

	def get_sparse_features(self) -> List[SparseFeature]:
		return [f for f in self._features if isinstance(f, SparseFeature)]

	def get_dense_inputs(self) -> List[Input]:
		return [i for index, i in enumerate(self._inputs) if isinstance(self._features[index], DenseFeature)]

	def get_sparse_inputs(self) -> List[Input]:
		return [i for index, i in enumerate(self._inputs) if isinstance(self._features[index], SparseFeature)]

	def get_interactions(self) -> Generator[Tuple[Interaction, List[Input]], None, None]:
		for i, interaction in enumerate(self._interactions):
			yield interaction, [self._inputs[f_index] for f_index in self._interactions_indices[i]]

	def get_inputs(self) -> List[Input]:
		return self._inputs
