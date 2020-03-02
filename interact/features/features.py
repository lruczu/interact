import enum
from functools import reduce
from typing import List, Seq, Tuple

import tensorflow as tf
from tensorflow.keras.layers import Input

from interact.exceptions import (
	DuplicateFeature,
	InteractionWithOnlyOneVariable,
	NoFeatureProvided,
	UnexpectedFeatureInInteractions
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
		group: str = 'default',
	):
		if vocabulary_size < m:
			raise ValueError('m cannot be bigger than vocabulary size.')
		self.name = name
		self.vocabulary_size = vocabulary_size
		self.m = m
		self.dtype = dtype
		self.group = group

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


class FeatureCollection:
	def __init__(
		self,
		features: List[Feature],
		interactions: List[Tuple[Feature, ...]],
	):
		self._features = features
		self._interactions = interactions

		self._check_input()

		self.global_features_type: InteractionType = self._get_features_type(features)

		self._feature_names = [f.get_name() for f in self._features]
		self._dense_indices = [index for index, f in enumerate(self._features) if isinstance(f, DenseFeature)]
		self._sparse_indices = [index for index, f in enumerate(self._features) if isinstance(f, SparseFeature)]
		self._interactions_indices = [
			tuple([
				self._feature_names.index(f.get_name()) for f in i
			]) for i in self._interactions
		]
		self._interactions_types = [
			self._get_features_type(i) for i in self._interactions
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
			[[f.get_name() for f in interaction] for interaction in self._interactions]) \
			if len(self._interactions) else []

		if len(all_names_from_features) != len(set(all_names_from_features)):
			raise DuplicateFeature

		unexpected_features = set(all_names_from_interactions).difference(all_names_from_features)
		if unexpected_features:
			raise UnexpectedFeatureInInteractions

		for interaction in self._interactions:
			if len(interaction) < 2:
				raise InteractionWithOnlyOneVariable

	def model_inputs_schema(self) -> List[str]:
		return [f.get_full_name() for f in self._features]

	def get_dense_features(self) -> List[DenseFeature]:
		return [f for f in self._features if isinstance(f, DenseFeature)]

	def get_sparse_features(self) -> List[SparseFeature]:
		return [f for f in self._features if isinstance(f, SparseFeature)]

	def get_interactions(self) -> List[Tuple[InteractionType, Tuple[Feature], Tuple[Input]]]:
		for t, indices in zip(self._interactions_types, self._interactions_indices):
			yield t, [self._inputs[index] for index in indices]

	def get_inputs(self) -> List[Input]:
		return self._inputs

	def _get_features_type(self, features: Seq[Feature]):
		types = [
			InteractionType.DENSE if isinstance(f, DenseFeature)
			else InteractionType.SPARSE
			for f in features
		]
		types = list(set(types))
		if len(types) > 1:
			return InteractionType.MIXED
		return types[0]
