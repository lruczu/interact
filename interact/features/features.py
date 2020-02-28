from functools import reduce
from typing import List, Tuple

from tensorflow import keras

from interact.exceptions import DuplicateFeature, UnexpectedFeatureInInteractions, InteractionWithOnlyOneVariable


class Feature:
	def get_name(self) -> str:
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
		return f'{self.name}: {self.dtype} of shape (None, 1)'

	def __eq__(self, other):
		if not isinstance(other, Feature):
			return False
		return self.get_name() == other.get_name()

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
		self.name = name
		self.vocabulary_size = vocabulary_size
		self.m = m
		self.dtype = dtype
		self.group = group

	def get_name(self) -> str:
		return f'{self.name}: {self.dtype} of shape (None, {self.n_levels})'

	def __eq__(self, other):
		if not isinstance(other, Feature):
			return False
		return self.get_name() == other.get_name()

	def __repr__(self):
		return str(self.__dict__)


class DenseFeatureCollection:
	def __init__(self, features: List[DenseFeature]):
		self.features = features


class SparseFeatureCollection:
	def __init__(self, features: List[SparseFeature]):
		self.features = features


class FeatureCollection:
	def __init__(
		self,
		features: List[Feature],
		interactions: List[Tuple[Feature, ...]],
	):
		self.features = features
		self.interactions = interactions

		self._check_input()

	def _check_input(self):
		all_names_from_features = [f.get_name() for f in self.features]

		all_names_from_interactions = reduce(
			lambda arg1, arg2: arg1 + arg2,
			[[f.get_name() for f in interaction] for interaction in self.interactions]) \
			if len(self.interactions) else []

		if len(all_names_from_features) != len(set(all_names_from_features)):
			raise DuplicateFeature

		unexpected_features = set(all_names_from_interactions).difference(all_names_from_features)
		if unexpected_features:
			raise UnexpectedFeatureInInteractions

		for interaction in self.interactions:
			if len(interaction) < 2:
				raise InteractionWithOnlyOneVariable

	def model_inputs_schema(self) -> List[str]:
		return [f.get_name() for f in self.features]
