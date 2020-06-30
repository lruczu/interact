class Field:
	def get_name(self) -> str:
		raise NotImplementedError

	def get_full_name(self) -> str:
		raise NotImplementedError


class DenseField(Field):
	def __init__(
		self, 
		name: str,
		d: int,
		dtype: str = 'float32',
	):
		self.name = name
		self.d = int(d)
		self.dtype = dtype

	def get_name(self) -> str:
		return self.name

	def get_full_name(self) -> str:
		return f'{self.name}: {self.dtype} of shape (None, 1)'

	def __eq__(self, other):
		if not isinstance(other, Field):
			return False
		return self.get_full_name() == other.get_full_name()

	def __repr__(self):
		return str(self.__dict__)


class SparseField(Field):
	def __init__(
		self, 
		name: str, 
		vocabulary_size: int,
		m: int,
		d: int,
		dtype: str = 'int32',
	):
		if vocabulary_size < m:
			raise ValueError('m cannot be bigger than vocabulary size.')
		self.name = name
		self.vocabulary_size = int(vocabulary_size)
		self.m = int(m)
		self.d = int(d)
		self.dtype = dtype

	def get_name(self) -> str:
		return self.name

	def get_full_name(self) -> str:
		return f'{self.name}: {self.dtype} of shape (None, {self.m})'

	def __eq__(self, other):
		if not isinstance(other, Field):
			return False
		return self.get_full_name() == other.get_full_name()

	def __repr__(self):
		return str(self.__dict__)
