import abc

import tensorflow as tf
from tensorflow.python.keras import layers
from tensorflow.python.keras import initializers


class FM(abc.ABC):
    """Factorization Machine.

        Model described in the paper assumes the following form.
        y(x) = w_{0} + sum_{i=1}^{i=n}w_{i}x_{i} + sum_{i=1}^{n}sum_{i+1}^{n}<v_{i}, v_{j}>x_{i}x_{j}

        In the layer implemented here, the focus in only put on (2-order) interactions.
        So applying this layer on x will give:
        sum_{i=1}^{n}sum_{i+1}^{n}<v_{i}, v_{j}>x_{i}x_{j},
        where x_{i} and x{j} are i-th and j-th component of x vector, respectively and
        v_{i} and v_{j} are associated k-dimensional vectors modeling interactions.


        References:
        - [Factorization Machines](https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf)
        """
    pass


class DenseFM(layers.Layer, FM):
    def __init__(self, k: int, **kwargs):
        self.k = k
        super(DenseFM, self).__init__(**kwargs)

    def build(self, input_shape: tf.TensorShape):
        if input_shape.ndims != 2:
            raise ValueError('Unexpected input shape.')

        n_features = int(input_shape[1])

        if n_features < 1:
            raise ValueError('At least two features are needed to interact.')

        self.V = self.add_weight('V',
                                 shape=[n_features, self.k],
                                 initializer=initializers.glorot_uniform())

    def call(self, inputs):
        if isinstance(inputs, tf.SparseTensor):
            raise ValueError('Required to pass Tensor type.')

        if inputs.shape.ndims != 2:
            raise ValueError('Unexpected input shape.')

        rows_sum = tf.matmul(inputs, self.V) ** 2
        squaured_rows_sum = tf.matmul(inputs * inputs, self.V * self.V)

        return 0.5 * tf.reduce_sum(rows_sum - squaured_rows_sum, axis=1, keepdims=True)


class DenseFM2(layers.Layer, FM):
    def __init__(self, k: int, **kwargs):
        self.k = k
        super(DenseFM, self).__init__(**kwargs)

    def build(self, input_shape1: tf.TensorShape, input_shape2: tf.TensorShape):
        self._check_dimensions(input_shape1)
        self._check_dimensions(input_shape2)

        self.V = self.add_weight('V',
                                 shape=[2, self.k],
                                 initializer=initializers.glorot_uniform())

    def call(self, input1, input2):
        if isinstance(inputs, tf.SparseTensor):
            raise ValueError('Required to pass Tensor type.')
        self._check_dimensions(input1.shape)
        self._check_dimensions(input2.shape)
        
        rows_sum = tf.matmul(inputs, self.V) ** 2
        squaured_rows_sum = tf.matmul(inputs * inputs, self.V * self.V)

        return 0.5 * tf.reduce_sum(rows_sum - squaured_rows_sum, axis=1, keepdims=True)
    
    def _check_dimensions(self, shape: tf.TensorShape):
        if shape.ndims != 2 or int(shape[1]) != 1:
            raise ValueError('Unexpected input shape.')


class SparseFM(layers.Layer, FM):
    def __init__(self, k: int, m_non_zero, **kwargs):
        self.k = k
        self.m_non_zero = m_non_zero
        super(SparseFM, self).__init__(**kwargs)

    def build(self, input_shape: tf.TensorShape):
        if input_shape.ndims != 2:
            raise ValueError('Unexcepted input shape.')

        n_features = int(input_shape[1])

        if n_features < 1:
            raise ValueError('At least two features are needed to interact.')

        self.V = self.add_weight('V',
                                 shape=[n_features, self.k],
                                 initializer=initializers.glorot_uniform())

        self.I = tf.matmul(self.V, tf.transpose(self.V))

    def call(self, inputs):
        if isinstance(inputs, tf.SparseTensor):
            raise ValueError('Required to pass SparseTensor type.')
        if inputs.shape.ndims != 2:
            raise ValueError('Unexcepted input shape.')

        self.g = inputs

a = DenseFM([feature1, feature2])
