import tensorflow as tf


def second_order_interactions(
    X: tf.Tensor,
    V: tf.Tensor,
):
    """Computes second order interactions.

    Args:
        X: (m, n_f, K)
        V: of shape (n_f, K)

    Returns:

    """
    rows_sum = tf.matmul(X, V) ** 2
    squared_rows_sum = tf.matmul(X * X, V * V)

    return 0.5 * tf.reduce_sum(rows_sum - squared_rows_sum, axis=1, keepdims=True)
