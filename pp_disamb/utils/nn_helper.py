import tensorflow as tf


def noise_shape(tensor, shape):
    output_shape = list()
    tensor_shape = tf.shape(tensor)
    for i, noise_dim in enumerate(shape):
        if noise_dim is None:
            output_shape.append(tensor_shape[i])
        else:
            output_shape.append(noise_dim)
    return output_shape


def dropout_wrapper(is_traning, tensor, keep_prob, noise_shape=None, seed=None, name=None):
    return tf.cond(is_traning, lambda: tf.nn.dropout(tensor, keep_prob, noise_shape, seed, name), lambda: tensor)


def keep_prob_wrapper(is_training, keep_prob):
    return tf.cond(is_training, lambda: tf.constant(keep_prob), lambda: tf.constant(1.))


def recurrent_dropout_wrapper(is_training, cell, input_keep_prob=1.0, output_keep_prob=1.0, state_keep_prob=1.0,
                              variational_recurrent=False, input_size=None, dtype=None, seed=None,
                              dropout_state_filter_visitor=None):
    if input_keep_prob < 1.:
        input_keep_prob = keep_prob_wrapper(is_training, input_keep_prob)
    if output_keep_prob < 1.:
        output_keep_prob = keep_prob_wrapper(is_training, output_keep_prob)
    if state_keep_prob < 1.:
        state_keep_prob = keep_prob_wrapper(is_training, state_keep_prob)
    return tf.nn.rnn_cell.DropoutWrapper(cell,
                                         input_keep_prob=input_keep_prob,
                                         output_keep_prob=output_keep_prob,
                                         state_keep_prob=state_keep_prob,
                                         variational_recurrent=variational_recurrent,
                                         input_size=input_size,
                                         dtype=dtype,
                                         seed=seed,
                                         dropout_state_filter_visitor=dropout_state_filter_visitor)


def softmax(x, mask=None, axis=-1):
    x_max = tf.reduce_max(x, axis=axis, keepdims=True)
    x_dev = x - x_max
    e_x = tf.exp(x_dev - x_max)
    if mask is not None:
        e_x *= tf.to_float(mask)
    s = tf.reduce_sum(e_x, axis=axis, keepdims=True)

    if mask is not None:
        return tf.where(mask, e_x / s, tf.zeros_like(x))
    else:
        return e_x / s


def softmax_cross_entropy_with_logits(labels, logits, mask=None, axis=-1):
    if mask is None:
        mask = tf.ones_like(logits, dtype=tf.bool)

    x_max = tf.reduce_max(logits, axis=axis, keepdims=True)
    x_dev = logits - x_max
    x_dev = tf.where(mask, x_dev, tf.ones_like(x_dev) * float('-inf'))
    log_softmax = x_dev - tf.reduce_logsumexp(x_dev, axis=axis, keepdims=True)
    log_softmax = tf.where(mask, log_softmax, tf.zeros_like(log_softmax))
    return -tf.reduce_sum(labels * log_softmax, axis=axis)


def sigmoid_crossentropy(gold, pred, mask=None):
    entropy = tf.maximum(pred, 0) - pred * gold + tf.nn.softplus(-tf.abs(pred))
    if mask is not None:
        mask = tf.cast(mask, tf.float32)
        entropy *= mask
    else:
        mask = tf.ones_like(gold, dtype='float32')
    batch_size = tf.reduce_sum(mask)

    return tf.reduce_sum(entropy) / batch_size
