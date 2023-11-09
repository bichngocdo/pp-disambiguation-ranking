import tensorflow as tf

import pp_disamb.german_baseline.network_cross_entropy
from pp_disamb.utils.nn_helper import softmax
from pp_disamb.utils.ranking_loss import ranking_loss_all


class PPDisambiguator(pp_disamb.german_baseline.network_cross_entropy.PPDisambiguator):
    def __init__(self, args):
        super(PPDisambiguator, self).__init__(args)

    def _build(self, args, is_training):
        with tf.variable_scope('preposition'):
            pre_input = self._build_placeholders()
        with tf.variable_scope('object'):
            obj_input = self._build_placeholders()
        with tf.variable_scope('candidate'):
            can_input, label = self._build_candidate_placeholders()

        input_layers = self._build_input_layers(args, is_training=is_training)
        candidate_input_layers = self._build_candidate_input_layers(args, is_training=is_training)
        scoring_model = self._build_scoring_model(args, is_training=is_training)

        with tf.variable_scope('preposition_input_layers'):
            pre_repr = input_layers(*pre_input)
        with tf.variable_scope('object_input_layers'):
            obj_repr = input_layers(*obj_input)
        with tf.variable_scope('candidate_input_layers'):
            can_repr = candidate_input_layers(*can_input)

        with tf.variable_scope('mask'):
            mask = tf.greater_equal(label, 0)

        logit = scoring_model(pre_repr, obj_repr, can_repr)

        with tf.variable_scope('output_layer'):
            probability = softmax(logit, mask)
            probability = tf.where(mask, probability, tf.zeros_like(probability) - 1)
            rounded = tf.to_float(tf.greater_equal(probability, tf.reduce_max(probability, axis=-1, keepdims=True)))
            rounded = tf.where(mask, rounded, tf.zeros_like(label) - 1)
            prediction = tf.argmax(probability, axis=-1, output_type=tf.int32)

        with tf.variable_scope('loss'):
            loss = ranking_loss_all(logit, label)

        inputs = pre_input + obj_input + can_input + [label]
        outputs = {
            'logit': logit,
            'probability': probability,
            'rounded': rounded,
            'prediction': prediction
        }

        return inputs, outputs, loss
