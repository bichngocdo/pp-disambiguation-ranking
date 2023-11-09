import tensorflow as tf


class PPDisambiguator(object):
    def __init__(self, args):
        self._build_embeddings(args)

        self.train = self._build_train_function(args)
        self.eval = self._build_eval_function(args)

        self.make_train_summary = self._build_train_summary_function()
        self.make_dev_summary = self._build_dev_summary_function()

    def initialize_global_variables(self, session):
        session.run(tf.global_variables_initializer(),
                    feed_dict={self.word_pt_embeddings_ph: self._word_pt_embeddings,
                               self.tag_pt_embeddings_ph: self._tag_pt_embeddings})

    def _build_train_summary_function(self):
        with tf.variable_scope('train_summary/'):
            x_clf_acc = tf.placeholder(tf.float32,
                                       shape=None,
                                       name='x_clf_acc')
            x_att_acc = tf.placeholder(tf.float32,
                                       shape=None,
                                       name='x_att_acc')

            tf.summary.scalar('train_clf_acc', x_clf_acc, collections=['train_summary'])
            tf.summary.scalar('train_att_acc', x_att_acc, collections=['train_summary'])

            summary = tf.summary.merge_all(key='train_summary')

        def f(session, clf_acc, att_acc):
            feed_dict = {
                x_clf_acc: clf_acc,
                x_att_acc: att_acc,
            }
            return session.run(summary, feed_dict=feed_dict)

        return f

    def _build_dev_summary_function(self):
        with tf.variable_scope('dev_summary/'):
            x_loss = tf.placeholder(tf.float32,
                                    shape=None,
                                    name='x_loss')
            x_clf_acc = tf.placeholder(tf.float32,
                                       shape=None,
                                       name='x_clf_acc')
            x_att_acc = tf.placeholder(tf.float32,
                                       shape=None,
                                       name='x_att_acc')

            tf.summary.scalar('dev_loss', x_loss, collections=['dev_summary'])
            tf.summary.scalar('dev_clf_acc', x_clf_acc, collections=['dev_summary'])
            tf.summary.scalar('dev_att_acc', x_att_acc, collections=['dev_summary'])

            summary = tf.summary.merge_all(key='dev_summary')

        def f(session, loss, clf_acc, att_acc):
            feed_dict = {
                x_loss: loss,
                x_clf_acc: clf_acc,
                x_att_acc: att_acc,
            }
            return session.run(summary, feed_dict=feed_dict)

        return f

    def _variable_summary(self, var):
        var_scope = var.name.split(':')[0]
        with tf.name_scope(var_scope + '/summaries'):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean', mean, collections=['train_instant_summary'])
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev', stddev, collections=['train_instant_summary'])
            tf.summary.scalar('max', tf.reduce_max(var), collections=['train_instant_summary'])
            tf.summary.scalar('min', tf.reduce_min(var), collections=['train_instant_summary'])
            tf.summary.histogram('histogram', var, collections=['train_instant_summary'])

    def _build_scope_summary(self):
        for var in tf.trainable_variables(scope=tf.get_variable_scope().name):
            self._variable_summary(var)

    def _build_placeholders(self):
        x_word = tf.placeholder(tf.int32,
                                shape=(None, 1),
                                name='x_word')
        x_pt_word = tf.placeholder(tf.int32,
                                   shape=(None, 1),
                                   name='x_pt_word')
        x_tag = tf.placeholder(tf.int32,
                               shape=(None, 1),
                               name='x_tag')
        x_pt_tag = tf.placeholder(tf.int32,
                                  shape=(None, 1),
                                  name='x_pt_tag')
        x_topo = tf.placeholder(tf.int32,
                                shape=(None, 1),
                                name='x_topo')
        return [x_word, x_pt_word, x_tag, x_pt_tag, x_topo]

    def _build_candidate_placeholders(self):
        x_word = tf.placeholder(tf.int32,
                                shape=(None, None),
                                name='x_word')
        x_pt_word = tf.placeholder(tf.int32,
                                   shape=(None, None),
                                   name='x_pt_word')
        x_tag = tf.placeholder(tf.int32,
                               shape=(None, None),
                               name='x_tag')
        x_pt_tag = tf.placeholder(tf.int32,
                                  shape=(None, None),
                                  name='x_pt_tag')
        x_topo = tf.placeholder(tf.int32,
                                shape=(None, None),
                                name='x_topo')
        x_abs_dist = tf.placeholder(tf.int32,
                                    shape=(None, None),
                                    name='x_abs_dist')
        x_rel_dist = tf.placeholder(tf.int32,
                                    shape=(None, None),
                                    name='x_rel_dist')
        y_label = tf.placeholder(tf.float32,
                                 shape=(None, None),
                                 name='y_label')

        return [x_word, x_pt_word, x_tag, x_pt_tag, x_topo, x_abs_dist, x_rel_dist], y_label

    def _build_embeddings(self, args):
        with tf.variable_scope('embeddings'):
            self.word_embeddings = tf.get_variable('word_embeddings',
                                                   shape=(args.no_words, args.word_dim),
                                                   dtype=tf.float32,
                                                   initializer=tf.zeros_initializer,
                                                   regularizer=tf.contrib.layers.l2_regularizer(args.word_l2))

            self.word_pt_embeddings_ph = tf.placeholder(tf.float32,
                                                        shape=args.word_embeddings.shape,
                                                        name='word_pt_embeddings_ph')
            self.word_pt_embeddings = tf.Variable(self.word_pt_embeddings_ph,
                                                  name='word_pt_embeddings',
                                                  trainable=False)
            self._word_pt_embeddings = args.word_embeddings

            self.tag_embeddings = tf.get_variable('tag_embeddings',
                                                  shape=(args.no_tags, args.tag_dim),
                                                  dtype=tf.float32,
                                                  initializer=tf.zeros_initializer,
                                                  regularizer=tf.contrib.layers.l2_regularizer(args.tag_l2))
            self.tag_pt_embeddings_ph = tf.placeholder(tf.float32,
                                                       shape=args.tag_embeddings.shape,
                                                       name='tag_pt_embeddings_ph')
            self.tag_pt_embeddings = tf.Variable(self.tag_pt_embeddings_ph,
                                                 name='tag_pt_embeddings',
                                                 trainable=False)
            self._tag_pt_embeddings = args.tag_embeddings

    def _build_input_layers(self, args, is_training):
        def f(x_word, x_pt_word, x_tag, x_pt_tag, x_topo):
            word_repr = tf.nn.embedding_lookup(self.word_embeddings, x_word)
            word_repr += tf.nn.embedding_lookup(self.word_pt_embeddings, x_pt_word)
            # if is_training:
            #     word_repr = tf.nn.dropout(word_repr,
            #                               keep_prob=1 - args.input_dropout)

            tag_repr = tf.nn.embedding_lookup(self.tag_embeddings, x_tag)
            tag_repr += tf.nn.embedding_lookup(self.tag_pt_embeddings, x_pt_tag)
            # if is_training:
            #     tag_repr = tf.nn.dropout(tag_repr,
            #                              keep_prob=1 - args.input_dropout)

            topo_repr = tf.one_hot(x_topo, depth=args.no_topological_fields)

            input = tf.concat([word_repr, tag_repr, topo_repr], axis=-1)
            return input

        return f

    def _build_candidate_input_layers(self, args, is_training):
        def f(x_word, x_pt_word, x_tag, x_pt_tag, x_topo, x_abs_dist, x_rel_dist):
            word_repr = tf.nn.embedding_lookup(self.word_embeddings, x_word)
            word_repr += tf.nn.embedding_lookup(self.word_pt_embeddings, x_pt_word)
            # if is_training:
            #     word_repr = tf.nn.dropout(word_repr,
            #                               keep_prob=1 - args.input_dropout)

            tag_repr = tf.nn.embedding_lookup(self.tag_embeddings, x_tag)
            tag_repr += tf.nn.embedding_lookup(self.tag_pt_embeddings, x_pt_tag)
            # if is_training:
            #     tag_repr = tf.nn.dropout(tag_repr,
            #                              keep_prob=1 - args.input_dropout)

            topo_repr = tf.one_hot(x_topo, depth=args.no_topological_fields)
            abs_dist = tf.to_float(tf.expand_dims(x_abs_dist, -1))
            log_abs_dist = tf.where(tf.greater(abs_dist, 0), tf.log(abs_dist), -tf.log(-abs_dist))
            log_abs_dist = tf.where(tf.equal(abs_dist, 0), tf.zeros_like(abs_dist), log_abs_dist)
            rel_dist = tf.to_float(tf.expand_dims(x_rel_dist, -1))

            input = tf.concat([word_repr, tag_repr, topo_repr, log_abs_dist, rel_dist], axis=-1)
            return input

        return f

    def _build_scoring_model(self, args, is_training):
        def f(pre_repr, obj_repr, can_repr):
            with tf.variable_scope('mlp', reuse=tf.AUTO_REUSE):
                with tf.variable_scope('input_transform'):
                    input = tf.concat([tf.tile(pre_repr, (1, tf.shape(can_repr)[1], 1)),
                                       tf.tile(obj_repr, (1, tf.shape(can_repr)[1], 1)),
                                       can_repr], axis=-1)

                with tf.variable_scope('layer1'):
                    if is_training:
                        input = tf.nn.dropout(input, keep_prob=1 - args.input_dropout)
                    hidden = tf.layers.dense(input,
                                             units=args.hidden_dim,
                                             kernel_regularizer=tf.contrib.layers.l2_regularizer(args.l2))
                    if args.batch_normalization and is_training:
                        hidden = tf.layers.batch_normalization(hidden, training=is_training)

                    if args.debug:
                        tf.summary.histogram('pre_activation', hidden, collections=['train_instant_summary'])

                    hidden = tf.nn.relu(hidden)

                    if args.debug:
                        tf.summary.histogram('activation', hidden, collections=['train_instant_summary'])

                with tf.variable_scope('layer2'):
                    if is_training:
                        hidden = tf.nn.dropout(hidden, keep_prob=1 - args.dropout)
                    output = tf.layers.dense(hidden,
                                             units=1,
                                             kernel_regularizer=tf.contrib.layers.l2_regularizer(args.l2))

                    if args.debug:
                        tf.summary.histogram('pre_activation', hidden, collections=['train_instant_summary'])

                    output = tf.squeeze(output, axis=-1)

                return output

        return f

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
            mask_float = tf.to_float(mask)

        logit = scoring_model(pre_repr, obj_repr, can_repr)

        with tf.variable_scope('output_layer'):
            probability = tf.sigmoid(logit)
            probability = tf.where(mask, probability, tf.zeros_like(probability) - 1)
            rounded = tf.where(tf.greater_equal(probability, .5),
                               tf.ones_like(probability),
                               tf.zeros_like(probability))
            rounded = tf.where(mask, rounded, tf.zeros_like(label) - 1)
            prediction = tf.argmax(probability, axis=-1, output_type=tf.int32)

        with tf.variable_scope('loss'):
            cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=label, logits=logit)
            cross_entropy *= mask_float
            loss = tf.reduce_sum(cross_entropy) / tf.reduce_sum(mask_float)

        inputs = pre_input + obj_input + can_input + [label]
        outputs = {
            'logit': logit,
            'probability': probability,
            'rounded': rounded,
            'prediction': prediction
        }

        return inputs, outputs, loss

    def _build_train_function(self, args):
        with tf.name_scope('train'):
            inputs, outputs, loss = self._build(args, is_training=True)

            self.iteration = tf.Variable(-1, name='iteration', trainable=False)
            with tf.variable_scope('learning_rate'):
                self.learning_rate = tf.train.exponential_decay(learning_rate=args.learning_rate,
                                                                global_step=self.iteration,
                                                                decay_steps=args.decay_step,
                                                                decay_rate=args.decay_rate)

            with tf.variable_scope('train_summary/'):
                tf.summary.scalar('lr', self.learning_rate, collections=['train_instant_summary'])
                tf.summary.scalar('train_loss', loss, collections=['train_instant_summary'])
                summary = tf.summary.merge_all(key='train_instant_summary')

            if args.debug:
                self._build_scope_summary()

            optimizer = tf.train.AdagradOptimizer(self.learning_rate)
            gradients_vars = optimizer.compute_gradients(loss, tf.trainable_variables())
            if args.max_norm is not None:
                with tf.variable_scope('max_norm'):
                    gradients = [gv[0] for gv in gradients_vars]
                    gradients, _ = tf.clip_by_global_norm(gradients, args.max_norm)
                    gradients_vars = [(g, gv[1]) for g, gv in zip(gradients, gradients_vars)]

            with tf.variable_scope('optimizer'):
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                with tf.control_dependencies(update_ops):
                    train_step = optimizer.apply_gradients(gradients_vars, global_step=self.iteration)

        def f(vars, session):
            feed_dict = dict()
            for input, var in zip(inputs, vars):
                feed_dict[input] = var
            _, output_, loss_, summary_ = session.run([train_step, outputs, loss, summary],
                                                      feed_dict=feed_dict)
            return output_, loss_, summary_

        return f

    def _build_eval_function(self, args):
        with tf.name_scope('eval'):
            inputs, outputs, loss = self._build(args, is_training=False)

        def f(vars, session):
            feed_dict = dict()
            for input, var in zip(inputs, vars):
                feed_dict[input] = var
            return session.run([outputs, loss], feed_dict=feed_dict)

        return f
