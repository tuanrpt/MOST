#  Copyright (c) 2021, Tuan Nguyen.
#  All rights reserved.

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import tensorflow as tf
from tensorflow.contrib.framework import arg_scope
from tensorflow.contrib.framework import add_arg_scope
from tensorbayes.layers import dense, conv2d, batch_norm, instance_norm
from tensorbayes.tfutils import softmax_cross_entropy_with_two_logits as softmax_x_entropy_two

from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator

from generic_utils import random_seed

from layers import leaky_relu
import os
from generic_utils import model_dir
import numpy as np
import tensorbayes as tb
from layers import batch_ema_acc
from keras.utils.np_utils import to_categorical
from alexnet.model import AlexNetModel


def build_block(input_layer, layout, info=1):
    x = input_layer
    for i in range(0, len(layout)):
        with tf.variable_scope('l{:d}'.format(i)):
            f, f_args, f_kwargs = layout[i]
            x = f(x, *f_args, **f_kwargs)
            if info > 1:
                print(x)
    return x


@add_arg_scope
def normalize_perturbation(d, scope=None):
    with tf.name_scope(scope, 'norm_pert'):
        output = tf.nn.l2_normalize(d, axis=np.arange(1, len(d.shape)))
    return output


def build_encode_template(
        input_layer, training_phase, scope, encode_layout,
        reuse=None, internal_update=False, getter=None, inorm=True, cnn_size='large'):
    with tf.variable_scope(scope, reuse=reuse, custom_getter=getter):
        with arg_scope([leaky_relu], a=0.1), \
             arg_scope([conv2d, dense], activation=leaky_relu, bn=True, phase=training_phase), \
             arg_scope([batch_norm], internal_update=internal_update):

            preprocess = instance_norm if inorm else tf.identity

            layout = encode_layout(preprocess=preprocess, training_phase=training_phase, cnn_size=cnn_size)
            output_layer = build_block(input_layer, layout)

    return output_layer


def build_class_discriminator_template(
        input_layer, training_phase, scope, num_classes, class_discriminator_layout,
        reuse=None, internal_update=False, getter=None, cnn_size='large'):
    with tf.variable_scope(scope, reuse=reuse, custom_getter=getter):
        with arg_scope([leaky_relu], a=0.1), \
             arg_scope([conv2d, dense], activation=leaky_relu, bn=True, phase=training_phase), \
             arg_scope([batch_norm], internal_update=internal_update):
            layout = class_discriminator_layout(num_classes=num_classes, global_pool=True, activation=None,
                                                cnn_size=cnn_size)
            output_layer = build_block(input_layer, layout)

    return output_layer


def build_domain_discriminator_template(x, domain_layout, c=1, reuse=None, scope='domain_disc'):
    with tf.variable_scope(scope, reuse=reuse):
        with arg_scope([dense], activation=tf.nn.relu):
            layout = domain_layout(c=c)
            output_layer = build_block(x, layout)

    return output_layer


def build_phi_network_template(x, domain_layout, c=1, reuse=None):
    with tf.variable_scope('phi_net', reuse=reuse):
        with arg_scope([dense], activation=tf.nn.relu):
            layout = domain_layout(c=c)
            output_layer = build_block(x, layout)

    return output_layer


def get_default_config():
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    tf_config.log_device_placement = False
    tf_config.allow_soft_placement = True
    return tf_config


class MOST():
    def __init__(self,
                 model_name="MOST-results",
                 learning_rate=0.001,
                 batch_size=128,
                 num_iters=80000,
                 summary_freq=400,
                 src_class_trade_off=1.0,
                 src_domain_trade_off='1.0,1.0',
                 src_vat_trade_off=1.0,
                 trg_vat_troff=0.1,
                 trg_ent_troff=0.1,
                 ot_trade_off=0.1,
                 domain_trade_off=0.1,
                 mimic_trade_off=0.1,
                 encode_layout=None,
                 classify_layout=None,
                 domain_layout=None,
                 phi_layout=None,
                 current_time='',
                 inorm=True,
                 theta=0.1,
                 g_network_trade_off=1.0,
                 mdaot_model_id='',
                 only_save_final_model=True,
                 cnn_size='large',
                 sample_size=50,
                 data_shift_troff=10.0,
                 lbl_shift_troff=1.0,
                 num_classes=10,
                 multi_scale='',
                 resnet_depth=101,
                 train_layers='fc8,fc7,fc6',
                 **kwargs):
        self.model_name = model_name
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_iters = num_iters
        self.summary_freq = summary_freq
        self.src_class_trade_off = src_class_trade_off
        self.src_domain_trade_off = [float(item) for item in src_domain_trade_off.split(',')]
        self.src_vat_trade_off = src_vat_trade_off
        self.trg_vat_troff = trg_vat_troff
        self.trg_ent_troff = trg_ent_troff
        self.ot_trade_off = ot_trade_off
        self.domain_trade_off = domain_trade_off
        self.mimic_trade_off = mimic_trade_off

        self.encode_layout = encode_layout
        self.classify_layout = classify_layout
        self.domain_layout = domain_layout
        self.phi_layout = phi_layout

        self.current_time = current_time
        self.inorm = inorm

        self.theta = theta
        self.g_network_trade_off = g_network_trade_off

        self.mdaot_model_id = mdaot_model_id
        self.only_save_final_model = only_save_final_model

        self.cnn_size = cnn_size
        self.sample_size = sample_size
        self.data_shift_troff = data_shift_troff
        self.lbl_shift_troff = lbl_shift_troff

        self.num_classes = num_classes
        self.multi_scale = multi_scale
        self.resnet_depth = resnet_depth
        self.train_layers = train_layers.split(',')

    def _init(self, src_preprocessors, trg_train_preprocessor, trg_test_preprocessor, num_src_domain):
        np.random.seed(random_seed())
        tf.set_random_seed(random_seed())
        tf.reset_default_graph()

        self.tf_graph = tf.get_default_graph()
        self.tf_config = get_default_config()
        self.tf_session = tf.Session(config=self.tf_config, graph=self.tf_graph)

        self.src_preprocessors = src_preprocessors
        self.trg_train_preprocessor = trg_train_preprocessor
        self.trg_test_preprocessor = trg_test_preprocessor
        self.num_src_domain = num_src_domain
        self.batch_size_src = self.sample_size*self.num_classes

        assert len(self.src_domain_trade_off) == self.num_src_domain
        assert self.sample_size*self.num_classes*self.num_src_domain == self.batch_size

    def _get_variables(self, list_scopes):
        variables = []
        for scope_name in list_scopes:
            variables.append(tf.get_collection('trainable_variables', scope_name))
        return variables

    def convert_one_hot(self, y):
        y_idx = y.reshape(-1).astype(int) if y is not None else None
        y = np.eye(self.num_classes)[y_idx] if y is not None else None
        return y

    def _get_scope(self, part_name, side_name, same_network=True):
        suffix = ''
        if not same_network:
            suffix = '/' + side_name
        return part_name + suffix

    def _get_all_g_h(self):
        return ['generator', 'classifier']

    def _get_teacher_scopes(self):
        return ['generator', 'classifier', 'domain_disc']

    def _get_student_primary_scopes(self):
        return ['generator', 'c-trg']

    def _get_student_secondary_scopes(self):
        return ['phi_net']

    def _build_source_middle(self, x_src, is_reused):
        scope_name = self._get_scope('generator', 'src')
        if is_reused == 0:
            generator_model = build_encode_template(x_src, encode_layout=self.encode_layout,
                                     scope=scope_name, training_phase=self.is_training, inorm=self.inorm, cnn_size=self.cnn_size)
        else:
            generator_model = build_encode_template(x_src, encode_layout=self.encode_layout,
                                                    scope=scope_name, training_phase=self.is_training, inorm=self.inorm,
                                                    reuse=True, internal_update=True,
                                                    cnn_size=self.cnn_size)
        return generator_model

    def _build_target_middle(self, x_trg, reuse=None):
        scope_name = 'generator'
        return build_encode_template(
            x_trg, encode_layout=self.encode_layout,
            scope=scope_name, training_phase=self.is_training, inorm=self.inorm,
            reuse=reuse, internal_update=True, cnn_size=self.cnn_size
        )

    def _build_classifier(self, x, num_classes, ema=None, is_teacher=False):
        g_teacher_scope = self._get_scope('generator', 'teacher', same_network=False)
        g_x = build_encode_template(
            x, encode_layout=self.encode_layout,
            scope=g_teacher_scope if is_teacher else 'generator', training_phase=False, inorm=self.inorm,
            reuse=False if is_teacher else True, getter=None if is_teacher else tb.tfutils.get_getter(ema),
            cnn_size=self.cnn_size
        )

        h_teacher_scope = self._get_scope('c-trg', 'teacher', same_network=False)
        h_g_x = build_class_discriminator_template(
            g_x, training_phase=False, scope=h_teacher_scope if is_teacher else 'c-trg', num_classes=num_classes,
            reuse=False if is_teacher else True, class_discriminator_layout=self.classify_layout,
            getter=None if is_teacher else tb.tfutils.get_getter(ema), cnn_size=self.cnn_size
        )
        return h_g_x

    def _build_domain_discriminator(self, x_mid, reuse=None, scope='domain_disc'):
        return build_domain_discriminator_template(x_mid, domain_layout=self.domain_layout, c=self.num_src_domain, reuse=reuse, scope=scope)

    def _build_phi_network(self, x_mid, reuse=None):
        return build_phi_network_template(x_mid, domain_layout=self.phi_layout, c=1, reuse=reuse)

    def _build_class_src_discriminator(self, x_src, num_src_classes, i, reuse=None):
        classifier_model = build_class_discriminator_template(
            x_src, training_phase=self.is_training, scope='classifier/{}'.format(i), num_classes=num_src_classes,
            reuse=reuse, internal_update=True, class_discriminator_layout=self.classify_layout, cnn_size=self.cnn_size
        )
        return classifier_model

    def _build_class_trg_discriminator(self, x_trg, num_trg_classes):
        return build_class_discriminator_template(
            x_trg, training_phase=self.is_training, scope='c-trg', num_classes=num_trg_classes,
            reuse=False, internal_update=True, class_discriminator_layout=self.classify_layout, cnn_size=self.cnn_size
        )

    def perturb_image(self, x, p, num_classes, class_discriminator_layout, encode_layout,
                      pert='vat', scope=None, radius=3.5, scope_classify=None, scope_encode=None, training_phase=None):
        with tf.name_scope(scope, 'perturb_image'):
            eps = 1e-6 * normalize_perturbation(tf.random_normal(shape=tf.shape(x)))

            # Predict on randomly perturbed image
            x_eps_mid = build_encode_template(
                x + eps, encode_layout=encode_layout, scope=scope_encode, training_phase=training_phase, reuse=True,
                inorm=self.inorm, cnn_size=self.cnn_size)
            x_eps_pred = build_class_discriminator_template(
                x_eps_mid, class_discriminator_layout=class_discriminator_layout,
                training_phase=training_phase, scope=scope_classify, reuse=True, num_classes=num_classes,
                cnn_size=self.cnn_size
            )
            # eps_p = classifier(x + eps, phase=True, reuse=True)
            loss = softmax_x_entropy_two(labels=p, logits=x_eps_pred)

            # Based on perturbed image, get direction of greatest error
            eps_adv = tf.gradients(loss, [eps], aggregation_method=2)[0]

            # Use that direction as adversarial perturbation
            eps_adv = normalize_perturbation(eps_adv)
            x_adv = tf.stop_gradient(x + radius * eps_adv)

        return x_adv

    def vat_loss(self, x, p, num_classes, class_discriminator_layout, encode_layout,
                 scope=None, scope_classify=None, scope_encode=None, training_phase=None):

        with tf.name_scope(scope, 'smoothing_loss'):
            x_adv = self.perturb_image(
                x, p, num_classes, class_discriminator_layout=class_discriminator_layout, encode_layout=encode_layout,
                scope_classify=scope_classify, scope_encode=scope_encode, training_phase=training_phase)

            x_adv_mid = build_encode_template(
                x_adv, encode_layout=encode_layout, scope=scope_encode, training_phase=training_phase, inorm=self.inorm,
                reuse=True, cnn_size=self.cnn_size)
            x_adv_pred = build_class_discriminator_template(
                x_adv_mid, training_phase=training_phase, scope=scope_classify, reuse=True, num_classes=num_classes,
                class_discriminator_layout=class_discriminator_layout, cnn_size=self.cnn_size
            )
            # p_adv = classifier(x_adv, phase=True, reuse=True)
            loss = tf.reduce_mean(softmax_x_entropy_two(labels=tf.stop_gradient(p), logits=x_adv_pred))

        return loss

    def _build_vat_loss(self, x, p, num_classes, scope=None, scope_classify=None, scope_encode=None):
        return self.vat_loss(
            x, p, num_classes,
            class_discriminator_layout=self.classify_layout,
            encode_layout=self.encode_layout,
            scope=scope, scope_classify=scope_classify, scope_encode=scope_encode,
            training_phase=self.is_training
        )

    def _compute_cosine_similarity(self, x_trg_mid, x_src_mid_all):
        x_trg_mid_flatten = tf.layers.Flatten()(x_trg_mid)
        x_src_mid_all_flatten = tf.layers.Flatten()(x_src_mid_all)
        similarity = tf.reduce_sum(x_trg_mid_flatten[:, tf.newaxis] * x_src_mid_all_flatten, axis=-1)
        similarity /= tf.norm(x_trg_mid_flatten[:, tf.newaxis], axis=-1) * tf.norm(x_src_mid_all_flatten, axis=-1)
        distance = 1.0 - similarity
        return distance

    def _compute_data_shift_loss(self, x_src_mid, x_trg_mid):
        x_src_mid_flatten = tf.layers.Flatten()(x_src_mid)
        x_trg_mid_flatten = tf.layers.Flatten()(x_trg_mid)

        data_shift_loss = tf.norm(tf.subtract(x_src_mid_flatten, tf.expand_dims(x_trg_mid_flatten, 1)), axis=2)
        return data_shift_loss

    def _compute_label_shift_loss(self, y_trg_logit, y_src_teacher):
        y_trg_logit_rep = K.repeat_elements(tf.expand_dims(y_trg_logit, axis=0), rep=self.batch_size, axis=0)
        y_trg_logit_rep = tf.reshape(y_trg_logit_rep, [-1, y_trg_logit_rep.get_shape()[-1]])
        y_src_teacher_rep = K.repeat_elements(y_src_teacher, rep=self.batch_size, axis=0)
        label_shift_loss = softmax_x_entropy_two(labels=y_src_teacher_rep,
                                                                      logits=y_trg_logit_rep)

        label_shift_loss = tf.reshape(label_shift_loss, [self.batch_size, self.batch_size])
        return label_shift_loss

    def _compute_teacher_hs(self, y_label_trg_output_each_h, y_d_trg_sofmax_output):
        y_label_trg_output_each_h = tf.transpose(tf.stack(y_label_trg_output_each_h), perm=[1, 0, 2])
        y_d_trg_sofmax_output_multi_y = y_d_trg_sofmax_output
        y_d_trg_sofmax_output_multi_y = tf.expand_dims(y_d_trg_sofmax_output_multi_y, axis=-1)
        y_d_trg_sofmax_output_multi_y = tf.tile(y_d_trg_sofmax_output_multi_y, [1, 1, self.num_classes])
        y_label_trg_output = y_d_trg_sofmax_output_multi_y * y_label_trg_output_each_h
        y_label_trg_output = tf.reduce_sum(y_label_trg_output, axis=1)
        return y_label_trg_output

    def _build_model(self):
        self.x_src_lst = []
        self.y_src_lst = []
        for i in range(self.num_src_domain):
            x_src = tf.placeholder(dtype=tf.float32, shape=tuple([None]) + self.dim_src,
                                        name='x_src_{}_input'.format(i))
            y_src = tf.placeholder(dtype=tf.float32, shape=(None, self.num_classes),
                                        name='y_src_{}_input'.format(i))

            self.x_src_lst.append(x_src)
            self.y_src_lst.append(y_src)

        self.x_trg = tf.placeholder(dtype=tf.float32, shape=tuple([None]) + self.dim_trg, name='x_trg_input')
        self.y_trg = tf.placeholder(dtype=tf.float32, shape=(None, self.num_classes),
                                    name='y_trg_input')
        self.y_src_domain = tf.placeholder(dtype=tf.float32, shape=(None, self.num_src_domain),
                                    name='y_src_domain_input')

        T = tb.utils.TensorDict(dict(
            x_tmp=tf.placeholder(dtype=tf.float32, shape=tuple([None]) + self.dim_src),
            y_tmp=tf.placeholder(dtype=tf.float32, shape=(None, self.num_classes))
        ))

        self.is_training = tf.placeholder(tf.bool, shape=(), name='is_training')
        self.alexNet = AlexNetModel(num_classes=self.num_classes, is_training=self.is_training)

        self.x_src_mid_lst = []
        for i in range(self.num_src_domain):
            if i == 0:
                x_src_mid_feat = tf.reshape(self.alexNet.inference(self.x_src_lst[i], extract_feat=True),
                                            (-1, 8, 8, 64))
            else:
                x_src_mid_feat = tf.reshape(self.alexNet.inference(self.x_src_lst[i], reuse=True, extract_feat=True),
                                            (-1, 8, 8, 64))

            x_src_mid = self._build_source_middle(x_src_mid_feat, is_reused=i)
            self.x_src_mid_lst.append(x_src_mid)

        self.x_trg_mid_feat = tf.reshape(self.alexNet.inference(self.x_trg, reuse=True, extract_feat=True), (-1, 8, 8, 64))
        self.x_trg_mid = self._build_target_middle(self.x_trg_mid_feat, reuse=True)

        # <editor-fold desc="Classifier-logits">
        self.y_src_logit_lst = []
        for i in range(self.num_src_domain):
            y_src_logit = self._build_class_src_discriminator(self.x_src_mid_lst[i], self.num_classes, i)
            self.y_src_logit_lst.append(y_src_logit)

        self.y_trg_logit = self._build_class_trg_discriminator(self.x_trg_mid,
                                                               self.num_classes)
        # </editor-fold>

        # <editor-fold desc="Classification">
        self.src_loss_class_lst = []
        self.src_loss_class_sum = tf.constant(0.0)
        for i in range(self.num_src_domain):
            src_loss_class_detail = tf.nn.softmax_cross_entropy_with_logits_v2(
                logits=self.y_src_logit_lst[i], labels=self.y_src_lst[i])
            src_loss_class = tf.reduce_mean(src_loss_class_detail)
            self.src_loss_class_lst.append(self.src_domain_trade_off[i]*src_loss_class)
            self.src_loss_class_sum += self.src_domain_trade_off[i]*src_loss_class
        # </editor-fold>

        # <editor-fold desc="Source domain discriminator">
        self.x_src_mid_all = tf.concat(self.x_src_mid_lst, axis=0)
        self.y_src_discriminator_logit = self._build_domain_discriminator(self.x_src_mid_all)
        self.src_loss_discriminator_detail = tf.nn.softmax_cross_entropy_with_logits_v2(
            logits=self.y_src_discriminator_logit, labels=self.y_src_domain)
        self.src_loss_discriminator = tf.reduce_mean(self.src_loss_discriminator_detail)
        # </editor-fold>

        # <editor-fold desc="Compute teacher hS(xS)">
        self.y_src_teacher_all = []
        for i, bs in zip(range(self.num_src_domain),
                         range(0, self.batch_size_src * self.num_src_domain, self.batch_size_src)):
            y_src_logit_each_h_lst = []
            for j in range(self.num_src_domain):
                y_src_logit_each_h = self._build_class_src_discriminator(self.x_src_mid_lst[i], self.num_classes,
                                                                  j, reuse=True)
                y_src_logit_each_h_lst.append(y_src_logit_each_h)
            y_src_logit_each_h_lst = tf.nn.softmax(tf.convert_to_tensor(y_src_logit_each_h_lst))
            y_src_discriminator_prob = tf.nn.softmax(tf.gather(self.y_src_discriminator_logit,
                                                               tf.range(bs, bs + self.batch_size_src,
                                                                        dtype=tf.int32), axis=0))
            y_src_teacher = self._compute_teacher_hs(y_src_logit_each_h_lst, y_src_discriminator_prob)
            self.y_src_teacher_all.append(y_src_teacher)
        self.y_src_teacher_all = tf.concat(self.y_src_teacher_all, axis=0)
        # </editor-fold>

        # <editor-fold desc="Compute teacher hS(xT)">
        y_trg_logit_each_h_lst = []
        for j in range(self.num_src_domain):
            y_trg_logit_each_h = self._build_class_src_discriminator(self.x_trg_mid, self.num_classes,
                                                                     j, reuse=True)
            y_trg_logit_each_h_lst.append(y_trg_logit_each_h)
        y_trg_logit_each_h_lst = tf.nn.softmax(tf.convert_to_tensor(y_trg_logit_each_h_lst))
        self.y_trg_src_domains_logit = self._build_domain_discriminator(self.x_trg_mid, reuse=True)
        y_trg_discriminator_prob = tf.nn.softmax(self.y_trg_src_domains_logit)
        self.y_trg_teacher = self._compute_teacher_hs(y_trg_logit_each_h_lst, y_trg_discriminator_prob)
        # </editor-fold>

        # <editor-fold desc="Compute pseudo-label loss">
        self.ht_g_xs = build_class_discriminator_template(
            self.x_src_mid_all, training_phase=self.is_training, scope='c-trg', num_classes=self.num_classes,
            reuse=True, internal_update=True, class_discriminator_layout=self.classify_layout, cnn_size=self.cnn_size
        )
        self.mimic_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
            logits=self.ht_g_xs, labels=self.y_src_teacher_all)) + \
                          tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
            logits=self.y_trg_logit, labels=self.y_trg_teacher))
        # </editor-fold>

        # <editor-fold desc="Compute WS loss">
        self.data_shift_loss = self._compute_cosine_similarity(self.x_trg_mid, self.x_src_mid_all)
        self.label_shift_loss = self._compute_label_shift_loss(self.y_trg_logit, self.ht_g_xs)
        self.data_label_shift_loss = self.data_shift_troff*self.data_shift_loss + self.label_shift_loss
        self.g_network = tf.reshape(self._build_phi_network(self.x_trg_mid), [-1])
        self.exp_term = (- self.data_label_shift_loss + self.g_network) / self.theta
        self.g_network_loss = tf.reduce_mean(self.g_network)
        self.OT_loss = tf.reduce_mean(
            - self.theta * \
            (
                    tf.log(1.0 / self.batch_size) +
                    tf.reduce_logsumexp(self.exp_term, axis=1)
            )
        ) + self.g_network_trade_off * self.g_network_loss
        # </editor-fold>

        # <editor-fold desc="Compute conditional entropy loss w.r.t. target distribution">
        self.trg_loss_cond_entropy = tf.reduce_mean(softmax_x_entropy_two(labels=self.y_trg_logit,
                                                                   logits=self.y_trg_logit))
        # </editor-fold>

        # <editor-fold desc="Accuracy">
        self.src_accuracy_lst = []
        for i in range(self.num_src_domain):
            y_src_pred = tf.argmax(self.y_src_logit_lst[i], 1, output_type=tf.int32)
            y_src_sparse = tf.argmax(self.y_src_lst[i], 1, output_type=tf.int32)
            src_accuracy = tf.reduce_mean(tf.cast(tf.equal(y_src_sparse, y_src_pred), 'float32'))
            self.src_accuracy_lst.append(src_accuracy)
        # compute acc for target domain
        self.y_trg_pred = tf.argmax(self.y_trg_logit, 1, output_type=tf.int32)
        self.y_trg_sparse = tf.argmax(self.y_trg, 1, output_type=tf.int32)
        self.trg_accuracy = tf.reduce_mean(tf.cast(tf.equal(self.y_trg_sparse, self.y_trg_pred), 'float32'))
        # compute acc for src domain disc
        self.y_src_domain_pred = tf.argmax(self.y_src_discriminator_logit, 1, output_type=tf.int32)
        self.y_src_domain_sparse = tf.argmax(self.y_src_domain, 1, output_type=tf.int32)
        self.src_domain_acc = tf.reduce_mean(tf.cast(tf.equal(self.y_src_domain_sparse, self.y_src_domain_pred), 'float32'))
        # </editor-fold>

        # <editor-fold desc="Put it all together">
        lst_losses = [
            (self.src_class_trade_off, self.src_loss_class_sum),
            (self.ot_trade_off, self.OT_loss),
            (self.domain_trade_off, self.src_loss_discriminator),
            (self.trg_ent_troff, self.trg_loss_cond_entropy),
            (self.mimic_trade_off, self.mimic_loss)
        ]
        self.total_loss = tf.constant(0.0)
        for trade_off, loss in lst_losses:
            self.total_loss += trade_off * loss
        # </editor-fold>

        # <editor-fold desc="Evaluation">
        primary_student_variables = self._get_variables(self._get_student_primary_scopes())

        self.batch_student_acc = batch_ema_acc(self.y_trg, self.y_trg_logit)
        self.fn_batch_student_acc = tb.function(self.tf_session, [self.x_trg, self.y_trg, self.is_training], self.batch_student_acc)

        teacher_variables = self._get_variables(self._get_teacher_scopes())
        pretrained_var_list = [v for v in tf.trainable_variables() if v.name.split('/')[1] in self.train_layers]
        self.train_student_main = \
            tf.train.AdamOptimizer(self.learning_rate, 0.5).minimize(self.total_loss,
                                                                     var_list=teacher_variables + [primary_student_variables[1]] + [pretrained_var_list])
        self.primary_train_student_op = tf.group(self.train_student_main)

        secondary_variables = self._get_variables(self._get_student_secondary_scopes())
        self.secondary_train_student_op = \
            tf.train.AdamOptimizer(self.learning_rate, 0.5).minimize(-self.OT_loss,
                                                                   var_list=secondary_variables)
        # </editor-fold>

        # <editor-fold desc="Loading pre-trained weights">
        g_h_variables = self._get_variables(self._get_all_g_h())
        self.all_g_h_variables = []
        for s in g_h_variables:
            self.all_g_h_variables += s
        # </editor-fold>

        # <editor-fold desc="Summaries">
        tf.summary.scalar('loss/total_loss', self.total_loss)
        tf.summary.scalar('loss/W_distance', self.OT_loss)
        tf.summary.scalar('loss/src_loss_discriminator', self.src_loss_discriminator)
        tf.summary.scalar('loss/data_shift_loss', tf.reduce_mean(self.data_shift_loss))
        tf.summary.scalar('loss/label_shift_loss', tf.reduce_mean(self.label_shift_loss))
        tf.summary.scalar('loss/data_label_shift_loss', tf.reduce_mean(self.data_label_shift_loss))
        tf.summary.scalar('loss/exp_term', tf.reduce_mean(self.exp_term))
        tf.summary.histogram('loss/g_batch', self.g_network)
        tf.summary.scalar('loss/g_network_loss', self.g_network_loss)

        for i in range(self.num_src_domain):
            tf.summary.scalar('loss/src_loss_class_{}'.format(i), self.src_loss_class_lst[i])
            tf.summary.scalar('acc/src_acc_{}'.format(i), self.src_accuracy_lst[i])
        tf.summary.scalar('acc/src_domain_acc', self.src_domain_acc)
        tf.summary.scalar('acc/trg_acc', self.trg_accuracy)

        tf.summary.scalar('hyperparameters/learning_rate', self.learning_rate)
        tf.summary.scalar('hyperparameters/src_class_trade_off', self.src_class_trade_off)
        tf.summary.scalar('hyperparameters/g_network_trade_off', self.g_network_trade_off)
        tf.summary.scalar('hyperparameters/domain_trade_off', self.domain_trade_off)
        tf.summary.scalar('hyperparameters/src_vat_trade_off', self.src_vat_trade_off)
        tf.summary.scalar('hyperparameters/trg_vat_troff', self.trg_vat_troff)
        tf.summary.scalar('hyperparameters/trg_ent_troff', self.trg_ent_troff)
        self.tf_merged_summaries = tf.summary.merge_all()
        # </editor-fold>

    def _fit_loop(self):
        print('Start training MOST at', os.path.abspath(__file__))
        print('============ LOG-ID: %s ============' % self.current_time)

        src_batchsize = self.batch_size // self.num_src_domain
        self.tf_session.run(tf.global_variables_initializer())
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=101)
        self.log_path = os.path.join(model_dir(), self.model_name, "logs",
                                     "{}".format(self.current_time))
        self.tf_summary_writer = tf.summary.FileWriter(self.log_path, self.tf_session.graph)

        print("Load pretrained AlexNet")
        self.alexNet.load_original_weights(self.tf_session, skip_layers=self.train_layers)

        # <editor-fold desc="Load pre-trained shallow network">
        print("Load pre-trained shallow network")
        graph = tf.Graph()
        with graph.as_default():
            with tf.Session() as sess:
                checkpoint_path = os.path.join(model_dir(), self.model_name, "saved-model",
                                               "{}".format(self.mdaot_model_id))
                check_point = tf.train.get_checkpoint_state(checkpoint_path)
                model_checkpoint_path = os.path.join(checkpoint_path, check_point.model_checkpoint_path.split('/')[-1])
                saver_old = tf.train.import_meta_graph("{}.meta".format(model_checkpoint_path))
                saver_old.restore(sess, model_checkpoint_path)
                print("Loaded model parameters from %s\n" % model_checkpoint_path)

                saved_variables = graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
                saved_values = [sess.run(v) for v in saved_variables]

                g_mean_var = []
                h_mean_var = []
                g_mean_var_id = [0]
                h_mean_var_id = [0, 0]

                for i in g_mean_var_id:
                    mean = graph.get_tensor_by_name("generator/l{}/conv2d/bn/mean:0".format(i))
                    var = graph.get_tensor_by_name("generator/l{}/conv2d/bn/var:0".format(i))
                    mean, var = sess.run([mean, var])
                    g_mean_var.append([mean, var])

                for i in range(len(h_mean_var_id)):
                    mean = graph.get_tensor_by_name("classifier/{}/l{}/dense/bn/mean:0".format(i, h_mean_var_id[i]))
                    var = graph.get_tensor_by_name("classifier/{}/l{}/dense/bn/var:0".format(i, h_mean_var_id[i]))
                    mean, var = sess.run([mean, var])
                    h_mean_var.append([mean, var])

        for i in range(len(g_mean_var)):
            mean = self.tf_graph.get_tensor_by_name("generator/l{}/conv2d/bn/mean:0".format(g_mean_var_id[i]))
            var = self.tf_graph.get_tensor_by_name("generator/l{}/conv2d/bn/var:0".format(g_mean_var_id[i]))
            self.tf_session.run(tf.assign(mean, g_mean_var[i][0]))
            self.tf_session.run(tf.assign(var, g_mean_var[i][1]))

        for i in range(len(h_mean_var)):
            mean = self.tf_graph.get_tensor_by_name(
                "classifier/{}/l{}/dense/bn/mean:0".format(i, h_mean_var_id[i]))
            var = self.tf_graph.get_tensor_by_name(
                "classifier/{}/l{}/dense/bn/var:0".format(i, h_mean_var_id[i]))
            self.tf_session.run(tf.assign(mean, h_mean_var[i][0]))
            self.tf_session.run(tf.assign(var, h_mean_var[i][1]))

        domain_disc_w = self.tf_graph.get_tensor_by_name('domain_disc/l0/dense/weights:0')
        domain_disc_b = self.tf_graph.get_tensor_by_name('domain_disc/l0/dense/biases:0')
        phi_net_w = self.tf_graph.get_tensor_by_name('phi_net/l0/dense/weights:0')
        phi_net_b = self.tf_graph.get_tensor_by_name('phi_net/l0/dense/biases:0')
        self.tf_session.run(tf.assign(domain_disc_w, saved_values[16]))
        self.tf_session.run(tf.assign(domain_disc_b, saved_values[17]))
        self.tf_session.run(tf.assign(phi_net_w, saved_values[18]))
        self.tf_session.run(tf.assign(phi_net_b, saved_values[19]))
        for i in range(12):
            self.tf_session.run(tf.assign(self.all_g_h_variables[i], saved_values[i]))
        # </editor-fold>

        feed_y_src_domain = to_categorical(np.repeat(np.arange(self.num_src_domain),
                                      repeats=self.sample_size * self.num_classes, axis=0))

        for it in range(self.num_iters):
            feed_data = dict()
            for k in range(self.num_src_domain):
                feed_data[self.x_src_lst[k]], feed_data[self.y_src_lst[k]] = self.src_preprocessors[k].next_batch(
                    src_batchsize)

            feed_data[self.x_trg], feed_data[self.y_trg] = self.trg_train_preprocessor.next_batch(self.batch_size)

            feed_data[self.y_src_domain] = feed_y_src_domain
            feed_data[self.is_training] = True

            for i in range(0, 5):
                g_feed_data = dict()
                for k in range(self.num_src_domain):
                    g_feed_data[self.x_src_lst[k]], g_feed_data[self.y_src_lst[k]] = self.src_preprocessors[k].next_batch(
                src_batchsize)
                g_feed_data[self.x_trg], g_feed_data[self.y_trg] = self.trg_train_preprocessor.next_batch(self.batch_size)
                g_feed_data[self.is_training] = True

                _, W_dist = \
                    self.tf_session.run(
                        [self.secondary_train_student_op, self.OT_loss],
                        feed_dict=g_feed_data
                    )
            _, total_loss, src_loss_class_sum, src_loss_class_lst, src_loss_discriminator, src_acc_lst, trg_acc, src_domain_acc, mimic_loss = \
                self.tf_session.run(
                    [self.primary_train_student_op, self.total_loss, self.src_loss_class_sum, self.src_loss_class_lst, self.src_loss_discriminator,
                     self.src_accuracy_lst, self.trg_accuracy, self.src_domain_acc, self.mimic_loss],
                    feed_dict=feed_data
                )

            if it == 0 or (it + 1) % self.summary_freq == 0:
                print(
                    "iter %d/%d total_loss %.3f; src_loss_class_sum %.3f; W_dist %.3f;\n src_loss_discriminator %.3f, pseudo_lbl_loss %.3f" % (
                        it + 1, self.num_iters, total_loss, src_loss_class_sum, W_dist,
                        src_loss_discriminator, mimic_loss))
                for k in range(self.num_src_domain):
                    print('src_loss_class_{}: {:.3f} acc {:.2f}'.format(k, src_loss_class_lst[k], src_acc_lst[k]*100))
                print("src_domain_disc_acc: %.2f, trg_acc: %.2f;" % (src_domain_acc*100, trg_acc*100))

                summary = self.tf_session.run(self.tf_merged_summaries, feed_dict=feed_data)
                self.tf_summary_writer.add_summary(summary, it + 1)
                self.tf_summary_writer.flush()

            if it == 0 or (it + 1) % self.summary_freq == 0:
                if not self.only_save_final_model:
                    self.save_trained_model(saver, it + 1)
                elif it + 1 == self.num_iters:
                    self.save_trained_model(saver, it + 1)
                if (it + 1) % (self.num_iters // 50) == 0:
                    self.save_value(step=it + 1)

    def save_trained_model(self, saver, step):
        checkpoint_path = os.path.join(model_dir(), self.model_name, "saved-model",
                                       "{}".format(self.current_time))
        checkpoint_path = os.path.join(checkpoint_path, "mdaot_" + self.current_time + ".ckpt")

        directory = os.path.dirname(checkpoint_path)
        if not os.path.exists(directory):
            os.makedirs(directory)
        saver.save(self.tf_session, checkpoint_path, global_step=step)

    def save_value(self, step):
        student_acc, summary = self.compute_value()

        self.tf_summary_writer.add_summary(summary, step)
        self.tf_summary_writer.flush()

        print_list = ['test_acc', round(student_acc * 100, 2)]
        print(print_list)

    def compute_value(self,):
        n = len(self.trg_test_preprocessor.labels)
        bs = 200
        student_acc_full = np.ones(n, dtype=float)

        for i in range(0, n, bs):
            x, y = self.trg_test_preprocessor.next_batch(bs)
            student_acc_batch = self.fn_batch_student_acc(x, y, False)
            student_acc_full[i:i + bs] = student_acc_batch

        student_acc = np.mean(student_acc_full)
        self.trg_test_preprocessor.reset_pointer()

        summary = tf.Summary.Value(tag='trg_test/student_acc', simple_value=student_acc)
        summary = tf.Summary(value=[summary])

        return student_acc, summary
