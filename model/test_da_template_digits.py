#  Copyright (c) 2021, Tuan Nguyen.
#  All rights reserved.

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import sys
from scipy.io import loadmat
import numpy as np
import tensorflow as tf
from generic_utils import random_seed
from generic_utils import feat_dir
from dataLoader import DataLoader


def test_real_dataset(create_obj_func, src_name=None, trg_name=None, show=False, block_figure_on_end=False):
    print('Running {} ...'.format(os.path.basename(__file__)))

    if src_name is None:
        if len(sys.argv) > 2:
            src_name = sys.argv[2]
        else:
            raise Exception('Not specify source dataset')
    if trg_name is None:
        if len(sys.argv) > 3:
            trg_name = sys.argv[3]
        else:
            raise Exception('Not specify trgget dataset')

    np.random.seed(random_seed())
    tf.set_random_seed(random_seed())
    tf.reset_default_graph()

    print("========== Test on real data ==========")
    users_params = dict()
    users_params = parse_arguments(users_params)
    data_format = 'mat'

    if 'format' in users_params:
        data_format, users_params = extract_param('format', data_format, users_params)

    data_loader = DataLoader(src_domain=src_name,
                             trg_domain=trg_name,
                             data_path=feat_dir(),
                             data_format=data_format,
                             cast_data=users_params['cast_data'])

    assert users_params['batch_size'] % data_loader.num_src_domain == 0
    print('users_params:', users_params)

    learner = create_obj_func(users_params)
    learner.dim_src = data_loader.data_shape
    learner.dim_trg = data_loader.data_shape

    learner.x_trg_test = data_loader.trg_test[0][0]
    learner.y_trg_test = data_loader.trg_test[0][1]
    learner._init(data_loader)
    learner._build_model()
    learner._fit_loop()


def main_func(
        create_obj_func,
        choice_default=0,
        src_name_default='svmguide1',
        trg_name_default='svmguide1',
        run_exp=False,
        keep_vars=[],
        **kwargs):

    if not run_exp:
        choice_lst = [0, 1, 2]
        src_name = src_name_default
        trg_name = trg_name_default
    elif len(sys.argv) > 1:
        choice_lst = [int(sys.argv[1])]
        src_name = None
        trg_name = None
    else:
        choice_lst = [choice_default]
        src_name = src_name_default
        trg_name = trg_name_default

    for choice in choice_lst:
        if choice == 0:
            pass  # for synthetic data if possible
        elif choice == 1:
            test_real_dataset(create_obj_func, src_name, trg_name, show=False, block_figure_on_end=run_exp)


def parse_arguments(params, as_array=False):
    for it in range(4, len(sys.argv), 2):
        params[sys.argv[it]] = parse_argument(sys.argv[it + 1], as_array)
    return params


def parse_argument(string, as_array=False):
    try:
        result = int(string)
    except ValueError:
        try:
            result = float(string)
        except ValueError:
            if str.lower(string) == 'true':
                result = True
            elif str.lower(string) == 'false':
                result = False
            elif string == "[]":
                return []
            elif ('|' in string) and ('[' in string) and (']' in string):
                result = [float(item) for item in string[1:-1].split('|')]
                return result
            elif (',' in string) and ('(' in string) and (')' in string):
                split = string[1:-1].split(',')
                result = float(split[0]) ** np.arange(float(split[1]), float(split[2]), float(split[3]))
                return result
            else:
                result = string

    return [result] if as_array else result


def resolve_conflict_params(primary_params, secondary_params):
    for key in primary_params.keys():
        if key in secondary_params.keys():
            del secondary_params[key]
    return secondary_params


def extract_param(key, value, params_gridsearch, scalar=False):
    if key in params_gridsearch.keys():
        value = params_gridsearch[key]
        del params_gridsearch[key]
        if scalar and (value is not None):
            value = value[0]
    return value, params_gridsearch


def dict2string(params):
    result = ''
    for key, value in params.items():
        if type(value) is np.ndarray:
            if value.size < 16:
                result += key + ': ' + '|'.join('{0:.4f}'.format(x) for x in value.ravel()) + ', '
        else:
            result += key + ': ' + str(value) + ', '
    return '{' + result[:-2] + '}'


def load_mat_file_single_label(filename):
    filename_list = ['mnist', 'stl32', 'synsign', 'gtsrb', 'cifar32', 'usps32']
    data = loadmat(filename)
    x = data['X']
    y = data['y']
    if any(fn in filename for fn in filename_list):
        if 'mnist32_60_10' not in filename and 'mnistg' not in filename:
            y = y[0]
        else:
            y = np.argmax(y, axis=1)
    # process one-hot label encoder
    elif len(y.shape) > 1:
        y = np.argmax(y, axis=1)
    return x, y


def u2t(x):
    """Convert uint8 to [-1, 1] float
    """
    return x.astype('float32') / 255 * 2 - 1
