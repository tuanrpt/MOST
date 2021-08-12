#  Copyright (c) 2021, Tuan Nguyen.
#  All rights reserved.

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import sys
import numpy as np
import tensorflow as tf

from generic_utils import random_seed
from generic_utils import data_dir
from resnet.preprocessor import BatchPreprocessor


def test_real_dataset(create_obj_func, src_name=None, trg_name=None):
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

    src_domains = src_name.split(',')
    num_src_domain = len(src_domains)
    input_size = [227, 227]
    n_channels = 3
    src_preprocessors = []
    dataset_path = os.path.join(data_dir(), 'office31')
    multi_scale = list(map(int, users_params['multi_scale'].split(',')))

    for src_domain in src_domains:
        file_path_train = os.path.join(dataset_path, '{}_train.txt'.format(src_domain))
        src_preprocessor_i = BatchPreprocessor(dataset_file_path=file_path_train,
                                                 num_classes=users_params['num_classes'],
                                                 output_size=input_size, horizontal_flip=True, shuffle=True,
                                                 multi_scale=multi_scale)
        src_preprocessors.append(src_preprocessor_i)

    trg_train_preprocessor = BatchPreprocessor(dataset_file_path=os.path.join(dataset_path, '{}_train.txt'.format(trg_name)),
                                         num_classes=users_params['num_classes'], output_size=input_size, horizontal_flip=True, shuffle=True,
                                               multi_scale=multi_scale)

    trg_test_preprocessor = BatchPreprocessor(dataset_file_path=os.path.join(dataset_path, '{}_test.txt'.format(trg_name)),
                                         num_classes=users_params['num_classes'], output_size=input_size)

    assert users_params['batch_size'] % num_src_domain == 0

    print('users_params:', users_params)
    print('src_name:', src_name, ', trg_name:', trg_name)
    for i in range(len(src_domains)):
        print(src_domains[i], len(src_preprocessors[i].labels))
    print(trg_name, len(trg_test_preprocessor.labels))

    learner = create_obj_func(users_params)
    learner.dim_src = tuple(input_size + [n_channels])
    learner.dim_trg = tuple(input_size + [n_channels])

    print("dim_src:", tuple(input_size + [n_channels]))
    print("dim_trg:", tuple(input_size + [n_channels]))

    learner._init(src_preprocessors, trg_train_preprocessor, trg_test_preprocessor, num_src_domain)
    learner._build_model()
    learner._fit_loop()


def main_func(
        create_obj_func,
        choice_default=0,
        src_name_default='svmguide1',
        trg_name_default='svmguide1',
        run_exp=False):

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
            pass
            # add another function here
        elif choice == 1:
            test_real_dataset(create_obj_func, src_name, trg_name)


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
