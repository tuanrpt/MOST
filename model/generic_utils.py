#  Copyright (c) 2021, Tuan Nguyen.
#  All rights reserved.

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import time
import numpy as np
from pathlib import Path
import os
_RANDOM_SEED = 6789


def model_dir():
    cur_dir = Path(os.path.abspath(__file__))
    return str(cur_dir.parent.parent)


def feat_dir():
    cur_dir = Path(os.path.abspath(__file__))
    par_dir = cur_dir.parent.parent
    return str(par_dir / "features")

def data_dir():
    cur_dir = Path(os.path.abspath(__file__))
    par_dir = cur_dir.parent.parent
    return str(par_dir / "data")


def random_seed():
    return _RANDOM_SEED


def tuid():
    '''
    Create a string ID based on current time
    :return: a string formatted using current time
    '''
    random_num = np.random.randint(0, 100)
    return time.strftime('%Y-%m-%d_%H.%M.%S') + str(random_num)
