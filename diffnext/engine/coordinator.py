# ------------------------------------------------------------------------
# Copyright (c) 2024-present, BAAI. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, esither express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ------------------------------------------------------------------------
"""Experiment coordinator."""

import os
import os.path as osp
import time

import numpy as np

from diffnext.config import cfg
from diffnext.utils import logging


class Coordinator(object):
    """Manage the unique experiments."""

    def __init__(self, cfg_file, exp_dir=None):
        cfg.merge_from_file(cfg_file)
        if logging.is_root():
            if exp_dir is None:
                name = time.strftime("%Y%m%d_%H%M%S", time.localtime(time.time()))
                exp_dir = "../experiments/{}".format(name)
                if not osp.exists(exp_dir):
                    os.makedirs(exp_dir, exist_ok=True)
            else:
                if not osp.exists(exp_dir):
                    os.makedirs(exp_dir, exist_ok=True)
        self.exp_dir = exp_dir
        self.deepspeed = None

    def path_at(self, file, auto_create=True):
        try:
            path = osp.abspath(osp.join(self.exp_dir, file))
            if auto_create and not osp.exists(path):
                os.makedirs(path)
        except OSError:
            path = osp.abspath(osp.join("/tmp", file))
            if auto_create and not osp.exists(path):
                os.makedirs(path)
        return path

    def get_checkpoint(self, step=None, last_idx=1, wait=False):
        path = self.path_at("checkpoints")

        def locate(last_idx=None):
            files = os.listdir(path)
            files = list(filter(lambda x: "_iter_" in x, files))
            file_steps = []
            for i, file in enumerate(files):
                file_step = int(file.split("_iter_")[-1].split(".")[0])
                if step == file_step:
                    return osp.join(path, files[i]), file_step
                file_steps.append(file_step)
            if step is None:
                if len(files) == 0:
                    return None, 0
                if last_idx > len(files):
                    return None, 0
                file = files[np.argsort(file_steps)[-last_idx]]
                file_step = file_steps[np.argsort(file_steps)[-last_idx]]
                return osp.join(path, file), file_step
            return None, 0

        file, file_step = locate(last_idx)
        while file is None and wait:
            logging.info("Wait for checkpoint at {}.".format(step))
            time.sleep(10)
            file, file_step = locate(last_idx)
        return file, file_step
