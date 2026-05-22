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
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ------------------------------------------------------------------------
"""Default configurations."""

from diffnext.config.yacs import CfgNode

_C = cfg = CfgNode()

# ------------------------------------------------------------
# Training options
# ------------------------------------------------------------
_C.TRAIN = CfgNode()

# The train dataset
_C.TRAIN.DATASET = ""

# The train dataset2
_C.TRAIN.DATASET2 = ""

# The loader type for training
_C.TRAIN.LOADER = ""

# The number of threads to load train data per GPU
_C.TRAIN.NUM_THREADS = 4

# Images to fill per mini-batch
_C.TRAIN.BATCH_SIZE = 1

# The EMA decay to smooth the checkpoints
_C.TRAIN.MODEL_EMA = 0.99

# Device to place the EMA model ("" or "cpu")
_C.TRAIN.DEVICE_EMA = ""

# Condition repeat factor to enlarge noise sampling
_C.TRAIN.LOSS_REPEAT = 4

# The model checkpointing level
_C.TRAIN.CHECKPOINTING = 0

# ------------------------------------------------------------
# Model options
# ------------------------------------------------------------
_C.MODEL = CfgNode()

# The module type of model
_C.MODEL.TYPE = "transformer"

# The config dict
_C.MODEL.CONFIG = {}

# Initialize model with weights from this file
_C.MODEL.WEIGHTS = ""

# The compute precision
_C.MODEL.PRECISION = "bfloat16"

# ------------------------------------------------------------
# Pipeline options
# ------------------------------------------------------------
_C.PIPELINE = CfgNode()

# The registered pipeline type
_C.PIPELINE.TYPE = ""

# The dict of pipeline modules
_C.PIPELINE.MODULES = {}

# ------------------------------------------------------------
# Solver options
# ------------------------------------------------------------
_C.SOLVER = CfgNode()

# The interval to display logs
_C.SOLVER.DISPLAY = 20

# The interval to update ema model
_C.SOLVER.EMA_EVERY = 100

# The interval to snapshot a model
_C.SOLVER.SNAPSHOT_EVERY = 5000

# Prefix to yield the path: <prefix>_iter_XYZ
_C.SOLVER.SNAPSHOT_PREFIX = ""

# Maximum number of SGD iterations
_C.SOLVER.MAX_STEPS = 2147483647

# Base learning rate for the specified scheduler
_C.SOLVER.BASE_LR = 0.0001

# Minimal learning rate for the specified scheduler
_C.SOLVER.MIN_LR = 0.0

# The decay intervals for LRScheduler
_C.SOLVER.DECAY_STEPS = []

# The decay factor for exponential LRScheduler
_C.SOLVER.DECAY_GAMMA = 0.5

# Warm up to ``BASE_LR`` over this number of steps
_C.SOLVER.WARM_UP_STEPS = 250

# Start the warm up from ``BASE_LR`` * ``FACTOR``
_C.SOLVER.WARM_UP_FACTOR = 1.0 / 1000

# The type of optimizier
_C.SOLVER.OPTIMIZER = "AdamW"

# The adam beta2 value
_C.SOLVER.ADAM_BETA2 = 0.95

# The type of lr scheduler
_C.SOLVER.LR_POLICY = ""

# Gradient accumulation steps per SGD iteration
_C.SOLVER.ACCUM_STEPS = 1

# L2 regularization for weight parameters
_C.SOLVER.WEIGHT_DECAY = 0.02

# ------------------------------------------------------------
# Misc options
# ------------------------------------------------------------
# Number of GPUs for distributed training
_C.NUM_GPUS = 1

# Random seed for reproducibility
_C.RNG_SEED = 3

# Default GPU device index
_C.GPU_ID = 0
