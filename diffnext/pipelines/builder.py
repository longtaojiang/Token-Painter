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
"""Pipeline builders."""

from typing import Dict

import json
import os
import tempfile

import torch

from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.schedulers.scheduling_utils import SchedulerMixin
from diffnext.utils.registry import Registry

PIPELINES = Registry("pipelines")


def get_pipeline_path(
    pretrained_path,
    module_dict: dict = None,
    module_config: Dict[str, dict] = None,
    target_path: str = None,
) -> str:
    """Return the pipeling loading path.

    Args:
        pretrained_path (str)
            The pretrained path to load pipeline.
        module_dict (dict, *optional*)
            The path dict to load custom modules.
        module_config (Dict[str, dict], *optional*)
            The custom configurations to dump into ``config.json``.
        target_path (str, *optional*)
            The path to store custom modules and configs.

    Returns:
       str: The pipeline loading path.

    """
    if module_dict is None and module_config is None:
        return pretrained_path
    target_path = target_path or tempfile.mkdtemp()
    for k in os.listdir(pretrained_path):
        if not os.path.isdir(os.path.join(pretrained_path, k)):
            continue
        os.makedirs(os.path.join(target_path, k), exist_ok=True)
        for _ in os.listdir(os.path.join(pretrained_path, k)):
            os.symlink(os.path.join(pretrained_path, k, _), os.path.join(target_path, k, _))
    module_dict = module_dict.copy() if module_dict is not None else {}
    model_index = module_dict.pop("model_index", os.path.join(pretrained_path, "model_index.json"))
    model_index = json.load(open(model_index))
    for k, v in module_dict.items():
        model_index.pop(k) if not v else None
        os.symlink(v, os.path.join(target_path, k)) if v else None
    for k, v in (module_config or {}).items():
        config_file = os.path.join(target_path, k, "config.json")
        os.remove(config_file) if os.path.exists(config_file) else None
        json.dump(v, open(config_file, "w"))
    json.dump(model_index, open(os.path.join(target_path, "model_index.json"), "w"))
    return target_path


def build_diffusion_scheduler(scheduler_path, sample=False, **kwargs) -> SchedulerMixin:
    """Create a diffusion scheduler instance.

    Args:
        scheduler_path (str or scheduler instance)
            The path to load a diffusion scheduler.
        sample (bool, *optional*, default to False)
            Whether to create the sampling-specific scheduler.

    Returns:
        SchedulerMixin: The diffusion scheduler.

    """
    from diffnext.schedulers.scheduling_ddpm import DDPMScheduler
    from diffnext.schedulers.scheduling_flow import FlowMatchEulerDiscreteScheduler  # noqa

    if isinstance(scheduler_path, str):
        class_key = "_{}_class_name".format("sample" if sample else "noise")
        class_type = locals()[DDPMScheduler.load_config(**locals())[class_key]]
        return class_type.from_pretrained(**locals())
    elif hasattr(scheduler_path, "config"):
        class_type = locals()[type(scheduler_path).__name__]
        return class_type.from_config(scheduler_path.config)
    return None


def build_pipeline(
    pretrained_path,
    pipe_type=None,
    precison="float16",
    config=None,
    **kwargs,
) -> DiffusionPipeline:
    """Create a diffnext pipeline instance.

    Examples:
        ```py
        >>> from diffnext.pipelines import build_pipeline
        >>> pipe = build_pipeline("BAAI/nova-d48w768-sdxl1024", "nova_train_t2i")
        ```

    Args:
        pretrained_path (str):
            The model path that includes ``model_index.json`` to create pipeline.
        pipe_type (str or `type(XXXPipeline)`, *optional*)
            The registered pipeline class or specific pipeline type.
        precision (str, *optional*, default to ``float16``)
            The compute precision used for all pipeline components.
        cfg (object, *optional*)
            The config object.

    Returns:
        DiffusionPipeline: The diffusion pipeline.

    """
    pipe_type = config.PIPELINE.TYPE if config else pipe_type
    pipe_type = PIPELINES.get(pipe_type).func if isinstance(pipe_type, str) else pipe_type
    precison = config.MODEL.PRECISION if config else precison
    kwargs.setdefault("trust_remote_code", True)
    kwargs.setdefault("torch_dtype", getattr(torch, precison.lower()))
    return pipe_type.from_pretrained(pretrained_path, **kwargs)
