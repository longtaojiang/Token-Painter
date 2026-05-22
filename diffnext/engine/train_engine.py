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
"""Custom deepspeed trainer focused on data parallelism specialization."""

import collections
import os
import shutil

import torch

from diffnext import engine
from diffnext.config import cfg
from diffnext.data.builder import build_loader_train
from diffnext.pipelines.builder import build_pipeline, get_pipeline_path
from diffnext.utils import logging
from diffnext.utils import profiler


class Trainer(object):
    """Schedule the iterative model training."""

    def __init__(self, coordinator, start_iter=0):
        self.coordinator = coordinator
        self.loader = build_loader_train()
        self.precision = cfg.MODEL.PRECISION.lower()
        self.dtype = getattr(torch, self.precision)
        self.device_type = engine.get_device(0).type
        pipe_conf = {cfg.MODEL.TYPE: cfg.MODEL.CONFIG}
        pipe_path = get_pipeline_path(cfg.MODEL.WEIGHTS, cfg.PIPELINE.MODULES, pipe_conf)
        self.pipe = build_pipeline(pipe_path, config=cfg)
        self.pipe = self.pipe.to(device=engine.get_device(cfg.GPU_ID))
        self.ema_model = engine.build_model_ema(self.pipe.model, cfg.TRAIN.MODEL_EMA)
        self.ema_model.ema.cpu() if cfg.TRAIN.DEVICE_EMA.lower() == "cpu" else None
        self.model = self.pipe.configure_model(config=cfg)
        self.autocast = torch.autocast(self.device_type, self.dtype)
        param_groups = engine.get_param_groups(self.model)
        self.optimizer = engine.build_optimizer(param_groups)
        self.loss_scaler = torch.amp.GradScaler("cuda", enabled=self.precision == "float16")
        self.ds_model = engine.apply_deepspeed(self.model, self.optimizer, coordinator.deepspeed)
        self.ddp_model = engine.apply_ddp(self.model.float()) if self.ds_model is None else None
        self.scheduler = engine.build_lr_scheduler()
        self.metrics, self.board = collections.OrderedDict(), None
        if self.ema_model and start_iter > 0:
            ema_weights = cfg.MODEL.WEIGHTS.replace("checkpoints", "ema_checkpoints")
            ema_weights += "/%s/diffusion_pytorch_model.bin" % cfg.MODEL.TYPE
            engine.load_weights(self.ema_model.ema, ema_weights)

    @property
    def iter(self):
        return self.scheduler._step_count

    def snapshot(self):
        """Save the checkpoint of current iterative step."""
        f = cfg.SOLVER.SNAPSHOT_PREFIX + "_iter_{}/{}".format(self.iter, cfg.MODEL.TYPE)
        f = os.path.join(self.coordinator.path_at("checkpoints"), f)
        if logging.is_root() and not os.path.exists(f):
            self.model.save_pretrained(f, safe_serialization=False)
            logging.info("Wrote snapshot to: {:s}".format(f))
            if self.ema_model is not None:
                config_json = os.path.join(f, "config.json")
                f = f.replace("checkpoints", "ema_checkpoints")
                os.makedirs(f), shutil.copy(config_json, os.path.join(f, "config.json"))
                f = os.path.join(f, "diffusion_pytorch_model.bin")
                torch.save(self.ema_model.ema.state_dict(), f)

    def add_metrics(self, stats):
        """Add or update the metrics."""
        for k, v in stats["metrics"].items():
            if k not in self.metrics:
                self.metrics[k] = profiler.SmoothedValue()
            self.metrics[k].update(v)

    def display_metrics(self, stats):
        """Send metrics to the monitor."""
        iter_template = "Iteration %d, lr = %.8f, time = %.2fs"
        metric_template = " " * 4 + "Train net output({}): {:.4f} ({:.4f})"
        logging.info(iter_template % (stats["iter"], stats["lr"], stats["time"]))
        for k, v in self.metrics.items():
            logging.info(metric_template.format(k, stats["metrics"][k], v.average()))
        if self.board is not None:
            self.board.scalar_summary("lr", stats["lr"], stats["iter"])
            self.board.scalar_summary("time", stats["time"], stats["iter"])
            for k, v in self.metrics.items():
                self.board.scalar_summary(k, v.average(), stats["iter"])

    def step_ddp(self, metrics, accum_steps=1):
        """Single DDP optimization step."""
        self.optimizer.zero_grad()
        for _ in range(accum_steps):
            inputs, _ = self.loader.next()[0], self.autocast.__enter__()
            outputs, losses, _ = self.ddp_model(inputs), [], self.autocast.__exit__(0, 0, 0)
            for k, v in outputs.items():
                if "loss" not in k:
                    continue
                if isinstance(v, torch.Tensor) and v.requires_grad:
                    losses.append(v)
                metrics[k] += float(v) / accum_steps
            losses = sum(losses[1:], losses[0])
            losses = losses.mul_(1.0 / accum_steps) if accum_steps > 1 else losses
            self.loss_scaler.scale(losses).backward()
        if self.loss_scaler.is_enabled():
            metrics["~loss_scale"] += self.loss_scaler.get_scale()
        self.loss_scaler.step(self.optimizer)
        self.loss_scaler.update()

    def step_ds(self, metrics, accum_steps=1):
        """Single DeepSpeed optimization step."""
        for _ in range(accum_steps):
            inputs = self.loader.next()[0]
            outputs, losses = self.ds_model(inputs), []
            for k, v in outputs.items():
                if "loss" not in k:
                    continue
                if isinstance(v, torch.Tensor) and v.requires_grad:
                    losses.append(v)
                metrics[k] += float(v) / accum_steps
            losses = sum(losses[1:], losses[0])
            losses = losses.mul_(1.0 / accum_steps) if accum_steps > 1 else losses
            self.ds_model.backward(losses)
        if self.loss_scaler.is_enabled():
            metrics["~loss_scale"] += float(self.ds_model.optimizer._get_loss_scale())
        self.ds_model.step()

    def step(self, accum_steps=1):
        """Single model optimization step."""
        stats = {"iter": self.iter}
        metrics = collections.defaultdict(float)
        timer = profiler.Timer().tic()
        stats["lr"] = self.scheduler.get_lr()
        for group in self.optimizer.param_groups:
            group["lr"] = stats["lr"] * group.get("lr_scale", 1.0)
        self.step_ds(metrics, accum_steps) if self.ds_model else None
        self.step_ddp(metrics, accum_steps) if self.ddp_model else None
        self.scheduler.step()
        stats["time"] = timer.toc()
        stats["metrics"] = collections.OrderedDict(sorted(metrics.items()))
        return stats

    def train_model(self, start_iter=0):
        """Training loop."""
        timer = profiler.Timer()
        max_steps = cfg.SOLVER.MAX_STEPS
        accum_steps = cfg.SOLVER.ACCUM_STEPS
        display_every = cfg.SOLVER.DISPLAY
        progress_every = 10 * display_every
        ema_every = cfg.SOLVER.EMA_EVERY
        snapshot_every = cfg.SOLVER.SNAPSHOT_EVERY
        self.scheduler._step_count = start_iter
        while self.iter < max_steps:
            with timer.tic_and_toc():
                stats = self.step(accum_steps)
            self.add_metrics(stats)
            if stats["iter"] % display_every == 0:
                self.display_metrics(stats)
            if self.iter % progress_every == 0:
                logging.info(profiler.get_progress(timer, self.iter, max_steps))
            if self.iter % ema_every == 0 and self.ema_model:
                self.ema_model.update(self.model)
            if self.iter % snapshot_every == 0:
                self.snapshot()
                self.metrics.clear()


def run_train(coordinator, start_iter=0, enable_tensorboard=False):
    """Start a model training task."""
    trainer = Trainer(coordinator, start_iter=start_iter)
    if enable_tensorboard and logging.is_root():
        trainer.board = engine.build_tensorboard(coordinator.path_at("logs"))
    logging.info("#Params: %.2fM" % engine.count_params(trainer.model))
    logging.info("Start training...")
    trainer.train_model(start_iter)
    trainer.ema_model.update(trainer.model) if trainer.ema_model else None
    trainer.snapshot()
