import datetime
import os
import random
import sys
import time
import uuid

import hydra.utils
import numpy as np
import torch
from lightning_fabric import Fabric
from lightning_fabric.strategies import DDPStrategy
from lightning_fabric.utilities.types import ReduceOp
from loguru import logger
from torch.nn.functional import binary_cross_entropy_with_logits
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from torchvision.transforms.v2 import GaussianBlur
from tqdm import trange, tqdm

from src.data import GraphDataset
from src.definitions import AppConfig
from src.eval import ModelEvaluator
from src.logging import configure_logging
from src.monitoring import SystemMonitor, TrainingMonitor
from src.nn import GraphRecognitionModel
from src.plot import plot_training
from src.util import (
    patch_environment,
    SaltAndPepperNoise,
    identity,
    RunningNormalize,
)


def get_datasets(fabric: Fabric, conf: AppConfig):
    generator_train = hydra.utils.instantiate(conf.environment.train)
    generator_val = hydra.utils.instantiate(conf.environment.val)

    dataset_args = dict(
        image_size=conf.training.image_size,
        device=str(fabric.device),
        # use half of everything to account for second buffer
        min_size=conf.training.cache_warmup // fabric.world_size,
        cache_size=conf.training.cache_size // fabric.world_size,
        fake_size=conf.training.samples_per_epoch // fabric.world_size,
        batch_size=conf.training.batch_size // fabric.world_size,
        max_node_degree=conf.environment.max_node_degree,
    )

    dataset_train = GraphDataset(generator_train, **dataset_args)
    dataset_val = GraphDataset(generator_val, **dataset_args)

    return dataset_train, dataset_val


def get_model(fabric: Fabric, conf: AppConfig, sample):
    normalizer = RunningNormalize(
        num_channels=3,
        num_steps=conf.training.image_warmup // fabric.world_size,
        device=str(fabric.device),
    )

    model = GraphRecognitionModel(conf.agent, sample, normalizer)

    params = model.parameter_count()

    logger.info(params)

    conf.overrides.append(f"meta.image_embedding={model.image_embedding_size}")
    conf.overrides.append(f"meta.graph_embedding={model.graph_embedding_size}")

    conf.overrides.append(f"params.cnn={params['cnn']:,}")
    conf.overrides.append(f"params.gnn={params['gnn']:,}")
    conf.overrides.append(f"params.mlp={params['mlp']:,}")
    conf.overrides.append(f"params.all={params['all']:,}")

    optimizer = torch.optim.RAdam(
        params=model.parameters(),
        lr=conf.training.lr,
        weight_decay=conf.training.weight_decay,
    )

    model, optimizer = fabric.setup(model, optimizer)

    return model, optimizer


@torch.inference_mode()
def evaluate_epoch(
    conf, fabric, model, evaluator_train, evaluator_val, batch_size, epoch
):
    model.eval()

    eval_iters = max(conf.training.evals_per_epoch // fabric.world_size, 1)

    traj_train, x_ent_train, top_k_train = evaluator_train.evaluate(
        model, eval_iters, batch_size
    )
    traj_val, x_ent_val, top_k_val = evaluator_val.evaluate(
        model, eval_iters, batch_size
    )

    traj_train.sort(key=lambda x: len(x.steps), reverse=True)
    traj_val.sort(key=lambda x: len(x.steps), reverse=True)

    plot_start = time.time()

    if fabric.is_global_zero:
        evaluator_train.plot(f"epoch_{epoch}_train_{fabric.global_rank}", traj_train[0])
        evaluator_val.plot(f"epoch_{epoch}_val_{fabric.global_rank}", traj_val[0])

    plot_end = time.time()

    corrects_train = [x.correct for x in traj_train]
    corrects_val = [x.correct for x in traj_val]

    matches_train = [x.match for x in traj_train]
    matches_val = [x.match for x in traj_val]

    len_train = [len(x.steps) for x in traj_train]
    len_val = [len(x.steps) for x in traj_val]

    corrects_train, matches_train, corrects_val, matches_val, len_train, len_val = (
        fabric.all_gather(
            [
                corrects_train,
                matches_train,
                corrects_val,
                matches_val,
                len_train,
                len_val,
            ]
        )
    )

    return {
        "train_correct_pred": torch.stack(corrects_train).float().mean().item(),
        "train_match_pred": torch.stack(matches_train).float().mean().item(),
        "x_ent_train": torch.as_tensor(x_ent_train).float().mean().item(),
        "x_ent_val": torch.as_tensor(x_ent_val).float().mean().item(),
        "top_k_train": torch.as_tensor(top_k_train).float().mean().item(),
        "top_k_val": torch.as_tensor(top_k_val).float().mean().item(),
        "val_correct_pred": torch.stack(corrects_val).float().mean().item(),
        "val_match_pred": torch.stack(matches_val).float().mean().item(),
        "len_train": torch.stack(len_train).float().mean().item(),
        "len_val": torch.stack(len_val).float().mean().item(),
        "duration_plot": (plot_end - plot_start) / 60,
    }


def train_epoch(
    fabric,
    dl,
    dataset,
    conf,
    train_monitor,
    model,
    optimizer,
    lr_scheduler,
    pbar_args,
):
    noise = v2.Compose(
        [
            SaltAndPepperNoise(),
            GaussianBlur(3),
        ]
    )

    model.train()

    start = time.time()
    hard_samples_total = 0

    for batch in tqdm(dl, desc="training", **pbar_args):
        batch = dataset.collate(batch)

        images = batch["x"]["images"]
        batch["x"]["images"] = noise(images)

        y_pred = model(batch["x"])
        y_true = batch["y"]

        eps = conf.training.label_smoothing
        y_smooth = (1 - eps) * y_true + eps / 2

        # don't use label smoothing, may interfere with prioritization and also
        # increases test loss artificially
        graph_loss = binary_cross_entropy_with_logits(
            y_pred, y_smooth, reduction="none"
        )

        loss = graph_loss.mean()

        optimizer.zero_grad()
        fabric.backward(loss)
        grad_norm = fabric.clip_gradients(
            model, optimizer, max_norm=conf.training.max_grad_norm or np.inf
        )
        optimizer.step()

        with torch.no_grad():
            loss_pos = graph_loss[torch.where(y_true == 1)[0]]  # noqa
            loss_neg = graph_loss[torch.where(y_true == 0)[0]]  # noqa

            train_monitor.update(
                graph_loss, loss_pos, loss_neg, grad_norm, y_pred, y_true
            )

            try:
                lr_scheduler.step()
            except Exception as e:
                pass

    end = time.time()
    return hard_samples_total / (end - start)


def train(conf: AppConfig):
    patch_environment()

    torch.set_float32_matmul_precision("medium")

    n_nodes = int(os.environ.get("SLURM_NNODES", 1))
    job_id = os.environ.get("SLURM_JOBID", int.from_bytes(random.randbytes(4)))
    model_id = str(uuid.uuid4())

    fabric = Fabric(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=torch.cuda.device_count(),
        num_nodes=n_nodes,
        precision="bf16-mixed",
        strategy=DDPStrategy(
            timeout=datetime.timedelta(seconds=300),
        ),
    )

    fabric.launch()

    local_batch_size = conf.training.batch_size // fabric.world_size

    conf.output_dir += f"-{job_id}"

    os.makedirs("models", exist_ok=True)
    os.makedirs(conf.output_dir, exist_ok=True)

    configure_logging(
        f"{conf.output_dir}/training.log", level="INFO", rank=fabric.global_rank
    )

    dataset_train, dataset_val = get_datasets(fabric, conf)

    sample = dataset_train.get_item(encode_image=False)

    model, optimizer = get_model(fabric, conf, sample)

    lr_scheduler = OneCycleLR(
        optimizer,
        max_lr=conf.training.lr,
        total_steps=conf.training.lr_cycle // conf.training.batch_size,
        pct_start=conf.training.lr_warmup / conf.training.lr_cycle,
        cycle_momentum=False,
        div_factor=conf.training.lr_factor,
        final_div_factor=1,
    )

    dl = DataLoader(dataset_train, batch_size=1, collate_fn=identity, shuffle=False)

    evaluator_train = ModelEvaluator(conf.output_dir, dataset_train)
    evaluator_val = ModelEvaluator(conf.output_dir, dataset_val)

    sys_monitor = SystemMonitor().start()
    train_monitor = TrainingMonitor(device=fabric.device)

    training_logs = []

    pbar_args = dict(leave=False, file=sys.stdout, disable=not fabric.is_global_zero)

    logger.info("training...")

    try:
        for epoch in trange(conf.training.epochs, desc="training", **pbar_args):
            epoch_start = time.time()

            model.eval()

            train_start = time.time()

            buffer_sps_hard = train_epoch(
                fabric,
                dl,
                dataset_train,
                conf,
                train_monitor,
                model,
                optimizer,
                lr_scheduler,
                pbar_args,
            )

            train_end = time.time()

            buffer_size_pos = dataset_train.size_pos
            buffer_size_neg = dataset_train.size_neg
            buffer_sps_uni = dataset_train.sps

            buffer_size_pos, buffer_size_neg, buffer_sps_uni, buffer_sps_hard = (
                fabric.all_reduce(
                    [buffer_size_pos, buffer_size_neg, buffer_sps_uni, buffer_sps_hard],
                    reduce_op=ReduceOp.SUM,
                )
            )

            buffer_gpi = dataset_train.gpi
            buffer_pos = dataset_train.pos_true
            buffer_gpi, buffer_pos = fabric.all_reduce(
                [buffer_gpi, buffer_pos], reduce_op=ReduceOp.AVG
            )

            metrics_epoch = train_monitor.get_metrics()

            metrics_train = {
                "lr": optimizer.param_groups[0]["lr"],
                **metrics_epoch,
                **model.parameter_norm(),
            }

            eval_start = time.time()
            metrics_eval = evaluate_epoch(
                conf,
                fabric,
                model,
                evaluator_train,
                evaluator_val,
                local_batch_size,
                epoch,
            )

            metrics_system = sys_monitor.get_metrics()

            eval_end = time.time()

            epoch_end = time.time()

            if not fabric.is_global_zero:
                continue

            torch.save(model.module, os.path.join("models", f"{job_id}_{model_id}.pt"))

            logs = {
                "epoch": epoch,
                "step": (epoch + 1) * conf.training.samples_per_epoch / 1e6,
                "duration_train": (train_end - train_start) / 60,
                "duration_epoch": (epoch_end - epoch_start) / 60,
                "duration_eval": (eval_end - eval_start) / 60,
                "buffer_sps_uni": buffer_sps_uni.item(),
                "buffer_sps_hard": buffer_sps_hard.item(),
                "buffer_size_pos": buffer_size_pos.item(),
                "buffer_size_neg": buffer_size_neg.item(),
                "buffer_gpi": buffer_gpi.item(),
                "pos_frac_true": buffer_pos.item(),
                **metrics_train,
                **metrics_eval,
                **metrics_system,
            }

            training_logs.append(logs)

            logs = {k: round(v, 6) for k, v in logs.items()}

            logger.info(logs)

            plot_training(training_logs, None, None, conf.output_dir, conf.overrides)

    except Exception as e:
        logger.exception(e)
    finally:
        sys_monitor.stop()
        logger.info("done")

        # terminate only if running non-interactively
        if os.environ.get("SLURM_JOB_NAME", None) == "alphagraph":
            logger.error(
                f"killing job {job_id} due to failure on rank {fabric.global_rank}"
            )
            os.system(f"scancel {job_id}")
