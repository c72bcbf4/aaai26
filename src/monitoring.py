from threading import Thread

import psutil
import torch
from pynvml import (
    nvmlInit,
    nvmlDeviceGetHandleByIndex,
    nvmlDeviceGetUtilizationRates,
    nvmlShutdown,
)
from torchmetrics import MeanMetric, CatMetric
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryF1Score,
    BinaryRecall,
    BinaryPrecision,
)


class SystemMonitor:
    def __init__(self, interval: float = 0.25):
        self.interval = interval

        args = dict(sync_on_compute=False)
        self.gpu_util = CatMetric(**args)
        self.cpu_util = CatMetric(**args)

        self.gpu_mem = CatMetric(**args)
        self.cpu_mem = CatMetric(**args)

        self.worker = Thread(target=self.get_utilization, daemon=True)
        self.collect = True

    def start(self):
        self.worker.start()
        return self

    def stop(self):
        self.collect = False

    @torch.no_grad()
    def get_utilization(self):
        nvmlInit()
        handle = nvmlDeviceGetHandleByIndex(0)

        while self.collect:
            if handle is not None:
                utilization = nvmlDeviceGetUtilizationRates(handle)

                self.gpu_util.update(utilization.gpu)
                self.gpu_mem.update(utilization.memory)

            # cpu_percent blocks for interval, no need for sleep
            cpu_util = psutil.cpu_percent(self.interval, percpu=False)
            cpu_ram = psutil.virtual_memory().percent

            self.cpu_mem.update(cpu_ram)
            self.cpu_util.update(cpu_util)

        nvmlShutdown()

    @torch.no_grad()
    def get_metrics(self):
        gpu_util = self.gpu_util.compute()
        cpu_util = self.cpu_util.compute()
        gpu_mem = self.gpu_mem.compute()
        cpu_mem = self.cpu_mem.compute()

        stats = dict(
            gpu_util_mean=gpu_util.mean().item() if len(gpu_util) != 0 else 0.0,
            gpu_util_max=gpu_util.max().item() if len(gpu_util) != 0 else 0.0,
            cpu_util_mean=cpu_util.mean().item() if len(cpu_util) != 0 else 0.0,
            cpu_util_max=cpu_util.max().item() if len(cpu_util) != 0 else 0.0,
            gpu_mem_mean=gpu_mem.mean().item() if len(gpu_mem) != 0 else 0.0,
            gpu_mem_max=gpu_mem.max().item() if len(gpu_mem) != 0 else 0.0,
            cpu_mem_mean=cpu_mem.mean().item() if len(cpu_mem) != 0 else 0.0,
            cpu_mem_max=cpu_mem.max().item() if len(cpu_mem) != 0 else 0.0,
        )

        for m in [self.gpu_util, self.cpu_util, self.gpu_mem, self.cpu_mem]:
            m.reset()

        return stats


class TrainingMonitor:
    def __init__(self, device):
        self.device = device

        self.loss_graph = MeanMetric().to(device)
        self.loss_pos = MeanMetric().to(device)
        self.loss_neg = MeanMetric().to(device)
        self.grad = MeanMetric().to(device)
        self.pos_frac = MeanMetric().to(device)
        self.logits = MeanMetric().to(device)
        self.precision = BinaryPrecision().to(device)
        self.recall = BinaryRecall().to(device)
        self.f1 = BinaryF1Score().to(device)
        self.pos = MeanMetric().to(device)
        self.acc = BinaryAccuracy().to(device)

        self.metrics = [
            self.loss_graph,
            self.loss_pos,
            self.loss_neg,
            self.grad,
            self.pos_frac,
            self.logits,
            self.precision,
            self.recall,
            self.f1,
            self.pos,
            self.acc,
        ]

    def reset(self):
        for m in self.metrics:
            m.reset()

    @torch.no_grad()
    def update(self, loss_graph, loss_pos, loss_neg, grad_norm, y_pred, y_true):
        self.precision.update(y_pred, y_true)
        self.recall.update(y_pred, y_true)
        self.f1.update(y_pred, y_true)
        self.acc.update(y_pred, y_true)
        self.loss_graph.update(loss_graph)
        self.loss_pos.update(loss_pos)
        self.loss_neg.update(loss_neg)
        self.grad.update(grad_norm)
        self.pos_frac.update(y_true.mean())
        self.logits.update(y_pred.mean())

    @torch.no_grad()
    def get_metrics(self):
        metrics = {
            "loss_graph": self.loss_graph.compute().item(),
            "loss_graph_pos": self.loss_pos.compute().item(),
            "loss_graph_neg": self.loss_neg.compute().item(),
            "grad_norm": self.grad.compute().item(),
            "acc": self.acc.compute().item(),
            "precision": self.precision.compute().item(),
            "recall": self.recall.compute().item(),
            "f1": self.f1.compute().item(),
            "pos_frac_train": self.pos_frac.compute().item(),
            "logits": self.logits.compute().item(),
        }

        self.reset()
        return metrics
