import os
import random
import socket
import subprocess
from itertools import islice
from typing import TypedDict

import numpy as np
import torch
from torch import Tensor
from torch import nn
from torch.nn import Linear, Sequential, ModuleList
from torch_geometric.nn import (
    JumpingKnowledge,
    MessagePassing,
    GraphNorm,
)
from torchrl.data import PrioritizedSampler, ListStorage, ReplayBuffer
from torchvision.transforms.v2 import Transform


class ConcatConv(MessagePassing):
    def __init__(self, in_channels, edge_dim, hidden_channels, act, bias):
        super().__init__(aggr="mean")

        self.mlp_node = Sequential(
            Linear(in_channels, 2 * in_channels, bias=bias),
            act,
            Linear(2 * in_channels, in_channels, bias=bias),
        )

        self.mlp_edge = Sequential(
            Linear(edge_dim, 2 * edge_dim, bias=bias),
            act,
            Linear(2 * edge_dim, edge_dim, bias=bias),
        )

        self.mlp_message = Sequential(
            Linear(2 * in_channels + edge_dim, hidden_channels, bias=bias),
            act,
            Linear(hidden_channels, hidden_channels, bias=bias),
        )

    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr):
        x_i = self.mlp_node(x_i)
        x_j = self.mlp_node(x_j)
        edge_attr = self.mlp_edge(edge_attr)

        m = torch.cat([x_i, x_j, edge_attr], dim=-1)
        return self.mlp_message(m)


class ConcatNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        edge_dim: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int,
        bias: bool,
        act: callable,
        jk: str = "last",
        norm: str = "LayerNorm",
        dropout: float = 0.0,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.edge_dim = edge_dim
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.bias = bias
        self.act = act
        self.jk = jk
        self.norm = norm
        self.dropout = dropout

        self.convs = ModuleList()
        self.norms = ModuleList()

        for i in range(num_layers):
            c_in = in_channels if i == 0 else hidden_channels
            self.convs.append(ConcatConv(c_in, edge_dim, hidden_channels, act, bias))
            self.norms.append(make_norm1d(norm, hidden_channels))

        self.drop = nn.Dropout(dropout)

        assert jk in ("last", "cat")
        self.jk_layer = (
            JumpingKnowledge(jk, channels=hidden_channels, num_layers=num_layers)
            if jk != "last"
            else None
        )

        final_in_dim = hidden_channels if jk == "last" else hidden_channels * num_layers
        self.final_lin = Linear(final_in_dim, out_channels, bias=bias)

    def forward(self, x, edge_index, edge_attr):
        xs = []

        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index, edge_attr)
            x = norm(x)
            x = self.act(x)
            x = self.drop(x)
            xs.append(x)

        if self.jk_layer is not None:
            x = self.jk_layer(xs)
        else:
            x = xs[-1]

        return self.final_lin(x)


class RunningNormalize:
    def __init__(
        self,
        num_channels: int,
        num_steps: int,
        momentum=0.1,
        epsilon=1e-6,
        device="cpu",
    ):
        self.num_channels = num_channels
        self.num_steps = num_steps
        self.num_steps_done = 0
        self.momentum = momentum
        self.epsilon = epsilon
        self.device = device

        # Initialize running mean and variance tensors on device
        self.running_mean = torch.zeros(num_channels, device=device)
        self.running_var = torch.ones(num_channels, device=device)
        self.num_updates = 0

    @torch.no_grad()
    def update(self, x: torch.Tensor):
        if self.num_steps_done >= self.num_steps:
            return

        B, C, H, W = x.shape
        assert C == self.num_channels

        # Flatten batch and spatial dims: (C, B*H*W)
        x_flat = x.permute(1, 0, 2, 3).reshape(C, -1)

        mean = x_flat.mean(dim=1)
        var = x_flat.var(dim=1, unbiased=False)

        if self.num_updates == 0:
            self.running_mean = mean
            self.running_var = var
        else:
            self.running_mean = (
                1 - self.momentum
            ) * self.running_mean + self.momentum * mean
            self.running_var = (
                1 - self.momentum
            ) * self.running_var + self.momentum * var

        self.num_updates += 1
        self.num_steps_done += x.shape[1]

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        mean = self.running_mean.view(1, -1, 1, 1)
        std = (self.running_var + self.epsilon).sqrt().view(1, -1, 1, 1)
        return (x - mean) / std


class SaltAndPepperNoise(Transform):
    def __init__(self, amount=0.05, salt_vs_pepper=0.5):
        super().__init__()
        self.amount = amount
        self.salt_vs_pepper = salt_vs_pepper

    @torch.no_grad()
    def transform(self, inpt: torch.Tensor, params=None):
        batch_size, num_channels, height, width = inpt.shape
        img = inpt.clone()
        num_pixels = height * width

        num_salt = int(self.amount * num_pixels * self.salt_vs_pepper)
        num_pepper = int(self.amount * num_pixels * (1.0 - self.salt_vs_pepper))

        flat_coords = torch.randperm(num_pixels)

        salt_coords = flat_coords[:num_salt]
        img.reshape(batch_size, num_channels, -1)[..., salt_coords] = 1.0

        pepper_coords = flat_coords[num_salt : num_salt + num_pepper]
        img.reshape(batch_size, num_channels, -1)[..., pepper_coords] = 0.0

        return img


class Data(TypedDict):
    images: Tensor
    nodes: Tensor
    edges: Tensor
    edge_index: Tensor
    batch: Tensor
    terminals: Tensor


@torch.no_grad()
def batch_fast(data, device):
    images, nodes, edges, edges_f, terminals, batch, edge_sizes = tuple(
        [] for _ in range(7)
    )

    batch_size = 0

    for x in data:
        im = x.image
        n = x.nodes
        e = x.edges
        f = x.edges_f
        t = x.terminal
        n_nodes = n.shape[0]
        n_edges = e.shape[1]

        images.append(im)
        nodes.append(n)
        edges.append(e)
        edges_f.append(f)
        terminals.append(t)

        batch.append(n_nodes)
        edge_sizes.append(n_edges)

        batch_size += 1

    # is used for both edge offsets and batch index
    batch = torch.as_tensor([0] + batch)

    edge_sizes = torch.tensor(edge_sizes)

    images_batch = torch.cat(images, dim=0)
    nodes_batch = torch.cat(nodes, dim=0)
    edges_f_batch = torch.cat(edges_f, dim=0)
    edges_batch = torch.cat(edges, dim=1)

    terminals_batch = torch.as_tensor(terminals).float().unsqueeze(-1)

    edge_offsets = torch.as_tensor(batch[:-1]).cumsum(0)
    edge_offsets = edge_offsets.repeat_interleave(edge_sizes)
    edge_offsets = edge_offsets.to(edges_batch.device)

    edge_index = edges_batch + edge_offsets

    arange = torch.arange(batch_size)
    batch_batch = arange.repeat_interleave(batch[1:])

    if device is not None:
        images_batch = images_batch.to(device, non_blocking=True)
        nodes_batch = nodes_batch.to(device, non_blocking=True)
        edges_f_batch = edges_f_batch.to(device, non_blocking=True)
        edge_index = edge_index.to(device, non_blocking=True)
        batch_batch = batch_batch.to(device, non_blocking=True)
        terminals_batch = terminals_batch.to(device, non_blocking=True)

    return Data(
        images=images_batch,
        nodes=nodes_batch,
        edges=edges_f_batch,
        edge_index=edge_index,
        terminals=terminals_batch,
        batch=batch_batch,
    )


def identity(x):
    return x


def batch_stream(iterable, n):
    it = iter(iterable)
    while True:
        batch = list(islice(it, n))
        if not batch:
            break
        yield batch


def patch_environment():
    if not socket.gethostname().startswith("jw"):
        return
    os.environ["NCCL_NET"] = "Socket"
    os.environ["GLOO_SOCKET_IFNAME"] = "ib0"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

    # patching the master is necessary
    if "SLURM_JOBID" in os.environ:
        node_list = subprocess.check_output(
            ["scontrol", "show", "hostnames", os.environ["SLURM_NODELIST"]]
        )
        nodes = node_list.decode("utf-8").splitlines()
        os.environ["MASTER_ADDR"] = nodes[0] + "i"


def make_norm1d(norm, channels):
    if norm is None:
        return torch.nn.Identity()

    if norm == "LayerNorm":
        return nn.LayerNorm(channels)

    if norm == "InstanceNorm":
        return torch.nn.InstanceNorm1d(channels)

    if norm == "GraphNorm":
        return GraphNorm(channels)

    raise ValueError(f"unknown normalization layer '{norm}'!")


def make_norm2d(norm, channels):
    if norm is None:
        return torch.nn.Identity()

    if norm == "InstanceNorm":
        return torch.nn.InstanceNorm2d(channels)

    if norm == "GroupNorm":
        return torch.nn.GroupNorm(channels // 8, channels)

    raise ValueError(f"unknown normalization layer '{norm}'!")
